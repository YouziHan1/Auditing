#!/usr/bin/env python3
"""
Analyze canary artifacts in Test/BEST and explain why audit performance is strong.

Focus:
1) Neighbor construction and graph connectivity of canaries
2) Canary feature statistics vs original nodes
3) Canary label assignment patterns
4) IN/OUT selection validity and train-mask consistency
"""
# python3 Test/BEST/analyze_best_canaries.py \
#   --best-dir ./Test/BEST \
#   --graph-file amazon_subgraph_black_0.9878.pt
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return int(default)


def _load_pt(path: Path):
    return torch.load(path, map_location="cpu")


def _normalize_anchors(raw_anchors) -> List[List[int]]:
    """Normalize anchors into shape: List[num_base][k_neighbors]."""
    if isinstance(raw_anchors, torch.Tensor):
        raw_anchors = raw_anchors.tolist()

    anchors: List[List[int]] = []
    if isinstance(raw_anchors, list):
        for item in raw_anchors:
            if isinstance(item, torch.Tensor):
                item = item.tolist()
            if isinstance(item, (list, tuple)):
                anchors.append([_safe_int(v) for v in item])
            else:
                anchors.append([_safe_int(item)])
    else:
        raise TypeError(f"Unsupported anchor format: {type(raw_anchors)}")

    return anchors


def _normalize_labels(raw_labels) -> np.ndarray:
    if isinstance(raw_labels, torch.Tensor):
        raw_labels = raw_labels.detach().cpu().numpy()
    labels = np.asarray(raw_labels).reshape(-1)
    return labels.astype(np.int64, copy=False)


def _normalize_features(raw_feats) -> torch.Tensor:
    if not isinstance(raw_feats, torch.Tensor):
        raw_feats = torch.as_tensor(raw_feats)
    feats = raw_feats.detach().cpu()
    if feats.dim() == 3 and feats.shape[1] == 1:
        feats = feats.squeeze(1)
    return feats.float()


def _build_out_neighbors(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    nbrs: List[List[int]] = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for s, d in zip(src, dst):
        if 0 <= s < num_nodes and 0 <= d < num_nodes:
            nbrs[s].append(d)
    return nbrs


def _infer_layout(data, num_base_canaries: int, num_in_base: int) -> Dict[str, int]:
    """
    Infer original node count and repeats from masks and total node count.

    Generation logic in Canary_Compare.py:
    - test_mask contains only original nodes (20% of original)
    - canary nodes are appended at tail: num_base_canaries * num_repeats
    """
    total_nodes = int(data.num_nodes)
    test_count = int(data.test_mask.sum().item()) if hasattr(data, "test_mask") else 0

    # Primary estimate from 20% test split.
    original_est = test_count * 5 if test_count > 0 else total_nodes
    canary_total = total_nodes - original_est

    if num_base_canaries <= 0:
        raise ValueError("num_base_canaries must be > 0")

    repeats = canary_total // num_base_canaries if canary_total > 0 else 0

    # Fallback if strict arithmetic fails.
    if canary_total < 0 or (canary_total % num_base_canaries != 0):
        # Try to infer with train mask relation:
        # train = floor(0.8 * original) + num_in_base * repeats
        train_count = int(data.train_mask.sum().item()) if hasattr(data, "train_mask") else 0
        best = None
        for r in range(1, 500):
            orig = total_nodes - r * num_base_canaries
            if orig <= 0:
                break
            expected_train = int(orig * 0.8) + num_in_base * r
            expected_test = orig - int(orig * 0.8)
            err = abs(expected_train - train_count) + abs(expected_test - test_count)
            if best is None or err < best[0]:
                best = (err, orig, r)
        if best is None:
            raise RuntimeError("Failed to infer original node count/repeats")
        _, original_est, repeats = best
        canary_total = total_nodes - original_est

    canary_start = original_est
    return {
        "total_nodes": total_nodes,
        "original_nodes": original_est,
        "canary_total": canary_total,
        "num_repeats": repeats,
        "canary_start": canary_start,
    }


def _summary_stats(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def analyze(best_dir: Path, graph_file: str) -> Dict:
    graph_path = best_dir / graph_file
    feats_path = best_dir / "canary_features.pt"
    anchors_path = best_dir / "canary_anchors.pt"
    labels_path = best_dir / "canary_labels.pt"
    mask_path = best_dir / "canary_mask.npy"

    for p in [graph_path, feats_path, anchors_path, labels_path, mask_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    data = _load_pt(graph_path)
    raw_feats = _load_pt(feats_path)
    raw_anchors = _load_pt(anchors_path)
    raw_labels = _load_pt(labels_path)
    canary_mask = np.load(mask_path)

    canary_feats = _normalize_features(raw_feats)
    canary_anchors = _normalize_anchors(raw_anchors)
    canary_labels = _normalize_labels(raw_labels)
    canary_mask = canary_mask.astype(bool)

    num_base = int(len(canary_mask))
    if num_base == 0:
        raise ValueError("canary_mask is empty")

    if canary_feats.shape[0] != num_base:
        raise ValueError(
            f"Feature count mismatch: feats={canary_feats.shape[0]} vs mask={num_base}"
        )
    if len(canary_anchors) != num_base:
        raise ValueError(
            f"Anchor count mismatch: anchors={len(canary_anchors)} vs mask={num_base}"
        )
    if len(canary_labels) != num_base:
        raise ValueError(
            f"Label count mismatch: labels={len(canary_labels)} vs mask={num_base}"
        )

    num_in = int(canary_mask.sum())
    num_out = int((~canary_mask).sum())

    layout = _infer_layout(data, num_base_canaries=num_base, num_in_base=num_in)
    canary_start = layout["canary_start"]
    repeats = layout["num_repeats"]

    x = data.x.detach().cpu().float()
    y = data.y.detach().cpu().long() if hasattr(data, "y") else None
    edge_index = data.edge_index.detach().cpu().long()
    train_mask = data.train_mask.detach().cpu().bool() if hasattr(data, "train_mask") else None
    test_mask = data.test_mask.detach().cpu().bool() if hasattr(data, "test_mask") else None

    neighbors = _build_out_neighbors(edge_index, layout["total_nodes"])

    # Neighbor and structure analysis.
    canary_degrees = []
    anchor_hit_ratio = []
    anchor_class_match = []
    per_canary_unique_nbrs = []

    in_train_all_repeats_ok = 0
    out_train_leak_count = 0

    for base_idx in range(num_base):
        expected_anchors = set(_safe_int(v) for v in canary_anchors[base_idx])
        repeated_ids = [canary_start + r * num_base + base_idx for r in range(repeats)]

        this_anchor_hits = []
        this_deg = []
        this_unique_nbrs = set()
        this_train_flags = []

        for cid in repeated_ids:
            if cid < 0 or cid >= layout["total_nodes"]:
                continue
            nbr = neighbors[cid]
            nbr_set = set(nbr)
            this_deg.append(len(nbr))
            this_unique_nbrs |= nbr_set

            if len(expected_anchors) == 0:
                this_anchor_hits.append(1.0)
            else:
                this_anchor_hits.append(len(expected_anchors & nbr_set) / max(1, len(expected_anchors)))

            if train_mask is not None:
                this_train_flags.append(bool(train_mask[cid].item()))

        canary_degrees.extend(this_deg)
        per_canary_unique_nbrs.append(len(this_unique_nbrs))
        anchor_hit_ratio.append(float(np.mean(this_anchor_hits)) if this_anchor_hits else 0.0)

        # Label vs anchor-class agreement.
        if y is not None and len(expected_anchors) > 0:
            valid_anchors = [a for a in expected_anchors if 0 <= a < layout["original_nodes"]]
            if valid_anchors:
                anchor_cls = y[torch.tensor(valid_anchors, dtype=torch.long)]
                maj_cls = int(torch.mode(anchor_cls).values.item())
                anchor_class_match.append(int(maj_cls == int(canary_labels[base_idx])))

        # IN/OUT training consistency.
        if train_mask is not None and this_train_flags:
            if canary_mask[base_idx]:
                in_train_all_repeats_ok += int(all(this_train_flags))
            else:
                out_train_leak_count += int(any(this_train_flags))

    canary_degree_stats = _summary_stats(np.asarray(canary_degrees, dtype=np.float64))
    anchor_hit_stats = _summary_stats(np.asarray(anchor_hit_ratio, dtype=np.float64))
    uniq_nbr_stats = _summary_stats(np.asarray(per_canary_unique_nbrs, dtype=np.float64))

    # Feature analysis.
    original_x = x[: layout["original_nodes"]]
    feat_mean = original_x.mean(dim=0)
    feat_std = original_x.std(dim=0).clamp(min=1e-6)

    z = (canary_feats - feat_mean) / feat_std
    canary_l2_z = torch.norm(z, p=2, dim=1).numpy()

    original_z_l2 = torch.norm((original_x - feat_mean) / feat_std, p=2, dim=1).numpy()

    in_idx = np.where(canary_mask)[0]
    out_idx = np.where(~canary_mask)[0]

    in_l2 = canary_l2_z[in_idx] if len(in_idx) > 0 else np.array([])
    out_l2 = canary_l2_z[out_idx] if len(out_idx) > 0 else np.array([])

    # Label distribution analysis.
    unique_labels, label_counts = np.unique(canary_labels, return_counts=True)
    label_dist = {int(k): int(v) for k, v in zip(unique_labels, label_counts)}

    # IN/OUT mask pattern analysis.
    first_half_true = bool(np.all(canary_mask[: num_base // 2])) if num_base >= 2 else bool(canary_mask[0])
    second_half_false = bool(np.all(~canary_mask[num_base // 2 :])) if num_base >= 2 else True
    contiguous_split = first_half_true and second_half_false

    # If labels are available for original graph, estimate neighbor majority class concentration.
    neighbor_class_concentration = None
    if y is not None:
        anchor_major_classes = []
        for alist in canary_anchors:
            valid = [a for a in alist if 0 <= a < layout["original_nodes"]]
            if not valid:
                continue
            cls = y[torch.tensor(valid, dtype=torch.long)]
            anchor_major_classes.append(int(torch.mode(cls).values.item()))
        if anchor_major_classes:
            vals, cnts = np.unique(anchor_major_classes, return_counts=True)
            top_idx = int(np.argmax(cnts))
            neighbor_class_concentration = {
                "dominant_class": int(vals[top_idx]),
                "dominant_ratio": float(cnts[top_idx] / len(anchor_major_classes)),
                "num_unique_anchor_major_classes": int(len(vals)),
            }

    # Build explanation of strong audit result.
    reasons: List[str] = []

    if contiguous_split:
        reasons.append(
            "IN/OUT 按基础金丝雀顺序严格二分（前半 IN、后半 OUT），标签真值无噪声，审计监督信号干净。"
        )

    if in_train_all_repeats_ok == num_in and out_train_leak_count == 0:
        reasons.append(
            "训练成员关系非常干净：IN 的所有重复副本都在训练集，OUT 没有训练泄漏。"
        )
    else:
        reasons.append(
            "IN/OUT 与训练集关系存在不一致，可能削弱审计表现（建议先修复该问题）。"
        )

    if anchor_hit_stats["mean"] > 0.95:
        reasons.append(
            "图结构注入稳定：canary 节点与期望锚点匹配率高，训练时可重复观察到一致拓扑信号。"
        )

    if neighbor_class_concentration and neighbor_class_concentration["dominant_ratio"] > 0.8:
        reasons.append(
            "锚点类别高度集中，canary 邻域语义一致，容易形成可审计的成员/非成员分布差异。"
        )

    gap = _safe_float(np.mean(in_l2) - np.mean(out_l2), 0.0)
    if abs(gap) < 1e-5:
        reasons.append(
            "IN/OUT 的特征分布基本一致，这本身不会制造伪信号；审计优势更可能来自训练成员关系与结构注入的一致性。"
        )
    else:
        reasons.append(
            "IN/OUT 在标准化特征距离上存在偏移，可能进一步放大可分性。"
        )

    report = {
        "paths": {
            "graph": str(graph_path),
            "canary_features": str(feats_path),
            "canary_anchors": str(anchors_path),
            "canary_labels": str(labels_path),
            "canary_mask": str(mask_path),
        },
        "layout": layout,
        "in_out": {
            "num_base_canaries": num_base,
            "num_in": num_in,
            "num_out": num_out,
            "contiguous_split_first_half_in": contiguous_split,
        },
        "train_consistency": {
            "in_all_repeats_in_train": int(in_train_all_repeats_ok),
            "in_total": int(num_in),
            "out_with_any_train_leak": int(out_train_leak_count),
        },
        "neighbor_analysis": {
            "canary_degree_stats": canary_degree_stats,
            "anchor_hit_ratio_stats": anchor_hit_stats,
            "unique_neighbor_count_per_base_canary": uniq_nbr_stats,
            "label_matches_anchor_majority_ratio": (
                float(np.mean(anchor_class_match)) if anchor_class_match else None
            ),
            "neighbor_class_concentration": neighbor_class_concentration,
        },
        "feature_analysis": {
            "canary_l2_z_stats": _summary_stats(canary_l2_z),
            "original_l2_z_stats": _summary_stats(original_z_l2),
            "in_l2_z_stats": _summary_stats(in_l2),
            "out_l2_z_stats": _summary_stats(out_l2),
            "in_minus_out_mean_l2_z": gap,
        },
        "label_distribution": {
            "counts": label_dist,
            "num_unique_labels": int(len(label_dist)),
        },
        "why_audit_good": reasons,
    }

    return report


def print_report(report: Dict):
    print("=" * 80)
    print("BEST Canary Audit Analysis")
    print("=" * 80)

    layout = report["layout"]
    in_out = report["in_out"]
    train = report["train_consistency"]
    nbr = report["neighbor_analysis"]
    feat = report["feature_analysis"]

    print("\n[Layout]")
    print(f"Total nodes: {layout['total_nodes']}")
    print(f"Original nodes (inferred): {layout['original_nodes']}")
    print(f"Total canary nodes (all repeats): {layout['canary_total']}")
    print(f"Base canaries: {in_out['num_base_canaries']}")
    print(f"Repeats per base canary: {layout['num_repeats']}")

    print("\n[IN/OUT Selection]")
    print(f"IN: {in_out['num_in']}, OUT: {in_out['num_out']}")
    print(f"Contiguous split (first half IN): {in_out['contiguous_split_first_half_in']}")

    print("\n[Train Consistency]")
    print(
        f"IN with all repeats in train: {train['in_all_repeats_in_train']} / {train['in_total']}"
    )
    print(f"OUT with any train leakage: {train['out_with_any_train_leak']}")

    print("\n[Neighbor Analysis]")
    print(
        "Anchor hit ratio mean/std: "
        f"{nbr['anchor_hit_ratio_stats']['mean']:.4f} / {nbr['anchor_hit_ratio_stats']['std']:.4f}"
    )
    print(
        "Canary degree mean/std: "
        f"{nbr['canary_degree_stats']['mean']:.2f} / {nbr['canary_degree_stats']['std']:.2f}"
    )
    if nbr["neighbor_class_concentration"] is not None:
        cc = nbr["neighbor_class_concentration"]
        print(
            f"Anchor-major class concentration: class={cc['dominant_class']}, "
            f"ratio={cc['dominant_ratio']:.3f}, unique_major_classes={cc['num_unique_anchor_major_classes']}"
        )

    print("\n[Feature Analysis]")
    print(
        "Canary L2(z) mean/std: "
        f"{feat['canary_l2_z_stats']['mean']:.4f} / {feat['canary_l2_z_stats']['std']:.4f}"
    )
    print(
        "IN minus OUT mean L2(z): "
        f"{feat['in_minus_out_mean_l2_z']:.6f}"
    )

    print("\n[Label Distribution]")
    print(f"Unique canary labels: {report['label_distribution']['num_unique_labels']}")
    print(f"Counts: {report['label_distribution']['counts']}")

    print("\n[Why Audit Result Is Strong]")
    for idx, reason in enumerate(report["why_audit_good"], start=1):
        print(f"{idx}. {reason}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze BEST canary artifacts for audit-quality explanation")
    parser.add_argument(
        "--best-dir",
        type=str,
        default="./Test/BEST",
        help="Directory containing BEST canary files",
    )
    parser.add_argument(
        "--graph-file",
        type=str,
        default="amazon_subgraph_black_0.9878.pt",
        help="Graph file name under best-dir",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="canary_analysis_report.json",
        help="Output JSON file name under best-dir",
    )
    args = parser.parse_args()

    best_dir = Path(args.best_dir)
    report = analyze(best_dir, graph_file=args.graph_file)
    print_report(report)

    out_path = best_dir / args.save_json
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved JSON report to: {out_path}")


if __name__ == "__main__":
    main()
