import torch
from torch_geometric.data import Data
import random
import os
import numpy as np


def save_canary_features(canaries_feats, save_path="./Test/canary_features.pt"):
    """保存Canary特征以便复用（只保存特征张量）"""
    # 如果输入是元组列表，提取特征张量
    if isinstance(canaries_feats, list) and len(canaries_feats) > 0 and isinstance(canaries_feats[0], tuple):
        feats_tensor = torch.cat([feat for feat, _, _ in canaries_feats], dim=0)
    else:
        feats_tensor = canaries_feats
    torch.save(feats_tensor, save_path)
    print(f"✓ Canary features saved to: {save_path} ({feats_tensor.shape[0]} features)")


def load_canary_features(load_path="./Test/canary_features.pt"):
    """加载已保存的Canary特征"""
    if os.path.exists(load_path):
        canaries_feats = torch.load(load_path)
        print(f"✓ Canary features loaded from: {load_path} ({len(canaries_feats)} features)")
        return canaries_feats
    return None


def save_canary_anchors(canaries_anchors, save_path="./Test/canary_anchors.pt"):
    """保存Canary邻居节点标号（随机选择的邻居节点信息）"""
    # 转换为可保存的格式（list -> tensor 或直接保存 list）
    if isinstance(canaries_anchors, list):
        anchors_tensor = torch.tensor(canaries_anchors, dtype=torch.long)
        torch.save(anchors_tensor, save_path)
    else:
        torch.save(canaries_anchors, save_path)
    print(f"✓ Canary anchors saved to: {save_path} ({len(canaries_anchors)} anchors)")


def load_canary_anchors(load_path="./Test/canary_anchors.pt"):
    """加载已保存的Canary邻居节点"""
    if os.path.exists(load_path):
        canaries_anchors = torch.load(load_path)
        # 转换为列表格式以匹配原始格式
        if isinstance(canaries_anchors, torch.Tensor):
            canaries_anchors = canaries_anchors.tolist()
        print(f"✓ Canary anchors loaded from: {load_path} ({len(canaries_anchors)} anchors)")
        return canaries_anchors
    return None


def save_canary_labels(canaries_labels, save_path="./Test/canary_labels.pt"):
    """保存Canary的类别标签"""
    if isinstance(canaries_labels, list):
        labels_tensor = torch.tensor(canaries_labels, dtype=torch.long)
        torch.save(labels_tensor, save_path)
    else:
        torch.save(canaries_labels, save_path)
    print(f"✓ Canary labels saved to: {save_path} ({len(canaries_labels)} labels)")


def load_canary_labels(load_path="./Test/canary_labels.pt"):
    """加载已保存的Canary类别标签"""
    if os.path.exists(load_path):
        canaries_labels = torch.load(load_path)
        if isinstance(canaries_labels, torch.Tensor):
            canaries_labels = canaries_labels.tolist()
        print(f"✓ Canary labels loaded from: {load_path} ({len(canaries_labels)} labels)")
        return canaries_labels
    return None


def get_canary_data(
    num_base_canaries,
    num_nodes,
    feat_dim,
    canary_class,
    y,
    neighbor_class_idx=None,
    canaries_file="./Test/canary_features.pt",
    edge_index=None,
    use_dynamic_degree=False,
    degree_strategy="mean",
    canaries_anchors_file="./Test/canary_anchors.pt",
    canaries_labels_file="./Test/canary_labels.pt",
):
    """获取Canary特征和锚点，支持加载缓存或生成新特征

    修改点（按你的最新要求）：
    - 金丝雀的标签必须与其邻居(锚点)节点标签不同
    - 从“剩余类别(除去邻居类别)”中随机选择一个作为金丝雀标签
    """
    # ===== 1) features =====
    cached_feats = load_canary_features(canaries_file)

    if cached_feats is None:
        print(f"Generating new canary features (not found in {canaries_file})...")
        canaries_feats = []
        for _ in range(num_base_canaries):
            canary_feat = torch.randn(feat_dim).unsqueeze(0)
            canaries_feats.append(canary_feat)
        canaries_feats = torch.cat(canaries_feats, dim=0)
        save_canary_features(canaries_feats, canaries_file)
    else:
        print("Using cached canary features...")
        num_cached = len(cached_feats)
        if num_cached > num_base_canaries:
            print(f"  Warning: Cached {num_cached} features > needed {num_base_canaries}")
            canaries_feats = cached_feats[:num_base_canaries]
        elif num_cached < num_base_canaries:
            print(f"  Warning: Cached {num_cached} features < needed {num_base_canaries}")
            print(f"  Generating {num_base_canaries - num_cached} new ones")
            canaries_feats = cached_feats.clone()
            for _ in range(num_base_canaries - num_cached):
                canary_feat = torch.randn(feat_dim).unsqueeze(0)
                canaries_feats = torch.cat([canaries_feats, canary_feat], dim=0)
        else:
            canaries_feats = cached_feats

    # 确保形状正确
    if canaries_feats.dim() == 3:
        canaries_feats = canaries_feats.squeeze(1)

    # ===== 2) dynamic degree (保留接口；当前 anchors 仍为每个金丝雀 1 个邻居) =====
    if use_dynamic_degree:
        # 如果你确实需要动态度数，请确保外部已实现 compute_dynamic_degree
        dynamic_degree, degree_stats = compute_dynamic_degree(edge_index, num_nodes, strategy=degree_strategy)
        print(f"\n{'='*60}")
        print(f"动态度数计算 (策略: {degree_strategy})")
        print(f"{'='*60}")
        print("原图度数统计:")
        print(f"  最小: {degree_stats['min']:.0f}")
        print(f"  最大: {degree_stats['max']:.0f}")
        print(f"  平均: {degree_stats['mean']:.2f}")
        print(f"  中位数: {degree_stats['median']:.0f}")
        print(f"  25百分位: {degree_stats['percentile_25']:.2f}")
        print(f"  75百分位: {degree_stats['percentile_75']:.2f}")
        print(f"=> Canary节点连接数: {dynamic_degree} 个邻居")
        print(f"{'='*60}\n")
    else:
        dynamic_degree = 1

    # ===== 3) anchors =====
    cached_anchors = load_canary_anchors(canaries_anchors_file)

    if cached_anchors is None:
        print(f"Generating new canary anchors (not found in {canaries_anchors_file})...")
        print("从整个图中随机选择节点作为邻居...")
        canaries_anchors = [torch.randint(num_nodes, (1,)).item() for _ in range(num_base_canaries)]
        save_canary_anchors(canaries_anchors, canaries_anchors_file)
    else:
        print("Using cached canary anchors...")
        canaries_anchors = cached_anchors

        if len(canaries_anchors) > num_base_canaries:
            print(f"  Warning: Cached {len(canaries_anchors)} anchors > needed {num_base_canaries}")
            canaries_anchors = canaries_anchors[:num_base_canaries]
        elif len(canaries_anchors) < num_base_canaries:
            print(f"  Warning: Cached {len(canaries_anchors)} anchors < needed {num_base_canaries}")
            print(f"  Generating {num_base_canaries - len(canaries_anchors)} new ones")
            new_anchors = [torch.randint(num_nodes, (1,)).item() for _ in range(num_base_canaries - len(canaries_anchors))]
            canaries_anchors = canaries_anchors + new_anchors
            save_canary_anchors(canaries_anchors, canaries_anchors_file)

    # ===== 4) labels: must be different from anchor label; random from remaining classes (关键修改) =====
    num_classes = int(y.max().item()) + 1
    all_classes = list(range(num_classes))

    def _anchor_to_label(anchor):
        # 若未来 anchor 变成 list/tuple（多邻居），此处默认取第一个邻居的标签
        if isinstance(anchor, (list, tuple)):
            anchor = anchor[0]
        return int(y[int(anchor)].item())

    def _random_label_not_equal(nei_label: int) -> int:
        candidates = [c for c in all_classes if c != nei_label]
        if not candidates:
            raise ValueError(
                f"No remaining classes to choose from. num_classes={num_classes}, neighbor_label={nei_label}"
            )
        return random.choice(candidates)

    # 依据 anchors 的邻居标签，选择不同的 canary label
    computed_labels = []
    for a in canaries_anchors:
        neighbor_label = _anchor_to_label(a)
        computed_labels.append(_random_label_not_equal(neighbor_label))

    cached_labels = load_canary_labels(canaries_labels_file)
    if cached_labels is None:
        print(f"Generating new canary labels (different from anchor labels) (not found in {canaries_labels_file})...")
        canaries_labels = computed_labels
        save_canary_labels(canaries_labels, canaries_labels_file)
    else:
        # 为确保“与 anchors 不同”，这里做校验：若存在等于邻居标签的情况，或长度不一致，则覆盖缓存
        print("Using cached canary labels... (will verify against anchors)")
        canaries_labels = cached_labels

        need_recompute = False
        if len(canaries_labels) != num_base_canaries:
            print(f"  Warning: Cached {len(canaries_labels)} labels != needed {num_base_canaries}, recomputing.")
            need_recompute = True
        else:
            # 校验：canary_label != neighbor_label
            mism_eq = 0
            for i in range(num_base_canaries):
                neighbor_label = _anchor_to_label(canaries_anchors[i])
                if int(canaries_labels[i]) == int(neighbor_label):
                    mism_eq += 1
            if mism_eq > 0:
                print(f"  Warning: {mism_eq} cached labels equal to anchor labels; overwriting cache.")
                need_recompute = True

        if need_recompute:
            canaries_labels = computed_labels
            save_canary_labels(canaries_labels, canaries_labels_file)

    # ===== 5) pack (feature, label, anchors) =====
    # 当前每个金丝雀仍然只连 1 个邻居，因此这里 anchors 用 [canaries_anchors[i]]
    canaries_data = [
        (canaries_feats[i : i + 1], torch.tensor([canaries_labels[i]], dtype=y.dtype), [canaries_anchors[i]])
        for i in range(num_base_canaries)
    ]
    return canaries_data


def create_canary_graph(num_base_canaries, num_repeats, use_dynamic_degree=False, degree_strategy="mean"):
    # ========== 1. 加载原始子图 ==========
    data = torch.load("amazon_subgraph_all.pt")

    x = data.x.clone()  # [N, F]
    y = data.y.clone()  # [N]
    edge_index = data.edge_index.clone()  # [2, E]

    num_nodes, feat_dim = x.size()
    num_classes = int(y.max().item()) + 1

    print(f"Original graph: {num_nodes} nodes")

    # ========== 2. Canary 参数 ==========
    num_canaries = num_base_canaries * num_repeats

    # 找到节点最多的类别（用作邻居类别）
    class_counts = torch.bincount(y)
    neighbor_class = class_counts.argmax().item()
    max_class_count = class_counts[neighbor_class].item()

    print(f"Using class {neighbor_class} with {max_class_count} nodes as neighbor class (most frequent)")

    # 邻居节点：总是来自最多的类别
    neighbor_class_idx = (y == neighbor_class).nonzero(as_tuple=True)[0]
    print(f"Found {len(neighbor_class_idx)} nodes in neighbor class {neighbor_class}")

    # ========== 金丝雀类别设置（保留变量，但现在不用于 label；label 将与邻居不同并从剩余类别随机选） ==========
    canary_class = 4
    if canary_class >= num_classes or canary_class < 0:
        raise ValueError(f"Invalid canary_class {canary_class}, must be in [0, {num_classes-1}]")
    print(f"Canary class (note: final labels are random but != anchor labels): {canary_class}")

    # ========== 2b. 获取Canary特征和锚点 ==========
    canaries_file = "./Test/canary_features.pt"
    canaries_data = get_canary_data(
        num_base_canaries,
        num_nodes,
        feat_dim,
        canary_class,
        y,
        neighbor_class_idx,
        canaries_file,
        edge_index,
        use_dynamic_degree=use_dynamic_degree,
        degree_strategy=degree_strategy,
        canaries_anchors_file="./Test/canary_anchors.pt",
        canaries_labels_file="./Test/canary_labels.pt",
    )

    # ========== 3. 按照重复轮次插入Canary ==========
    x_new = x
    y_new = y
    new_edges = []

    print(f"\n插入 {num_base_canaries} 个基础金丝雀，每个重复 {num_repeats} 次")

    for repeat_idx in range(num_repeats):
        for i in range(num_base_canaries):
            canary_feat, canary_label, anchor_nodes = canaries_data[i]  # anchor_nodes 是列表(当前长度=1)

            canary_id = num_nodes + repeat_idx * num_base_canaries + i

            x_new = torch.cat([x_new, canary_feat], dim=0)
            y_new = torch.cat([y_new, canary_label], dim=0)

            # 双向边
            for anchor_node in anchor_nodes:
                new_edges.append([canary_id, int(anchor_node)])
                new_edges.append([int(anchor_node), canary_id])

    # ========== 4. 拼接边 ==========
    new_edges = torch.tensor(new_edges, dtype=torch.long).t()
    edge_index_new = torch.cat([edge_index, new_edges], dim=1)

    # ========== 6. 创建自定义 mask ==========
    num_all = x_new.shape[0]
    num_nodes_original = num_nodes

    # 为 num_base_canaries 个“基础金丝雀”生成审计 mask；其重复版本继承该标记
    num_in_base = num_base_canaries // 2
    num_out_base = num_base_canaries - num_in_base

    # 方式：选择前 num_in_base 个作为 IN
    in_base_indices = set(range(num_in_base))

    canary_mask = np.zeros(num_base_canaries, dtype=bool)
    for i in range(num_base_canaries):
        if i in in_base_indices:
            canary_mask[i] = True

    np.save("./Test/canary_mask.npy", canary_mask)
    print(f"✓ Canary mask saved to ./Test/canary_mask.npy (size={num_base_canaries})")
    print(f"  - 前半部分（IN）: {list(sorted(list(in_base_indices))[:10])}... (共 {num_in_base} 个)")
    print(f"  - 后半部分（OUT）: {list(range(num_in_base, num_base_canaries))[:10]}... (共 {num_out_base} 个)")

    train_mask = torch.zeros(num_all, dtype=torch.bool)
    val_mask = torch.zeros(num_all, dtype=torch.bool)
    test_mask = torch.zeros(num_all, dtype=torch.bool)

    # 原始节点划分 (80% 训练，20% 测试)
    train_end = int(num_nodes_original * 0.8)
    train_mask[0:train_end] = True
    test_mask[train_end:num_nodes_original] = True  # test_mask 只包含原始节点，不包含金丝雀

    # 将所有重复版本的 IN 金丝雀加入训练集
    for base_idx in in_base_indices:
        for repeat_idx in range(num_repeats):
            node_id = num_nodes_original + repeat_idx * num_base_canaries + base_idx
            train_mask[node_id] = True

    print(f"\n{'='*60}")
    print("金丝雀配置统计")
    print(f"{'='*60}")
    print(f"基础金丝雀数（用于审计）: {num_base_canaries}")
    print(f"每个金丝雀重复次数（用于训练）: {num_repeats}")
    print(f"总金丝雀数（图中）: {num_canaries}")
    print(f"总节点数: {num_all}")
    print(f"原始节点数: {num_nodes_original}")
    print(f"\n掩码信息（审计用）:")
    print(f"  掩码大小: {num_base_canaries} (仅对原始金丝雀进行审计)")
    print(f"  IN金丝雀: {num_in_base} 个")
    print(f"  OUT金丝雀: {num_out_base} 个")
    print(f"  示例IN索引: {sorted(list(in_base_indices))[:10]}")
    print("\n训练配置:")
    print(f"  图中包含的金丝雀副本: {num_in_base * num_repeats} 个IN + {num_out_base * num_repeats} 个OUT")
    print(f"  训练集大小: {train_mask.sum().item()} (含 {num_in_base * num_repeats} 个IN金丝雀副本)")
    print(f"  测试集大小: {test_mask.sum().item()} (仅原始节点，无金丝雀)")
    print(f"{'='*60}")

    # ========== 7. 保存新图（带 mask）==========
    data_new = Data(
        x=x_new,
        edge_index=edge_index_new,
        y=y_new,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    torch.save(data_new, "./Test/amazon_subgraph_black.pt")
    print(f"New graph with canaries and masks saved: {data_new.num_nodes} nodes")


if __name__ == "__main__":
    # 示例调用
    # create_canary_graph(num_base_canaries=3000, num_repeats=2)
    pass