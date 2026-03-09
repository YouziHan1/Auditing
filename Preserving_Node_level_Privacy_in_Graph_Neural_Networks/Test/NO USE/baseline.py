import torch
from torch_geometric.data import Data
import os
import numpy as np

def create_mislabeled_canary_graph(num_canaries):
    """将最后 num_canaries 个节点进行 mislabel，作为金丝雀
    
    关键特性：
    - 保持所有边不变（连边不改变）
    - 只改变选定节点的标签
    - 分成 IN（前一半，训练集中）和 OUT（后一半，测试集中）
    """
    # ========== 1. 加载原始子图 ==========
    data = torch.load("amazon_subgraph_all.pt")

    x = data.x.clone()                   # [N, F]
    y = data.y.clone()                   # [N]
    edge_index = data.edge_index.clone() # [2, E]

    num_nodes, feat_dim = x.size()
    num_classes = int(y.max().item()) + 1

    print(f"Original graph: {num_nodes} nodes, {num_classes} classes")
    print(f"Total edges: {edge_index.shape[1]}")

    # ========== 2. 检查金丝雀数量 ==========
    if num_canaries > num_nodes:
        raise ValueError(f"num_canaries ({num_canaries}) cannot exceed total nodes ({num_nodes})")
    
    # 最后 num_canaries 个节点为金丝雀
    canary_start_idx = num_nodes - num_canaries
    canary_end_idx = num_nodes

    print(f"\n金丝雀配置:")
    print(f"  总金丝雀数: {num_canaries}")
    print(f"  金丝雀节点范围: [{canary_start_idx}, {canary_end_idx})")
    
    # ========== 3. 对金丝雀节点进行 mislabel ==========
    # 获取原始标签分布
    original_labels = y[canary_start_idx:canary_end_idx].clone()
    print(f"\n  原始标签分布 (前20个): {original_labels[:20].tolist()}")
    
    # 为每个金丝雀节点分配错误标签（随机选择不同的类别）
    mislabeled_count = 0
    for i in range(canary_start_idx, canary_end_idx):
        original_label = y[i].item()
        # 随机选择一个不同的标签
        wrong_labels = [l for l in range(num_classes) if l != original_label]
        wrong_label = wrong_labels[np.random.randint(len(wrong_labels))]
        y[i] = wrong_label
        mislabeled_count += 1
    
    new_y = y
    mislabeled_labels = y[canary_start_idx:canary_end_idx].clone()
    print(f"  Mislabel后标签分布 (前20个): {mislabeled_labels[:20].tolist()}")
    print(f"  ✓ 已对 {mislabeled_count} 个节点进行 mislabel")

    # ========== 4. 创建 mask ==========
    num_all = x.shape[0]
    num_nodes_original = canary_start_idx

    # 金丝雀分割：
    # - 前一半金丝雀（IN）: 加入训练集
    # - 后一半金丝雀（OUT）: 加入测试集
    num_in_canaries = num_canaries // 2
    num_out_canaries = num_canaries - num_in_canaries

    train_mask = torch.zeros(num_all, dtype=torch.bool)
    val_mask = torch.zeros(num_all, dtype=torch.bool)
    test_mask = torch.zeros(num_all, dtype=torch.bool)

    # 原始节点划分 (80% 训练，20% 测试)
    train_end = int(num_nodes_original * 0.8)
    train_mask[0:train_end] = True
    test_mask[train_end:num_nodes_original] = True

    # IN金丝雀（前一半）: 加入训练集
    in_canary_start = canary_start_idx
    in_canary_end = canary_start_idx + num_in_canaries
    train_mask[in_canary_start:in_canary_end] = True
    print(f"\nIN配置 (金丝雀0-{num_in_canaries-1}):")
    print(f"  节点范围: [{in_canary_start}, {in_canary_end})")
    print(f"  所属集合: 训练集")

    # OUT金丝雀（后一半）: 加入测试集
    out_canary_start = in_canary_end
    out_canary_end = canary_end_idx
    test_mask[out_canary_start:out_canary_end] = True
    print(f"\nOUT配置 (金丝雀{num_in_canaries}-{num_canaries-1}):")
    print(f"  节点范围: [{out_canary_start}, {out_canary_end})")
    print(f"  所属集合: 测试集")

    print(f"\n{'='*60}")
    print("统计信息:")
    print(f"  总节点数: {num_all}")
    print(f"  原始节点数: {num_nodes_original}")
    print(f"  金丝雀节点数: {num_canaries}")
    print(f"  训练集大小: {train_mask.sum().item()} (含 {num_in_canaries} 个IN金丝雀)")
    print(f"  测试集大小: {test_mask.sum().item()} (含 {num_out_canaries} 个OUT金丝雀)")
    print(f"  边数: {edge_index.shape[1]} (保持不变)")
    print(f"{'='*60}")

    # ========== 5. 保存新图（带 mask）==========
    data_new = Data(
        x=x,
        edge_index=edge_index,
        y=new_y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    torch.save(data_new, "./Test/amazon_subgraph_black.pt")
    print(f"\n✓ Mislabeled graph saved: {data_new.num_nodes} nodes, {data_new.num_edges} edges")
    print(f"✓ Data file: ./Test/amazon_subgraph_black.pt")
    
    return data_new

create_mislabeled_canary_graph(12000)