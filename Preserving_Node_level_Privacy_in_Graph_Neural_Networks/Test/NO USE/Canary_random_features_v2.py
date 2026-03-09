import torch
from torch_geometric.data import Data
import os

# ========== 0. 特征管理函数 ==========
def save_canary_features(canaries_data, save_path="./Test/canary_features.pt"):
    """保存Canary特征以便复用（只保存特征张量）"""
    # 提取特征张量
    canaries_feats = torch.cat([feat for feat, _, _ in canaries_data], dim=0)
    torch.save(canaries_feats, save_path)
    print(f"✓ Canary features saved to: {save_path} ({canaries_feats.shape[0]} features)")

def load_canary_features(load_path="./Test/canary_features.pt"):
    """加载已保存的Canary特征"""
    if os.path.exists(load_path):
        canaries_feats = torch.load(load_path)
        print(f"✓ Canary features loaded from: {load_path} ({len(canaries_feats)} features)")
        return canaries_feats
    return None

def create_canary_graph(num_base_canaries, num_repeats):
    # ========== 1. 加载原始子图 ==========
    data = torch.load("amazon_subgraph_all.pt")

    x = data.x.clone()                   # [N, F]
    y = data.y.clone()                   # [N]
    edge_index = data.edge_index.clone() # [2, E]

    num_nodes, feat_dim = x.size()
    num_classes = int(y.max().item()) + 1

    print(f"Original graph: {num_nodes} nodes")

    # ========== 2. Canary 参数 ==========
    num_canaries = num_base_canaries * num_repeats  # 总金丝雀数 = 3000

    # 找到节点最多的类别
    class_counts = torch.bincount(y)
    canary_class = class_counts.argmax().item()
    max_class_count = class_counts[canary_class].item()

    print(f"Using class {canary_class} with {max_class_count} nodes as canary class")

    # 所有该类别节点
    class_idx = (y == canary_class).nonzero(as_tuple=True)[0]
    print(f"Found {len(class_idx)} nodes in class {canary_class}")
    # 处理锚点不足的情况（允许循环选择）
    if len(class_idx) < num_base_canaries:
        print(f"Warning: Only {len(class_idx)} anchor nodes available, will use circular selection")

    # ========== 3. 尝试加载已保存的Canary特征 ==========
    canaries_file = "./Test/canary_features.pt"
    cached_feats = load_canary_features(canaries_file)
    
    if cached_feats is None:
        print(f"Generating new canary features (not found in {canaries_file})...")
        # 生成新特征
        canaries_feats = []
        for i in range(num_base_canaries):
            canary_feat = torch.randn(feat_dim).unsqueeze(0)
            canaries_feats.append(canary_feat)
        canaries_feats = torch.cat(canaries_feats, dim=0)  # [num_base_canaries, feat_dim]
    else:
        print(f"Using cached canary features...")
        # 加载特征后处理数量不匹配的情况
        num_cached = len(cached_feats)
        if num_cached > num_base_canaries:
            print(f"  Warning: Cached {num_cached} features > needed {num_base_canaries}")
            print(f"  Using first {num_base_canaries} features from cache")
            canaries_feats = cached_feats[:num_base_canaries]
        elif num_cached < num_base_canaries:
            print(f"  Warning: Cached {num_cached} features < needed {num_base_canaries}")
            print(f"  Using cached features and generating {num_base_canaries - num_cached} new ones")
            canaries_feats = cached_feats.clone()
            for i in range(num_cached, num_base_canaries):
                canary_feat = torch.randn(feat_dim).unsqueeze(0)
                canaries_feats = torch.cat([canaries_feats, canary_feat], dim=0)
        else:
            canaries_feats = cached_feats
    
    # 确保形状正确 [num_base_canaries, feat_dim]
    if canaries_feats.dim() == 3:
        canaries_feats = canaries_feats.squeeze(1)
    
    # ========== 3b. 构造完整的canaries_data（包含标签和锚点）==========
    x_new = x
    y_new = y
    new_edges = []

    print(f"\n生成 {num_base_canaries} 个基础金丝雀，每个重复 {num_repeats} 次")

    # 转换为完整数据格式
    canaries_data = []
    anchor_selection_idx = 0  # 用于顺序选择锚点
    print("顺序选择邻居:", end=' ')
    for i in range(num_base_canaries):
        # 处理锚点选择（顺序选择）
        if len(class_idx) > 0:
            anchor_node = int(class_idx[anchor_selection_idx % len(class_idx)])
            anchor_selection_idx += 1
        else:
            anchor_node = 0  # 备选方案
        canary_feat = canaries_feats[i:i+1]  # [1, feat_dim]
        canary_label = torch.tensor([canary_class], dtype=y.dtype)
        canaries_data.append((canary_feat, canary_label, anchor_node))

    # 按照重复轮次插入：第1轮插入所有金丝雀的重复0，第2轮插入重复1，等等
    for repeat_idx in range(num_repeats):
        for i in range(num_base_canaries):
            canary_feat, canary_label, anchor_node = canaries_data[i]
            
            # ID: 相同基础金丝雀的重复不相邻
            # 第0轮: 9500, 9501, ..., 10499 (所有金丝雀的重复0)
            # 第1轮: 10500, 10501, ..., 11499 (所有金丝雀的重复1)
            # 第2轮: 11500, 11501, ..., 12499 (所有金丝雀的重复2)
            canary_id = num_nodes + repeat_idx * num_base_canaries + i
            
            # 拼接节点
            x_new = torch.cat([x_new, canary_feat], dim=0)
            y_new = torch.cat([y_new, canary_label], dim=0)
            
            # 【关键变更】边的连接方式
            if repeat_idx == 0:
                # 重复0的金丝雀连接到原图中的锚点节点
                new_edges.append([canary_id, anchor_node])
                new_edges.append([anchor_node, canary_id])
            else:
                # 重复1、2的金丝雀连接到重复0的对应金丝雀
                # 重复0金丝雀i的ID = num_nodes + 0 * num_base_canaries + i
                repeat0_canary_id = num_nodes + 0 * num_base_canaries + i
                new_edges.append([canary_id, repeat0_canary_id])
                new_edges.append([repeat0_canary_id, canary_id])
        
        if repeat_idx == 0:
            print(f"重复{repeat_idx}: ID范围 [{num_nodes}, {num_nodes + num_base_canaries - 1}] -> 连接到原图锚点")
        else:
            print(f"重复{repeat_idx}: ID范围 [{num_nodes + repeat_idx * num_base_canaries}, {num_nodes + (repeat_idx + 1) * num_base_canaries - 1}] -> 连接到重复0")
    
    # ========== 4a. 保存特征（如果是新生成）==========
    if cached_feats is None:
        save_canary_features(canaries_data, canaries_file)

    # ========== 4. 拼接边 ==========
    new_edges = torch.tensor(new_edges, dtype=torch.long).t()
    edge_index_new = torch.cat([edge_index, new_edges], dim=1)

    # ========== 5. 创建自定义 mask（新增）==========
    num_all = x_new.shape[0]
    num_nodes_original = num_nodes

    # 金丝雀分割：
    # - 金丝雀 0-499 的所有3个重复（共1500个节点）为 IN（训练集）
    # - 金丝雀 500-999 的所有3个重复（共1500个节点）为 OUT（测试集）
    num_in_base = num_base_canaries // 2  # 500个基础金丝雀
    num_out_base = num_base_canaries - num_in_base  # 500个基础金丝雀
    num_in_nodes = num_in_base * num_repeats  # 500 * 3 = 1500
    num_out_nodes = num_out_base * num_repeats  # 500 * 3 = 1500

    train_mask = torch.zeros(num_all, dtype=torch.bool)
    val_mask = torch.zeros(num_all, dtype=torch.bool)
    test_mask = torch.zeros(num_all, dtype=torch.bool)

    # 原始节点划分 (80% 训练，20% 测试)
    train_end = int(num_nodes_original * 0.8)
    train_mask[0:train_end] = True
    test_mask[train_end:num_nodes_original] = True  # 【关键】test_mask 只包含原始节点，不包含金丝雀

    # 【关键】设置金丝雀的 IN/OUT
    # 金丝雀 0-499 的所有重复加入训练集
    # 由于ID分配方式：第repeat轮的金丝雀i的ID = num_nodes_original + repeat_idx * num_base_canaries + i
    # 所以需要处理三个分段范围
    for repeat_idx in range(num_repeats):
        # 每个重复轮次中的前500个金丝雀（基础金丝雀0-499）
        start_idx = num_nodes_original + repeat_idx * num_base_canaries
        end_idx = start_idx + num_in_base
        train_mask[start_idx:end_idx] = True

    print(f"\n{'='*60}")
    print("金丝雀配置统计")
    print(f"{'='*60}")
    print(f"基础金丝雀数: {num_base_canaries}")
    print(f"每个金丝雀重复次数: {num_repeats}")
    print(f"总金丝雀数: {num_canaries}")
    print(f"总节点数: {num_all}")
    print(f"原始节点数: {num_nodes_original}")
    print(f"\nIN配置 (金丝雀0-{num_in_base-1}):")
    print(f"  基础金丝雀: {num_in_base} 个")
    print(f"  总节点数: {num_in_nodes} (3个重复)")
    for rep_idx in range(num_repeats):
        start_idx = num_nodes_original + rep_idx * num_base_canaries
        end_idx = start_idx + num_in_base
        if rep_idx == 0:
            print(f"  重复{rep_idx}: ID [{start_idx}, {end_idx-1}] -> 连接到原图")
        else:
            print(f"  重复{rep_idx}: ID [{start_idx}, {end_idx-1}] -> 连接到重复0")
        
    print(f"\nOUT配置 (金丝雀{num_in_base}-{num_base_canaries-1}):")
    print(f"  基础金丝雀: {num_out_base} 个")
    print(f"  总节点数: {num_out_nodes} (3个重复)")
    for rep_idx in range(num_repeats):
        start_idx = num_nodes_original + rep_idx * num_base_canaries + num_in_base
        end_idx = start_idx + num_out_base
        if rep_idx == 0:
            print(f"  重复{rep_idx}: ID [{start_idx}, {end_idx-1}] -> 连接到原图")
        else:
            print(f"  重复{rep_idx}: ID [{start_idx}, {end_idx-1}] -> 连接到重复0")

    print(f"\n训练集大小: {train_mask.sum().item()} (含 {num_in_nodes} 个IN金丝雀)")
    print(f"测试集大小: {test_mask.sum().item()} (仅原始节点，无金丝雀)")
    print(f"{'='*60}")

    # ========== 6. 保存新图（带 mask）==========
    data_new = Data(
        x=x_new,
        edge_index=edge_index_new,
        y=y_new,
        train_mask=train_mask,  # 新增
        val_mask=val_mask,      # 新增
        test_mask=test_mask     # 新增
    )

    torch.save(data_new, "./Test/amazon_subgraph_black.pt")
    print(f"New graph with canaries and masks saved: {data_new.num_nodes} nodes, {data_new.num_edges} edges")
