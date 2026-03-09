import torch
from torch_geometric.data import Data
import random
import os
import numpy as np
from datetime import datetime


# ========== 0. 特征管理函数 ==========
def compute_dynamic_degree(edge_index, num_nodes, strategy='mean', percentile=50):
    """
    根据原图统计信息动态计算Canary的连接数
    
    Args:
        edge_index: 图的边索引 [2, E]
        num_nodes: 图的节点数
        strategy: 计算策略
                'mean': 使用平均度数（推荐）
                'median': 使用中位数度数
                'percentile': 使用指定百分位数的度数
                'max': 使用最大度数
                'min': 使用最小度数
        percentile: 当strategy='percentile'时使用此参数（0-100）
    
    Returns:
        dynamic_degree: 计算出的Canary连接数（整数）
        degree_stats: 包含度数统计信息的字典
    """
    # 计算度数
    degree = torch.zeros(num_nodes, dtype=torch.long)
    degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.long))
    
    # 获取度数统计信息
    degree_stats = {
        'min': degree.min().item(),
        'max': degree.max().item(),
        'mean': degree.float().mean().item(),
        'median': degree.median().item(),
    }
    
    # 根据策略计算动态度数
    if strategy == 'mean':
        dynamic_degree = max(1, int(round(degree_stats['mean'])))
    elif strategy == 'median':
        dynamic_degree = max(1, int(degree_stats['median']))
    elif strategy == 'percentile':
        percentile_value = torch.quantile(degree.float(), percentile / 100.0).item()
        dynamic_degree = max(1, int(round(percentile_value)))
    elif strategy == 'max':
        dynamic_degree = degree_stats['max']
    elif strategy == 'min':
        dynamic_degree = degree_stats['min']
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # 添加百分位数统计信息
    for p in [25, 50, 75]:
        degree_stats[f'percentile_{p}'] = torch.quantile(degree.float(), p / 100.0).item()
    
    return dynamic_degree, degree_stats


def save_canary_features(canaries_feats, save_path="./Test/canary_features.pt", use_timestamp=True):
    """保存Canary特征以便复用（只保存特征张量）
    
    Args:
        canaries_feats: 特征张量或特征元组列表
        save_path: 保存路径（不含时间戳的基础路径）
        use_timestamp: 是否在文件名中添加时间戳（默认True）
    
    Returns:
        actual_save_path: 实际保存的文件路径（包含时间戳）
    """
    # 如果输入是元组列表，提取特征张量
    if isinstance(canaries_feats, list) and len(canaries_feats) > 0 and isinstance(canaries_feats[0], tuple):
        feats_tensor = torch.cat([feat for feat, _, _ in canaries_feats], dim=0)
    else:
        feats_tensor = canaries_feats
    
    # 如果使用时间戳，在文件名中插入时间戳
    if use_timestamp:
        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # YYYYMMDD_HHMMSS_ms
        # 分离路径和文件名
        dir_path = os.path.dirname(save_path)
        file_name = os.path.basename(save_path)
        # 在文件名中插入时间戳
        name, ext = os.path.splitext(file_name)  # 分离名称和扩展名
        actual_save_path = os.path.join(dir_path, f"{name}_{timestamp}{ext}")
    else:
        actual_save_path = save_path
    
    # 确保目录存在
    dir_path = os.path.dirname(actual_save_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    torch.save(feats_tensor, actual_save_path)
    print(f"✓ Canary features saved to: {actual_save_path} ({feats_tensor.shape[0]} features)")
    
    return actual_save_path

def load_canary_features(load_path="./Test/canary_features.pt"):
    """加载已保存的Canary特征"""
    if os.path.exists(load_path):
        canaries_feats = torch.load(load_path)
        print(f"✓ Canary features loaded from: {load_path} ({len(canaries_feats)} features)")
        return canaries_feats
    return None

def generate_canary_features_orthogonal(num_canaries, feat_dim):
    """
    使用图中所描述的方法生成金丝雀特征（改进版）
    
    步骤1：生成一个正交基 Q
    - 对随机高斯矩阵进行QR分解：Q, R ← QR(N(0,1)^{d×d})
    
    步骤2：生成系数向量 U
    - 采样：U ← N(0,1)^{m×d}
    - 不进行行归一化
    
    步骤3：映射到正交基
    - X = U · Q^T
    
    步骤4：值域约束 [0, 1]
    - 找到X的最小值和最大值
    - X ← (X - min) / (max - min)
    - 确保所有元素都在[0, 1]范围内
    
    Args:
        num_canaries: 要生成的金丝雀数量 m
        feat_dim: 特征维度 d
    
    Returns:
        canaries_feats: 形状为 [num_canaries, feat_dim] 的特征张量
                       保证：所有元素都在[0, 1]范围内，无行归一化约束
    """
    print(f"\n{'='*60}")
    print("生成金丝雀特征 - 使用正交基方法（无行归一化，值域[0,1]）")
    print(f"{'='*60}")
    
    # 第一步：生成一个正交基 Q
    print(f"第一步：生成正交基 Q")
    print(f"  - 对高斯随机矩阵进行QR分解: N(0,1)^{{{feat_dim}×{feat_dim}}}")
    Q_matrix = torch.randn(feat_dim, feat_dim)
    Q, R = torch.linalg.qr(Q_matrix)
    print(f"  ✓ Q 形状: {Q.shape}")
    print(f"  - 验证正交性（Q^T·Q应接近单位矩阵）:")
    Q_test = Q.T @ Q
    print(f"    Q^T·Q 对角线: {torch.diag(Q_test)[:5]}... (应全为1.0)")
    print(f"    Q^T·Q 非对角最大值: {(Q_test.abs() - torch.eye(feat_dim)).abs().max():.6f}")
    
    # 第二步：生成系数向量 U（不进行行归一化）
    print(f"\n第二步：生成系数向量 U（不进行行归一化）")
    print(f"  - 采样 U ← N(0,1)^{{{num_canaries}×{feat_dim}}}")
    U = torch.randn(num_canaries, feat_dim)
    print(f"  ✓ U 形状: {U.shape}")
    U_mean = U.mean()
    U_std = U.std()
    U_min = U.min()
    U_max = U.max()
    print(f"  - U 统计信息:")
    print(f"    • 均值: {U_mean:.6f}")
    print(f"    • 标准差: {U_std:.6f}")
    print(f"    • 最小值: {U_min:.6f}")
    print(f"    • 最大值: {U_max:.6f}")
    print(f"  ✓ 未进行行归一化（行范数不限制）")
    
    # 第三步：映射到正交基
    print(f"\n第三步：映射到正交基")
    print(f"  - 计算 X = U · Q^T")
    canaries_feats = U @ Q.T
    print(f"  ✓ 映射后X的形状: {canaries_feats.shape}")
    
    X_raw_min = canaries_feats.min()
    X_raw_max = canaries_feats.max()
    X_raw_mean = canaries_feats.mean()
    X_raw_std = canaries_feats.std()
    print(f"  - 映射后特征的统计信息 (缩放前):")
    print(f"    • 最小值: {X_raw_min:.6f}")
    print(f"    • 最大值: {X_raw_max:.6f}")
    print(f"    • 均值: {X_raw_mean:.6f}")
    print(f"    • 标准差: {X_raw_std:.6f}")
    
    # 第四步：值域约束 - 保证在[0, 1]范围内
    print(f"\n第四步：值域约束（缩放到[0, 1]）")
    X_min = canaries_feats.min()
    X_max = canaries_feats.max()
    X_range = X_max - X_min
    
    print(f"  - 原始值范围: [{X_min:.6f}, {X_max:.6f}]")
    print(f"  - 范围大小: {X_range:.6f}")
    
    # 线性变换到[0, 1]
    canaries_feats = (canaries_feats - X_min) / X_range
    
    # 验证约束
    X_min_after = canaries_feats.min()
    X_max_after = canaries_feats.max()
    X_mean_after = canaries_feats.mean()
    X_std_after = canaries_feats.std()
    
    print(f"  - 缩放因子: 1/{X_range:.6f}")
    print(f"  - 缩放后值范围: [{X_min_after:.6f}, {X_max_after:.6f}]")
    print(f"  ✓ 约束满足：所有元素都在[0, 1]范围内")
    
    # 最终统计信息
    print(f"\n最终特征统计:")
    print(f"  - 形状: {canaries_feats.shape}")
    print(f"  - 值范围: [{X_min_after:.6f}, {X_max_after:.6f}]")
    print(f"  - 均值: {X_mean_after:.6f}")
    print(f"  - 标准差: {X_std_after:.6f}")
    print(f"  - 最小值: {X_min_after:.6f} (≥ 0.0) ✓" if X_min_after >= 0.0 else f"  ✗ 最小值 < 0")
    print(f"  - 最大值: {X_max_after:.6f} (≤ 1.0) ✓" if X_max_after <= 1.0 else f"  ✗ 最大值 > 1")
    print(f"  - 每个样本的范数分布:")
    row_norms = canaries_feats.norm(dim=1)
    print(f"    • min: {row_norms.min():.4f}")
    print(f"    • max: {row_norms.max():.4f}")
    print(f"    • mean: {row_norms.mean():.4f}")
    print(f"  ✓ 未进行行归一化（行范数无约束）")
    
    print(f"{'='*60}\n")
    
    return canaries_feats


def get_canary_data(num_base_canaries, num_nodes, feat_dim, canary_class, y, neighbor_class_idx=None, canaries_file="./Test/canary_features.pt", edge_index=None, use_dynamic_degree=False, degree_strategy='mean', use_orthogonal_method=True, use_timestamp=True):
    """获取Canary特征和锚点，支持加载缓存或生成新特征
    
    Args:
        canary_class: 金丝雀节点的类别（可指定）
        neighbor_class_idx: 邻居节点所属类别的节点索引（始终来自最多的类别）
        edge_index: 图的边索引，用于计算节点度数（仅在选择高度数节点或动态度数时需要）
        use_dynamic_degree: 是否根据原图平均度数动态调整Canary的连接数
        degree_strategy: 动态度数计算策略 ('mean', 'median', 'percentile', 'max', 'min')
        use_orthogonal_method: 是否使用正交基方法生成特征（默认True）
        use_timestamp: 是否在保存特征文件时使用时间戳（默认True，每次保存为不同的文件）
    """
    #cached_feats = load_canary_features(canaries_file)
    cached_feats=None
    if cached_feats is None:
        print(f"Generating new canary features (not found in {canaries_file})...")
        # 生成新特征
        if use_orthogonal_method:
            print(f"正交化生成金丝雀特征...")
            canaries_feats = generate_canary_features_orthogonal(num_base_canaries, feat_dim)
        else:
            canaries_feats = []
            for i in range(num_base_canaries):
                canary_feat = torch.randn(feat_dim).unsqueeze(0)
                canaries_feats.append(canary_feat)
            canaries_feats = torch.cat(canaries_feats, dim=0)
        # 保存特征
        actual_save_path = save_canary_features(canaries_feats, canaries_file, use_timestamp=use_timestamp)
    else:
        print(f"Using cached canary features...")
        # 处理数量不匹配的情况
        num_cached = len(cached_feats)
        if num_cached > num_base_canaries:
            print(f"  Warning: Cached {num_cached} features > needed {num_base_canaries}")
            canaries_feats = cached_feats[:num_base_canaries]
        elif num_cached < num_base_canaries:
            print(f"  Warning: Cached {num_cached} features < needed {num_base_canaries}")
            print(f"  Generating {num_base_canaries - num_cached} new ones using orthogonal method...")
            canaries_feats = cached_feats.clone()
            new_feats = generate_canary_features_orthogonal(num_base_canaries - num_cached, feat_dim)
            canaries_feats = torch.cat([canaries_feats, new_feats], dim=0)
        else:
            canaries_feats = cached_feats
    
    # 确保形状正确
    if canaries_feats.dim() == 3:
        canaries_feats = canaries_feats.squeeze(1)
    
    # 为每个金丝雀生成锚点
    # ===== 选择邻居方式 =====
    '''方式1: 从整个图中随机选择节点作为邻居'''
    print(f"Selecting canary anchors randomly from all nodes...从整个图中随机选择节点作为邻居")
    canaries_anchors = [torch.randint(num_nodes, (1,)).item() for _ in range(num_base_canaries)]
    
    #'''方式2: 从指定类别中的前num_base_canaries个节点作为邻居（不循环，若不足则循环补充）'''
    # print(f"Selecting canary anchors from specified class nodes...从指定类别中的前num_base_canaries个节点作为邻居")
    # canaries_anchors = [int(neighbor_class_idx[i % len(neighbor_class_idx)]) for i in range(num_base_canaries)]
    
    '''方式3: 从指定类别中随机选择节点作为邻居'''
    # print(f"Selecting canary anchors from specified class nodes (random)...从指定类别中随机选择节点作为邻居")
    # canaries_anchors = [int(neighbor_class_idx[random.randint(0, len(neighbor_class_idx)-1)]) for i in range(num_base_canaries)]
    
    '''方式4: 从指定类别中选择度数高的节点作为邻居'''
    # print(f"Selecting canary anchors from specified class nodes (high degree)...")
    # # 计算每个节点的度数
    # degree = torch.zeros(num_nodes, dtype=torch.long)
    # degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.long))
    # # 获取class_idx中节点的度数
    # class_degrees = degree[class_idx]
    # # 按度数排序，选择度数最高的num_base_canaries个节点
    # _, sorted_indices = torch.sort(class_degrees, descending=False)
    # # 取前num_base_canaries个节点（如果不够则循环补充）
    # high_degree_nodes = []
    # for i in range(num_base_canaries):
    #    idx = sorted_indices[i % len(sorted_indices)].item()
    #    high_degree_nodes.append(int(class_idx[idx]))
    # canaries_anchors = high_degree_nodes
    # # 打印度数统计信息
    # print(f"  Degree range in class: min={class_degrees.min()}, max={class_degrees.max()}, mean={class_degrees.float().mean():.2f}")
    # print(f"  Selected {num_base_canaries} nodes with degrees: min={degree[high_degree_nodes].min()}, max={degree[high_degree_nodes].max()}")
    # ========== 方式 5: 从指定类别中随机选择两个节点作为邻居 ==========
    # print(f"Selecting 2 random canary anchors per canary from specified class nodes...")
    # canaries_anchors = []
    # class_idx_list = class_idx.tolist() # 转换为 list 方便 random.sample
    # 
    # for _ in range(num_base_canaries):
    #     # 随机抽取两个不重复的节点作为锚点
    #     # 如果该类节点数少于2，则允许重复
    #     if len(class_idx_list) >= 2:
    #         pair = random.sample(class_idx_list, 2)
    #     else:
    #         pair = [random.choice(class_idx_list) for _ in range(3)]
    #     canaries_anchors.append(pair)

    print("随机选择节点类型")
    canaries_data = [(canaries_feats[i:i+1], torch.tensor([random.randint(0, 9)], dtype=y.dtype), [canaries_anchors[i]]) 
                     for i in range(num_base_canaries)]
    
    return canaries_data





def create_canary_graph(num_base_canaries, num_repeats, use_dynamic_degree=False, degree_strategy='mean'):
    # ========== 1. 加载原始子图 ==========
    data = torch.load("amazon_subgraph_all.pt")

    x = data.x.clone()                   # [N, F]
    y = data.y.clone()                   # [N]
    edge_index = data.edge_index.clone() # [2, E]

    num_nodes, feat_dim = x.size()
    num_classes = int(y.max().item()) + 1

    print(f"Original graph: {num_nodes} nodes")

    # ========== 2. Canary 参数 ==========
    num_canaries = num_base_canaries * num_repeats  # 总金丝雀数 = 6000

    # 找到节点最多的类别（用作邻居类别）
    class_counts = torch.bincount(y)
    neighbor_class = class_counts.argmax().item()
    max_class_count = class_counts[neighbor_class].item()

    print(f"Using class {neighbor_class} with {max_class_count} nodes as neighbor class (most frequent)")

    # 邻居节点：总是来自最多的类别
    neighbor_class_idx = (y == neighbor_class).nonzero(as_tuple=True)[0]
    print(f"Found {len(neighbor_class_idx)} nodes in neighbor class {neighbor_class}")

    # ========== 金丝雀类别设置（在这里修改 canary_class） ==========
    # 可以修改下面的值来指定金丝雀节点的类型
    canary_class = 4  # 默认与邻居类型相同，可修改为其他类型（如 0, 1, 2 等）
    
    if canary_class >= num_classes or canary_class < 0:
        raise ValueError(f"Invalid canary_class {canary_class}, must be in [0, {num_classes-1}]")
    
    print(f"Canary class: {canary_class}")

    # ========== 2b. 获取Canary特征和锚点 ==========
    canaries_file = "./Test/canary_features.pt"
    canaries_data = get_canary_data(num_base_canaries, num_nodes, feat_dim, canary_class, y, neighbor_class_idx, canaries_file, edge_index, use_dynamic_degree=use_dynamic_degree, degree_strategy=degree_strategy, use_orthogonal_method=True, use_timestamp=True)

    # ========== 3. 按照重复轮次插入Canary ==========
    x_new = x
    y_new = y
    new_edges = []

    print(f"\n插入 {num_base_canaries} 个基础金丝雀，每个重复 {num_repeats} 次")

    # for repeat_idx in range(num_repeats):
    #     for i in range(num_base_canaries):
    #         canary_feat, canary_label, anchor_node = canaries_data[i]
            
    #         # ID: 相同基础金丝雀的重复不相邻
    #         # 第0轮: 9500, 9501, ..., 10499 (所有金丝雀的重复0)
    #         # 第1轮: 10500, 10501, ..., 11499 (所有金丝雀的重复1)
    #         # 第2轮: 11500, 11501, ..., 12499 (所有金丝雀的重复2)
    #         canary_id = num_nodes + repeat_idx * num_base_canaries + i
            
    #         # 拼接节点
    #         x_new = torch.cat([x_new, canary_feat], dim=0)
    #         y_new = torch.cat([y_new, canary_label], dim=0)
            
    #         # 双向边
    #         new_edges.append([canary_id, anchor_node])
    #         new_edges.append([anchor_node, canary_id])
        
    #     if repeat_idx == 0:
    #         print(f"重复{repeat_idx}: ID范围 [{num_nodes}, {num_nodes + num_base_canaries - 1}]")
    # # 在 create_canary_graph 函数的循环内：
    for repeat_idx in range(num_repeats):
        for i in range(num_base_canaries):
            canary_feat, canary_label, anchor_nodes = canaries_data[i] # 这里 anchor_nodes 是列表
            
            canary_id = num_nodes + repeat_idx * num_base_canaries + i
            
            x_new = torch.cat([x_new, canary_feat], dim=0)
            y_new = torch.cat([y_new, canary_label], dim=0)
            
            # 为两个锚点分别建立双向边
            for anchor_node in anchor_nodes:
                new_edges.append([canary_id, int(anchor_node)])
                new_edges.append([int(anchor_node), canary_id])
    
    # # ========== 3.5. 相同金丝雀的重复节点相互连接（形成集体）==========
    # print(f"\n连接相同金丝雀的重复节点，形成集体...")
    # for i in range(num_base_canaries):
    #     # 获取该基础金丝雀的所有重复节点的ID
    #     canary_ids = [num_nodes + repeat_idx * num_base_canaries + i for repeat_idx in range(num_repeats)]
        
    #     # 创建这些节点之间的双向边（完全连接）
    #     for j in range(len(canary_ids)):
    #         for k in range(j + 1, len(canary_ids)):
    #             new_edges.append([canary_ids[j], canary_ids[k]])
    #             new_edges.append([canary_ids[k], canary_ids[j]])
    
    #print(f"✓ 已为 {num_base_canaries} 个基础金丝雀的重复节点之间创建集体连接")
    
    # ========== 4. 拼接边 ==========
    new_edges = torch.tensor(new_edges, dtype=torch.long).t()
    edge_index_new = torch.cat([edge_index, new_edges], dim=1)

    # ========== 6. 创建自定义 mask（新增）==========
    num_all = x_new.shape[0]
    num_nodes_original = num_nodes

    # ⚠️  正确方式：为3000个原始金丝雀生成掩码，重复的版本继承原始的掩码标记
    num_in_base = num_base_canaries // 2  # 1500个原始金丝雀作为IN
    num_out_base = num_base_canaries - num_in_base  # 1500个原始金丝雀作为OUT
    
    # 步骤1：前半部分为IN，后半部分为OUT
    # 选择前 num_in_base 个原始金丝雀索引作为IN（连续索引，易于调试）
    in_base_indices = set(range(num_in_base))  # [0, 1, 2, ..., num_in_base-1]
    
    #随机选择 num_in_base 个原始金丝雀索引作为IN（随机索引，更真实）
    #in_base_indices = set(random.sample(range(num_base_canaries), num_in_base))
    
    # 步骤2：创建掩码（大小为原始金丝雀数，而不是总金丝雀数）
    canary_mask = np.zeros(num_base_canaries, dtype=bool)
    for i in range(num_base_canaries):
        if i in in_base_indices:
            canary_mask[i] = True
    
    # 保存canary_mask到数据中
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

    # 步骤3：将所有重复版本的IN金丝雀加入训练集（保持与原始版本一致）
    for base_idx in in_base_indices:
        # 该原始金丝雀的所有重复版本ID
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
    print(f"\n训练配置:")
    print(f"  图中包含的金丝雀副本: {num_in_base * num_repeats} 个IN + {num_out_base * num_repeats} 个OUT")
    print(f"  训练集大小: {train_mask.sum().item()} (含 {num_in_base * num_repeats} 个IN金丝雀副本)")
    print(f"  测试集大小: {test_mask.sum().item()} (仅原始节点，无金丝雀)")
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
    print(f"New graph with canaries and masks saved: {data_new.num_nodes} nodes")

