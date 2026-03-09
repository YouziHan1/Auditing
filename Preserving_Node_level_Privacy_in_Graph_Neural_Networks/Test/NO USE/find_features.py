import torch
from torch_geometric.data import Data
import random
import os
import numpy as np
import time
from datetime import datetime

# ========== 0. 稠密子图识别函数 ==========
def find_dense_subgraphs(edge_index, num_nodes, method='k_core', top_k=500, k_value=None):
    """
    识别原图中的稠密子图节点
    
    Args:
        edge_index: 图的边索引 [2, E]
        num_nodes: 图的节点数
        method: 稠密子图识别方法
                'k_core': 基于K-core分解（保留度数最高的K-core）
                'local_density': 基于节点度数和聚集系数的混合分数
                'degree': 基于度数排序
        top_k: 返回的稠密子图节点数
        k_value: K-core的k值，如果为None则自动选择最大k值
    
    Returns:
        dense_nodes: 稠密子图中的节点索引张量
    """
    
    if method == 'k_core':
        # K-core分解：递归删除度数<=k的节点
        dense_nodes = _k_core_decomposition(edge_index, num_nodes, k_value)
        
    elif method == 'local_density':
        # 基于局部密度的混合得分：degree * clustering_coefficient
        dense_nodes = _local_density_scoring(edge_index, num_nodes, top_k)
        
    elif method == 'degree':
        # 简单的度数排序
        degree = torch.zeros(num_nodes, dtype=torch.long)
        degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.long))
        _, top_indices = torch.topk(degree, min(top_k, num_nodes))
        dense_nodes = top_indices
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return dense_nodes[:min(len(dense_nodes), top_k)]


def _k_core_decomposition(edge_index, num_nodes, k_value=None):
    """
    K-core分解：找到最大k-core（或指定k值的k-core）的所有节点
    """
    # 计算初始度数
    degree = torch.zeros(num_nodes, dtype=torch.long)
    degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.long))
    
    # 迭代删除度数<=k的节点，直到稳定
    remaining_nodes = set(range(num_nodes))
    core_numbers = torch.zeros(num_nodes, dtype=torch.long)
    current_degree = degree.clone()
    
    # 从低度数开始剥离
    for k in range(1, int(degree.max().item()) + 1):
        # 找到度数<=k且还未处理的节点
        to_remove = []
        for node in remaining_nodes:
            if current_degree[node] <= k:
                to_remove.append(node)
                core_numbers[node] = k
        
        if not to_remove:
            continue
        
        # 更新度数
        for node in to_remove:
            remaining_nodes.discard(node)
            # 减少邻居的度数
            edge_mask = edge_index[0] == node
            neighbors = edge_index[1][edge_mask]
            current_degree[neighbors] -= 1
    
    # 剩余节点的core number是最高的
    max_k = int(core_numbers[list(remaining_nodes)].max().item()) if remaining_nodes else 0
    
    if k_value is None:
        k_value = max(1, max_k - 1)  # 取近似的最高k-core
    
    # 返回k值>=k_value的所有节点
    dense_nodes = torch.where(core_numbers >= k_value)[0]
    
    if len(dense_nodes) == 0:
        # 如果没有节点满足条件，返回高度节点
        dense_nodes = torch.where(degree >= degree.quantile(0.75))[0]
    
    print(f"  K-core分解: 最高core={max_k}, 选择k>={k_value}的节点, 共{len(dense_nodes)}个")
    
    return dense_nodes


def _local_density_scoring(edge_index, num_nodes, top_k=500):
    """
    基于节点的局部密度评分：度数 * 局部聚集系数
    这样既考虑节点的连接性，也考虑其邻域的紧密性
    """
    # 计算度数
    degree = torch.zeros(num_nodes, dtype=torch.long)
    degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.long))
    
    # 计算聚集系数
    clustering_coeff = _compute_clustering_coefficient(edge_index, num_nodes)
    
    # 组合评分：度数 * 聚集系数 * 正规化因子
    scores = degree.float() * clustering_coeff
    
    # 选择top_k个节点
    _, top_indices = torch.topk(scores, min(top_k, num_nodes))
    
    print(f"  局部密度得分: 选择top{min(top_k, num_nodes)}个节点")
    print(f"    度数范围: [{degree.min()}, {degree.max()}]")
    print(f"    聚集系数范围: [{clustering_coeff.min():.3f}, {clustering_coeff.max():.3f}]")
    print(f"    综合得分范围: [{scores.min():.3f}, {scores.max():.3f}]")
    
    return top_indices


def _compute_clustering_coefficient(edge_index, num_nodes):
    """
    计算每个节点的聚集系数
    聚集系数 = (节点i的邻域中的边数) / (邻域节点对的总数)
    """
    clustering_coeff = torch.zeros(num_nodes, dtype=torch.float32)
    
    # 构建邻接表
    adj_list = [set() for _ in range(num_nodes)]
    for src, dst in edge_index.t():
        src, dst = src.item(), dst.item()
        adj_list[src].add(dst)
        adj_list[dst].add(src)
    
    # 计算每个节点的聚集系数
    for node in range(num_nodes):
        neighbors = adj_list[node]
        if len(neighbors) <= 1:
            clustering_coeff[node] = 0.0
        else:
            # 计算邻域中的边数
            edges_in_neighborhood = 0
            neighbors_list = list(neighbors)
            for i in range(len(neighbors_list)):
                for j in range(i + 1, len(neighbors_list)):
                    if neighbors_list[j] in adj_list[neighbors_list[i]]:
                        edges_in_neighborhood += 1
            
            # 可能的边数
            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            clustering_coeff[node] = edges_in_neighborhood / possible_edges if possible_edges > 0 else 0.0
    
    return clustering_coeff


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
def save_canary_features(canaries_feats, save_path="./Test/canary_features.pt", use_timestamp=False):
    """保存Canary特征以便复用（只保存特征张量）
    
    Args:
        canaries_feats: 特征张量或元组列表
        save_path: 保存路径
        use_timestamp: 是否在文件名中添加时间戳
    """
    # 如果输入是元组列表，提取特征张量
    if isinstance(canaries_feats, list) and len(canaries_feats) > 0 and isinstance(canaries_feats[0], tuple):
        feats_tensor = torch.cat([feat for feat, _, _ in canaries_feats], dim=0)
    else:
        feats_tensor = canaries_feats
    
    # 如果使用时间戳，修改文件名
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
        base_path = save_path.rsplit('.', 1) if '.' in save_path else (save_path, '')
        if len(base_path) == 2:
            save_path = f"{base_path[0]}_{timestamp}.{base_path[1]}"
        else:
            save_path = f"{base_path[0]}_{timestamp}.pt"
    
    torch.save(feats_tensor, save_path)
    print(f"✓ Canary features saved to: {save_path} ({feats_tensor.shape[0]} features)")
    return save_path

def load_canary_features(load_path="./Test/canary_features.pt"):
    """加载已保存的Canary特征"""
    if os.path.exists(load_path):
        canaries_feats = torch.load(load_path)
        print(f"✓ Canary features loaded from: {load_path} ({len(canaries_feats)} features)")
        return canaries_feats
    return None

def get_canary_data(num_base_canaries, num_nodes, feat_dim, canary_class, y, neighbor_class_idx=None, canaries_file="./Test/canary_features.pt", edge_index=None, use_dynamic_degree=False, degree_strategy='mean'):
    """获取Canary特征和锚点，每次都生成新特征
    
    Args:
        canary_class: 金丝雀节点的类别（可指定）
        neighbor_class_idx: 邻居节点所属类别的节点索引（始终来自最多的类别）
        edge_index: 图的边索引，用于计算节点度数（仅在选择高度数节点或动态度数时需要）
        use_dynamic_degree: 是否根据原图平均度数动态调整Canary的连接数
        degree_strategy: 动态度数计算策略 ('mean', 'median', 'percentile', 'max', 'min')
    """
    # 总是生成新的随机特征
    print(f"Generating new random canary features...")
    canaries_feats = []
    for i in range(num_base_canaries):
        canary_feat = torch.rand((1, feat_dim)).unsqueeze(0)
        canaries_feats.append(canary_feat)
    canaries_feats = torch.cat(canaries_feats, dim=0)
    
    # 保存特征到带时间戳的文件
    saved_path = save_canary_features(canaries_feats, canaries_file, use_timestamp=True)
    print(f"✓ 特征已生成并保存: {saved_path}")
    
    # 确保形状正确
    if canaries_feats.dim() == 3:
        canaries_feats = canaries_feats.squeeze(1)
    
    # ========== 计算动态邻接数 ==========
    if use_dynamic_degree:
        dynamic_degree, degree_stats = compute_dynamic_degree(edge_index, num_nodes, strategy=degree_strategy)
        print(f"\n{'='*60}")
        print(f"动态度数计算 (策略: {degree_strategy})")
        print(f"{'='*60}")
        print(f"原图度数统计:")
        print(f"  最小: {degree_stats['min']:.0f}")
        print(f"  最大: {degree_stats['max']:.0f}")
        print(f"  平均: {degree_stats['mean']:.2f}")
        print(f"  中位数: {degree_stats['median']:.0f}")
        print(f"  25百分位: {degree_stats['percentile_25']:.2f}")
        print(f"  75百分位: {degree_stats['percentile_75']:.2f}")
        print(f"=> Canary节点连接数: {dynamic_degree} 个邻居")
        print(f"{'='*60}\n")
    else:
        dynamic_degree = 2  # 默认连接2个邻居
    
    # 为每个金丝雀生成锚点
    # ===== 选择邻居方式 =====
    '''方式1: 从整个图中随机选择节点作为邻居'''
    # print(f"Selecting canary anchors randomly from all nodes...")
    # canaries_anchors = [torch.randint(num_nodes, (1,)).item() for _ in range(num_base_canaries)]
    
    #'''方式2: 从指定类别中的前num_base_canaries个节点作为邻居（不循环，若不足则循环补充）'''
    print(f"Selecting canary anchors from specified class nodes...")
    canaries_anchors = [int(neighbor_class_idx[i % len(neighbor_class_idx)]) for i in range(num_base_canaries)]
    
    '''方式3: 从指定类别中随机选择节点作为邻居'''
    # print(f"Selecting canary anchors from specified class nodes (random)...")
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

    # ========== 方式 6: 从稠密子图中选择节点作为邻居（支持动态连接数）==========
    # print(f"Selecting canary anchors from dense subgraph (dynamic degree: {dynamic_degree})...")
    # # 识别稠密子图（使用K-core方法）
    # dense_nodes = find_dense_subgraphs(edge_index, num_nodes, method='k_core', top_k=num_base_canaries, k_value=None)
    # print(f"  Found {len(dense_nodes)} nodes in dense subgraph")
    
    # # 计算这些稠密节点中哪些属于邻居类别
    # neighbor_class_list = neighbor_class_idx.tolist()
    # dense_in_neighbor_class = [node.item() for node in dense_nodes if node.item() in neighbor_class_list]
    # print(f"  {len(dense_in_neighbor_class)} dense nodes in neighbor class")
    
    # # 如果稠密子图中没有足够的邻居类别节点，也使用稠密子图中的其他节点
    # if len(dense_in_neighbor_class) < num_base_canaries // 10:
    #     dense_nodes_list = dense_nodes.tolist()
    # else:
    #     dense_nodes_list = dense_in_neighbor_class
    
    # # 从稠密子图中选择锚点，根据动态度数选择相应数量的邻居
    # canaries_anchors = []
    # for _ in range(num_base_canaries):
    #     if len(dense_nodes_list) >= dynamic_degree:
    #         # 随机选择dynamic_degree个不重复的稠密节点
    #         neighbors = random.sample(dense_nodes_list, dynamic_degree)
    #     else:
    #         # 如果节点不足，使用全部节点并允许重复
    #         neighbors = [random.choice(dense_nodes_list) for _ in range(dynamic_degree)]
    #     canaries_anchors.append(neighbors)

    # 转换为完整数据格式 (feature, label, anchors)
    # 注意：此时的 anchor_node 是一个包含索引的列表
    canaries_data = [(canaries_feats[i:i+1], torch.tensor([canary_class], dtype=y.dtype), [canaries_anchors[i]]) 
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
    canary_class = class_counts.argmax().item()  # 默认与邻居类型相同，可修改为其他类型（如 0, 1, 2 等）
    
    if canary_class >= num_classes or canary_class < 0:
        raise ValueError(f"Invalid canary_class {canary_class}, must be in [0, {num_classes-1}]")
    
    print(f"Canary class: {canary_class}")

    # ========== 2b. 获取Canary特征和锚点 ==========
    canaries_file = "./Test/features/canary_features.pt"
    canaries_data = get_canary_data(num_base_canaries, num_nodes, feat_dim, canary_class, y, neighbor_class_idx, canaries_file, edge_index, use_dynamic_degree=use_dynamic_degree, degree_strategy=degree_strategy)

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
        print(f"  重复{rep_idx}: ID [{start_idx}, {end_idx-1}]")
        
    print(f"\nOUT配置 (金丝雀{num_in_base}-{num_base_canaries-1}):")
    print(f"  基础金丝雀: {num_out_base} 个")
    print(f"  总节点数: {num_out_nodes} (3个重复)")
    for rep_idx in range(num_repeats):
        start_idx = num_nodes_original + rep_idx * num_base_canaries + num_in_base
        end_idx = start_idx + num_out_base
        print(f"  重复{rep_idx}: ID [{start_idx}, {end_idx-1}]")

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
    print(f"New graph with canaries and masks saved: {data_new.num_nodes} nodes")

