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
        # 如果是列表，保存为张量或pickle格式
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
    """保存Canary的类别标签（随机选择的类别）"""
    # 转换为张量保存
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
        # 转换为列表格式以匹配原始格式
        if isinstance(canaries_labels, torch.Tensor):
            canaries_labels = canaries_labels.tolist()
        print(f"✓ Canary labels loaded from: {load_path} ({len(canaries_labels)} labels)")
        return canaries_labels
    return None

def get_canary_data(num_base_canaries, num_nodes, feat_dim, canary_class, y, neighbor_class_idx=None, canaries_file="./Test/canary_features.pt", edge_index=None, use_dynamic_degree=False, degree_strategy='mean', canaries_anchors_file="./Test/canary_anchors.pt", canaries_labels_file="./Test/canary_labels.pt"):
    """获取Canary特征和锚点，支持加载缓存或生成新特征
    
    Args:
        canary_class: 金丝雀节点的类别（可指定）
        neighbor_class_idx: 邻居节点所属类别的节点索引（始终来自最多的类别）
        canaries_file: 金丝雀特征的保存/读取路径
        canaries_anchors_file: 金丝雀邻居节点的保存/读取路径
        canaries_labels_file: 金丝雀类别标签的保存/读取路径
        edge_index: 图的边索引，用于计算节点度数（仅在选择高度数节点或动态度数时需要）
        use_dynamic_degree: 是否根据原图平均度数动态调整Canary的连接数
        degree_strategy: 动态度数计算策略 ('mean', 'median', 'percentile', 'max', 'min')
    """
    cached_feats = load_canary_features(canaries_file)
    
    if cached_feats is None:
        print(f"Generating new canary features (not found in {canaries_file})...")
        # 生成新特征
        canaries_feats = []
        for i in range(num_base_canaries):
            canary_feat = torch.randn(feat_dim).unsqueeze(0)
            canaries_feats.append(canary_feat)
        canaries_feats = torch.cat(canaries_feats, dim=0)
        # 保存特征
        save_canary_features(canaries_feats, canaries_file)
    else:
        print(f"Using cached canary features...")
        # 处理数量不匹配的情况
        num_cached = len(cached_feats)
        if num_cached > num_base_canaries:
            print(f"  Warning: Cached {num_cached} features > needed {num_base_canaries}")
            canaries_feats = cached_feats[:num_base_canaries]
        elif num_cached < num_base_canaries:
            print(f"  Warning: Cached {num_cached} features < needed {num_base_canaries}")
            print(f"  Generating {num_base_canaries - num_cached} new ones")
            canaries_feats = cached_feats.clone()
            for i in range(num_cached, num_base_canaries):
                canary_feat = torch.randn(feat_dim).unsqueeze(0)
                canaries_feats = torch.cat([canaries_feats, canary_feat], dim=0)
        else:
            canaries_feats = cached_feats
    
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
        dynamic_degree = 1  # 默认连接1个邻居
    
    # 为每个金丝雀生成锚点或加载缓存
    # ===== 尝试加载缓存的邻居节点 =====
    cached_anchors = load_canary_anchors(canaries_anchors_file)
    
    if cached_anchors is None:
        print(f"Generating new canary anchors (not found in {canaries_anchors_file})...")
        # ===== 选择邻居方式 =====
        '''方式1: 从整个图中随机选择节点作为邻居'''
        # print(f"从整个图中随机选择节点作为邻居...")
        # canaries_anchors = [torch.randint(num_nodes, (1,)).item() for _ in range(num_base_canaries)]
        '''方式2: 从指定类别中的前num_base_canaries个节点作为邻居（不循环，若不足则循环补充）'''
        print(f"从指定类别中的前num_base_canaries个节点作为邻居")
        canaries_anchors = [int(neighbor_class_idx[i % len(neighbor_class_idx)]) for i in range(num_base_canaries)]
        
        # 保存邻居节点
        save_canary_anchors(canaries_anchors, canaries_anchors_file)
    else:
        print(f"Using cached canary anchors...")
        canaries_anchors = cached_anchors
        # 处理数量不匹配的情况
        if len(canaries_anchors) > num_base_canaries:
            print(f"  Warning: Cached {len(canaries_anchors)} anchors > needed {num_base_canaries}")
            canaries_anchors = canaries_anchors[:num_base_canaries]
        elif len(canaries_anchors) < num_base_canaries:
            print(f"  Warning: Cached {len(canaries_anchors)} anchors < needed {num_base_canaries}")
            print(f"  Generating {num_base_canaries - len(canaries_anchors)} new ones")
            new_anchors = [torch.randint(num_nodes, (1,)).item() for _ in range(num_base_canaries - len(canaries_anchors))]
            canaries_anchors = canaries_anchors + new_anchors
            save_canary_anchors(canaries_anchors, canaries_anchors_file)
    
    # ===== 尝试加载缓存的标签 =====
    cached_labels = load_canary_labels(canaries_labels_file)
    
    if cached_labels is None:
        print(f"Generating new canary labels (not found in {canaries_labels_file})...")
        print("随机选择节点类型")
        canaries_labels = [random.randint(0, 9) for _ in range(num_base_canaries)]
        
        # 保存标签
        save_canary_labels(canaries_labels, canaries_labels_file)
    else:
        print(f"Using cached canary labels...")
        canaries_labels = cached_labels
        # 处理数量不匹配的情况
        if len(canaries_labels) > num_base_canaries:
            print(f"  Warning: Cached {len(canaries_labels)} labels > needed {num_base_canaries}")
            canaries_labels = canaries_labels[:num_base_canaries]
        elif len(canaries_labels) < num_base_canaries:
            print(f"  Warning: Cached {len(canaries_labels)} labels < needed {num_base_canaries}")
            print(f"  Generating {num_base_canaries - len(canaries_labels)} new ones")
            new_labels = [random.randint(0, 9) for _ in range(num_base_canaries - len(canaries_labels))]
            canaries_labels = canaries_labels + new_labels
            save_canary_labels(canaries_labels, canaries_labels_file)
    
    # '''方式2: 从指定类别中的前num_base_canaries个节点作为邻居（不循环，若不足则循环补充）'''
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
    canaries_data = [(canaries_feats[i:i+1], torch.tensor([canaries_labels[i]], dtype=y.dtype), [canaries_anchors[i]]) 
                     for i in range(num_base_canaries)]
    # print("使用指定的canary_class类型...")
    # canaries_data = [(canaries_feats[i:i+1],torch.tensor([canary_class], dtype=y.dtype), [canaries_anchors[i]]) 
    #                  for i in range(num_base_canaries)]
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

    # ⚠️  正确方式：为3000个原始金丝雀生成掩码，重复的版本继承原始的掩码标记
    num_in_base = num_base_canaries // 2  # 1500个原始金丝雀作为IN
    num_out_base = num_base_canaries - num_in_base  # 1500个原始金丝雀作为OUT
    

    '''选择前 num_in_base 个原始金丝雀索引作为IN（连续索引，易于调试）'''
    in_base_indices = set(range(num_in_base))  # [0, 1, 2, ..., num_in_base-1]
    
    '''随机选择 num_in_base 个原始金丝雀索引作为IN（随机索引，更真实）'''
    # in_base_indices = set(random.sample(range(num_base_canaries), num_in_base))
    
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

