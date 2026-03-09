import torch
from torch_geometric.data import Data
import random
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime

def generate_unique_filename(base_path, extension=".pt"):
    """生成带时间戳的唯一文件名
    
    Args:
        base_path: 文件的基础路径，例如 "./Test/canary_features"
        extension: 文件扩展名，例如 ".pt", ".npy"
    
    Returns:
        包含时间戳的文件路径，例如 "./Test/canary_features_20260308_143025_123456.pt"
    """
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 格式: YYYYMMDD_HHMMSS_mmm (毫秒)
    return f"{base_path}_{timestamp}{extension}"

def analyze_dataset_with_pca(x, y=None, n_components=None, save_path="./Test/pca_analysis"):
    """对原始数据集进行PCA分析
    
    Args:
        x: 特征矩阵 [N, F]
        y: 标签（可选）
        n_components: PCA主成分数（默认为特征维度的50%或10个，取较小值）
        save_path: 保存PCA模型和分析结果的路径前缀
    
    Returns:
        pca_model: 拟合的PCA模型
        pca_stats: 包含PCA统计信息的字典
    """
    print("\n" + "="*60)
    print("PCA分析原始数据集")
    print("="*60)
    
    x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    n_samples, n_features = x_np.shape
    
    # 确定PCA成分数
    if n_components is None:
        # 选择特征维度的50%，但最少10个，最多50个
        n_components = min(max(10, n_features // 2), 50)
    
    print(f"原始特征维度: {n_features}")
    print(f"数据样本数: {n_samples}")
    print(f"使用PCA成分数: {n_components}")
    
    # 标准化数据
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_np)
    
    # 拟合PCA
    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(x_scaled)
    
    # 统计信息
    explained_variance_ratio = pca.explained_variance_ratio_
    cumsum_variance = np.cumsum(explained_variance_ratio)
    
    print(f"\n方差解释率:")
    print(f"  前1个成分: {cumsum_variance[0]:.4f}")
    print(f"  前3个成分: {cumsum_variance[min(2, len(cumsum_variance)-1)]:.4f}")
    print(f"  前{min(10, len(cumsum_variance))}个成分: {cumsum_variance[min(9, len(cumsum_variance)-1)]:.4f}")
    print(f"  所有{n_components}个成分: {cumsum_variance[-1]:.4f}")
    
    print(f"\n主成分方差:")
    for i in range(min(5, n_components)):
        print(f"  PC{i+1}: {explained_variance_ratio[i]:.4f}")
    
    # 保存PCA模型和统计信息
    os.makedirs(save_path, exist_ok=True)
    pca_model_path = generate_unique_filename(f"{save_path}/pca_model", ".pt")
    torch.save({
        'pca_mean': torch.tensor(pca.mean_, dtype=torch.float32),
        'pca_components': torch.tensor(pca.components_, dtype=torch.float32),
        'pca_variance': torch.tensor(pca.explained_variance_, dtype=torch.float32),
        'scaler_mean': torch.tensor(scaler.mean_, dtype=torch.float32),
        'scaler_std': torch.tensor(scaler.scale_, dtype=torch.float32),
    }, pca_model_path)
    
    pca_stats = {
        'n_components': n_components,
        'explained_variance_ratio': explained_variance_ratio,
        'cumsum_variance': cumsum_variance,
        'pca': pca,
        'scaler': scaler,
        'x_pca': x_pca,
    }
    
    print(f"✓ PCA模型已保存到: {pca_model_path}")
    
    return pca, pca_stats


def visualize_pca_results(pca_stats, y=None, save_path="./Test/pca_analysis"):
    """可视化PCA分析结果
    
    Args:
        pca_stats: 从analyze_dataset_with_pca返回的统计信息字典
        y: 标签（用于着色）
        save_path: 保存图的路径
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 1. 方差解释率曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 单个成分的方差
    axes[0].bar(range(len(pca_stats['explained_variance_ratio'])), 
                pca_stats['explained_variance_ratio'], alpha=0.7)
    axes[0].set_xlabel('主成分')
    axes[0].set_ylabel('方差解释率')
    axes[0].set_title('PCA - 单个成分的方差')
    axes[0].grid(alpha=0.3)
    
    # 累积方差
    axes[1].plot(range(len(pca_stats['cumsum_variance'])), 
                 pca_stats['cumsum_variance'], 'o-', linewidth=2)
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% 阈值')
    axes[1].set_xlabel('主成分数')
    axes[1].set_ylabel('累积方差解释率')
    axes[1].set_title('PCA - 累积方差解释率')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    variance_fig_path = generate_unique_filename(f"{save_path}/variance_explained", ".png")
    plt.savefig(variance_fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ 方差曲线已保存到: {variance_fig_path}")
    plt.close()
    
    # 2. 2D投影（前两个主成分）
    if pca_stats['x_pca'].shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if y is not None:
            y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
            scatter = ax.scatter(pca_stats['x_pca'][:, 0], pca_stats['x_pca'][:, 1], 
                                c=y_np, cmap='tab10', alpha=0.6, s=20)
            plt.colorbar(scatter, ax=ax, label='类别')
        else:
            ax.scatter(pca_stats['x_pca'][:, 0], pca_stats['x_pca'][:, 1], 
                      alpha=0.6, s=20)
        
        ax.set_xlabel(f'PC1 ({pca_stats["explained_variance_ratio"][0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca_stats["explained_variance_ratio"][1]:.2%})')
        ax.set_title('PCA 2D投影（前两个主成分）')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        projection_fig_path = generate_unique_filename(f"{save_path}/pca_2d_projection", ".png")
        plt.savefig(projection_fig_path, dpi=150, bbox_inches='tight')
        print(f"✓ 2D投影已保存到: {projection_fig_path}")
        plt.close()


def generate_canaries_with_pca(num_canaries, feat_dim, pca, pca_stats, 
                                scale_factor=1.0, use_principal_components=True):
    """基于PCA结果生成增强的金丝雀特征
    
    Args:
        num_canaries: 金丝雀数量
        feat_dim: 原始特征维度
        pca: PCA模型对象
        pca_stats: PCA统计信息字典
        scale_factor: 在主成分空间中缩放的因子（控制金丝雀与原始数据的距离）
        use_principal_components: 是否在主成分空间中生成，然后投影回原始空间
    
    Returns:
        canaries_feats: 生成的金丝雀特征 [num_canaries, feat_dim]
    """
    print(f"\n基于PCA生成{num_canaries}个增强的金丝雀特征...")
    
    if use_principal_components:
        # 在PCA空间中生成特征
        n_components = pca.n_components_
        
        # 方法1：在主成分空间中生成，使用实际的方差作为尺度
        canaries_pca = []
        for i in range(num_canaries):
            # 对每个主成分，按其方差的倍数采样
            components_scores = []
            for j in range(n_components):
                # 从N(0, scale_factor * variance)采样
                variance = pca.explained_variance_[j]
                score = np.random.normal(0, scale_factor * np.sqrt(variance))
                components_scores.append(score)
            canaries_pca.append(components_scores)
        
        canaries_pca = np.array(canaries_pca)  # [num_canaries, n_components]
        
        # 投影回原始空间
        canaries_np = pca.inverse_transform(canaries_pca)
    else:
        # 直接在原始空间中生成（保守方法）
        canaries_np = np.random.randn(num_canaries, feat_dim)
        # 缩放到与原始数据相同的范围
        data_std = np.std(pca_stats['scaler'].scale_)
        canaries_np *= data_std * scale_factor
    
    # 转换为torch张量
    canaries_feats = torch.tensor(canaries_np, dtype=torch.float32)
    
    print(f"✓ 生成的金丝雀特征形状: {canaries_feats.shape}")
    print(f"  金丝雀特征统计:")
    print(f"    均值: {canaries_feats.mean(dim=0).norm():.4f}")
    print(f"    标准差: {canaries_feats.std(dim=0).mean():.4f}")
    
    return canaries_feats


def save_canary_features(canaries_feats, save_path="./Test/canary_features.pt"):
    """保存Canary特征以便复用（只保存特征张量），自动添加时间戳"""
    # 如果输入是元组列表，提取特征张量
    if isinstance(canaries_feats, list) and len(canaries_feats) > 0 and isinstance(canaries_feats[0], tuple):
        feats_tensor = torch.cat([feat for feat, _, _ in canaries_feats], dim=0)
    else:
        feats_tensor = canaries_feats
    
    # 从 save_path 分离目录和文件名
    if save_path.endswith(".pt"):
        base_path = save_path[:-3]  # 移除 .pt 后缀
    else:
        base_path = save_path
    
    # 生成带时间戳的文件名
    unique_path = generate_unique_filename(base_path, ".pt")
    torch.save(feats_tensor, unique_path)
    print(f"✓ Canary features saved to: {unique_path} ({feats_tensor.shape[0]} features)")

def load_canary_features(load_path="./Test/canary_features.pt"):
    """加载已保存的Canary特征"""
    if os.path.exists(load_path):
        canaries_feats = torch.load(load_path)
        print(f"✓ Canary features loaded from: {load_path} ({len(canaries_feats)} features)")
        return canaries_feats
    return None

def get_canary_data(num_base_canaries, num_nodes, feat_dim, canary_class, y, neighbor_class_idx=None, canaries_file="./Test/canary_features.pt", edge_index=None, use_dynamic_degree=False, degree_strategy='mean', use_pca=False, pca_info=None):
    """获取Canary特征和锚点，支持加载缓存或生成新特征
    
    Args:
        canary_class: 金丝雀节点的类别（可指定）
        neighbor_class_idx: 邻居节点所属类别的节点索引（始终来自最多的类别）
        edge_index: 图的边索引，用于计算节点度数（仅在选择高度数节点或动态度数时需要）
        use_dynamic_degree: 是否根据原图平均度数动态调整Canary的连接数
        degree_strategy: 动态度数计算策略 ('mean', 'median', 'percentile', 'max', 'min')
        use_pca: 是否使用PCA增强的金丝雀生成
        pca_info: 包含PCA模型和统计信息的字典 {'pca': pca_model, 'pca_stats': stats}
    """
    cached_feats = None
    
    if cached_feats is None:
        print(f"Generating new canary features (not found in {canaries_file})...")
        
        if use_pca and pca_info is not None:
            # 使用PCA增强生成
            print(f"使用PCA增强生成{num_base_canaries}个金丝雀特征...")
            pca = pca_info.get('pca')
            pca_stats = pca_info.get('pca_stats')
            canaries_feats = generate_canaries_with_pca(
                num_base_canaries, feat_dim, pca, pca_stats, 
                scale_factor=1.0, use_principal_components=True
            )
        else:
            # 使用默认随机生成
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
            
            if use_pca and pca_info is not None:
                # 生成额外的PCA增强特征
                pca = pca_info.get('pca')
                pca_stats = pca_info.get('pca_stats')
                extra_feats = generate_canaries_with_pca(
                    num_base_canaries - num_cached, feat_dim, pca, pca_stats,
                    scale_factor=1.0, use_principal_components=True
                )
                canaries_feats = torch.cat([canaries_feats, extra_feats], dim=0)
            else:
                for i in range(num_cached, num_base_canaries):
                    canary_feat = torch.randn(feat_dim).unsqueeze(0)
                    canaries_feats = torch.cat([canaries_feats, canary_feat], dim=0)
        else:
            canaries_feats = cached_feats
    
    # 确保形状正确
    if canaries_feats.dim() == 3:
        canaries_feats = canaries_feats.squeeze(1)
    
    
    # 为每个金丝雀生成锚点
    # ===== 选择邻居方式 =====
    '''方式1: 从整个图中随机选择节点作为邻居'''
    # print(f"从整个图中随机选择节点作为邻居...")
    # canaries_anchors = [torch.randint(num_nodes, (1,)).item() for _ in range(num_base_canaries)]
    
    # '''方式2: 从指定类别中的前num_base_canaries个节点作为邻居（不循环，若不足则循环补充）'''
    print(f"Selecting canary anchors from specified class nodes...从指定类别中的前num_base_canaries个节点作为邻居")
    canaries_anchors = [int(neighbor_class_idx[i % len(neighbor_class_idx)]) for i in range(num_base_canaries)]
    
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

    # 转换为完整数据格式 (feature, label, anchors)
    # 注意：此时的 anchor_node 是一个包含索引的列表

    print("随机选择节点类型")
    canaries_data = [(canaries_feats[i:i+1], torch.tensor([random.randint(0, 9)], dtype=y.dtype), [canaries_anchors[i]]) 
                     for i in range(num_base_canaries)]
    # print("使用指定的canary_class类型...")
    # canaries_data = [(canaries_feats[i:i+1],torch.tensor([canary_class], dtype=y.dtype), [canaries_anchors[i]]) 
    #                  for i in range(num_base_canaries)]
    return canaries_data





def create_canary_graph(num_base_canaries, num_repeats, use_dynamic_degree=False, degree_strategy='mean', use_pca_enhancement=True):
    # ========== 1. 加载原始子图 ==========
    data = torch.load("amazon_subgraph_all.pt")

    x = data.x.clone()                   # [N, F]
    y = data.y.clone()                   # [N]
    edge_index = data.edge_index.clone() # [2, E]

    num_nodes, feat_dim = x.size()
    num_classes = int(y.max().item()) + 1

    print(f"Original graph: {num_nodes} nodes")

    # ========== 1.5. PCA分析 ==========
    pca_info = None
    if use_pca_enhancement:
        print("\n" + "="*60)
        print("步骤1: 对原始数据集进行PCA分析")
        print("="*60)
        pca, pca_stats = analyze_dataset_with_pca(x, y, n_components=None, save_path="./Test/pca_analysis")
        
        # 可视化PCA结果
        visualize_pca_results(pca_stats, y, save_path="./Test/pca_analysis")
        
        pca_info = {
            'pca': pca,
            'pca_stats': pca_stats
        }
        print("\n✓ PCA分析完成！")
    
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
    print("\n" + "="*60)
    print("步骤2: 生成增强的金丝雀特征")
    print("="*60)
    canaries_file = "./Test/canary_features.pt"
    canaries_data = get_canary_data(
        num_base_canaries, num_nodes, feat_dim, canary_class, y, 
        neighbor_class_idx, canaries_file, edge_index, 
        use_dynamic_degree=use_dynamic_degree, 
        degree_strategy=degree_strategy,
        use_pca=use_pca_enhancement,
        pca_info=pca_info
    )

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
    
    # 保存canary_mask到数据中，使用时间戳增强唯一性
    #canary_mask_path = generate_unique_filename("./Test/canary_mask", ".npy")
    np.save("./Test/canary_mask", canary_mask)
    print(f"✓ Canary mask saved to ./Test/canary_mask (size={num_base_canaries})")
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

    # 生成带时间戳的文件名
    output_graph_path = generate_unique_filename("./Test/amazon_subgraph_black", ".pt")
    torch.save(data_new, output_graph_path)
    print(f"New graph with canaries and masks saved: {data_new.num_nodes} nodes")
    print(f"Output file: {output_graph_path}")


if __name__ == "__main__":
    """
    主函数：创建带有PCA增强金丝雀的图
    
    使用说明：
    1. 将数据文件 amazon_subgraph_all.pt 放在当前目录
    2. 运行此脚本
    3. 将生成（所有输出文件名都包含时间戳以确保唯一性）：
       - ./Test/pca_analysis/pca_model_TIMESTAMP.pt - PCA模型
       - ./Test/canary_features_TIMESTAMP.pt - 金丝雀特征
       - ./Test/canary_mask_TIMESTAMP.npy - 金丝雀掩码
       - ./Test/amazon_subgraph_black_TIMESTAMP.pt - 最终的增强图
    """
    
    print("\n" + "="*60)
    print("PCA增强金丝雀图构造")
    print("="*60)
    
    # 配置参数
    num_base_canaries = 3000  # 基础金丝雀数
    num_repeats = 2           # 重复次数（每个金丝雀在图中重复的次数）
    use_pca_enhancement = True  # 是否使用PCA增强
    
    print(f"配置:")
    print(f"  基础金丝雀数: {num_base_canaries}")
    print(f"  重复次数: {num_repeats}")
    print(f"  总金丝雀数: {num_base_canaries * num_repeats}")
    print(f"  使用PCA增强: {use_pca_enhancement}")
    
    # 创建带PCA增强的金丝雀图
    create_canary_graph(
        num_base_canaries=num_base_canaries,
        num_repeats=num_repeats,
        use_dynamic_degree=False,
        degree_strategy='mean',
        use_pca_enhancement=use_pca_enhancement
    )
    
    print("\n" + "="*60)
    print("✓ 金丝雀图构造完成！")
    print("="*60)
    print("\n生成的文件（包含时间戳以确保每次运行都生成独特的文件）:")
    print("  - ./Test/amazon_subgraph_black_TIMESTAMP.pt: 增强后的图")
    print("  - ./Test/canary_features_TIMESTAMP.pt: 金丝雀特征")
    print("  - ./Test/canary_mask_TIMESTAMP.npy: 金丝雀IN/OUT掩码")
    print("  - ./Test/pca_analysis/pca_model_TIMESTAMP.pt: PCA模型和参数")
    print("  - ./Test/pca_analysis/variance_explained.png: 方差解释率图")
    print("  - ./Test/pca_analysis/pca_2d_projection.png: 2D投影图")
