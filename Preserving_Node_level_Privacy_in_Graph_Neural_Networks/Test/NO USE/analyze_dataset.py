#!/usr/bin/env python3
"""
全面分析 Amazon 数据集的特征和连通性信息
重点关注：节点特征分析和图的连通性结构
"""

import torch
import numpy as np
from collections import Counter
import networkx as nx
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import random

def analyze_dataset(data_path="./Test/amazon_subgraph_all.pt"):
    """
    加载并分析数据集
    """
    print("=" * 80)
    print("Amazon 数据集全面分析")
    print("=" * 80)
    
    # ========== 1. 加载数据 ==========
    print("\n[1] 加载数据...")
    try:
        data = torch.load(data_path)
        print(f"✓ 成功加载: {data_path}")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return
    
    x = data.x  # 节点特征 [N, F]
    y = data.y  # 节点标签 [N]
    edge_index = data.edge_index  # 边索引 [2, E]
    
    num_nodes = x.shape[0]
    num_edges = edge_index.shape[1]
    feat_dim = x.shape[1]
    
    print(f"\n基本信息:")
    print(f"  • 节点数: {num_nodes:,}")
    print(f"  • 边数: {num_edges:,}")
    print(f"  • 特征维度: {feat_dim}")
    print(f"  • 类别数: {int(y.max().item()) + 1}")
    print(f"  • 数据类型: 特征={x.dtype}, 标签={y.dtype}")
    print(f"  • 是否有向图: {not is_undirected(edge_index)}")
    
    # ========== 2. 节点特征分析 ==========
    print("\n" + "=" * 80)
    print("[2] 节点特征分析")
    print("=" * 80)
    
    analyze_node_features(x, feat_dim)
    
    # ========== 3. 连通性分析 ==========
    print("\n" + "=" * 80)
    print("[3] 连通性分析")
    print("=" * 80)
    
    analyze_connectivity(edge_index, num_nodes)
    
    # ========== 4. 度数分析 ==========
    print("\n" + "=" * 80)
    print("[4] 度数分析")
    print("=" * 80)
    
    analyze_degree(edge_index, num_nodes)
    
    # ========== 5. 标签分布 ==========
    print("\n" + "=" * 80)
    print("[5] 类别分布分析")
    print("=" * 80)
    
    analyze_labels(y)
    
    # ========== 6. 图的结构特性 ==========
    print("\n" + "=" * 80)
    print("[6] 图的结构特性")
    print("=" * 80)
    
    analyze_graph_structure(edge_index, num_nodes, x, y)
    
    # ========== 7. 社区结构 ==========
    print("\n" + "=" * 80)
    print("[7] 社区结构分析")
    print("=" * 80)
    
    analyze_community(edge_index, num_nodes)
    
    # ========== 8. 邻接关系特性 ==========
    print("\n" + "=" * 80)
    print("[8] 邻接关系特性")
    print("=" * 80)
    
    analyze_neighborhood(edge_index, num_nodes, y)
    
    # ========== 9. 稀疏性分析 ==========
    print("\n" + "=" * 80)
    print("[9] 稀疏性分析")
    print("=" * 80)
    
    analyze_sparsity(num_nodes, num_edges, feat_dim, x, edge_index)
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80 + "\n")


def is_undirected(edge_index):
    """检查图是否为无向图"""
    edges_forward = set(map(tuple, edge_index.t().tolist()))
    edges_backward = set((dst, src) for src, dst in edges_forward)
    return edges_forward == edges_backward


def analyze_node_features(x, feat_dim):
    """节点特征详细分析"""
    print("\n特征统计:")
    print(f"  • 特征形状: {x.shape}")
    print(f"  • 特征数据类型: {x.dtype}")
    print(f"  • 特征设备: {x.device}")
    
    # 特征值统计
    x_numpy = x.numpy() if isinstance(x, torch.Tensor) else x
    
    print(f"\n特征值统计 (全局):")
    print(f"  • 最小值: {x_numpy.min():.6f}")
    print(f"  • 最大值: {x_numpy.max():.6f}")
    print(f"  • 平均值: {x_numpy.mean():.6f}")
    print(f"  • 标准差: {x_numpy.std():.6f}")
    print(f"  • 中位数: {np.median(x_numpy):.6f}")
    
    # 各维度特征统计
    print(f"\n各维度特征统计:")
    feature_mins = x_numpy.min(axis=0)
    feature_maxs = x_numpy.max(axis=0)
    feature_means = x_numpy.mean(axis=0)
    feature_stds = x_numpy.std(axis=0)
    
    print(f"  • 最小值范围: [{feature_mins.min():.6f}, {feature_mins.max():.6f}]")
    print(f"  • 最大值范围: [{feature_maxs.min():.6f}, {feature_maxs.max():.6f}]")
    print(f"  • 平均值范围: [{feature_means.min():.6f}, {feature_means.max():.6f}]")
    print(f"  • 标准差范围: [{feature_stds.min():.6f}, {feature_stds.max():.6f}]")
    
    # 检查零特征
    zero_mask = (x_numpy == 0).all(axis=0)
    zero_features = np.where(zero_mask)[0]
    if len(zero_features) > 0:
        print(f"  • ⚠ 全零特征维度: {zero_features.tolist()}")
    
    # 稀疏性
    sparsity = (x_numpy == 0).sum() / (x_numpy.shape[0] * x_numpy.shape[1])
    print(f"  • 特征稀疏度: {sparsity:.4f} ({100*sparsity:.2f}%)")
    
    # 特征相关性抽样检查
    if feat_dim > 1:
        sample_corr = np.corrcoef(x_numpy.T)
        print(f"\n特征相关性矩阵:")
        print(f"  • 形状: {sample_corr.shape}")
        print(f"  • 平均相关系数: {(sample_corr[np.triu_indices_from(sample_corr, k=1)]).mean():.6f}")


def analyze_connectivity(edge_index, num_nodes):
    """连通性详细分析"""
    # 转换为 NetworkX 图
    edges = edge_index.t().tolist()
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)
    
    # 检查有向/无向
    is_directed = True
    edges_set = set(map(tuple, edges))
    reverse_edges = [(dst, src) for src, dst in edges_set]
    if set(reverse_edges).issubset(edges_set):
        is_directed = False
        G = G.to_undirected()
    
    print(f"\n图类型:")
    print(f"  • 有向: {is_directed}")
    
    # 连通分量
    if is_directed:
        num_weakly_connected = nx.number_weakly_connected_components(G)
        num_strongly_connected = nx.number_strongly_connected_components(G)
        print(f"  • 弱连通分量数: {num_weakly_connected}")
        print(f"  • 强连通分量数: {num_strongly_connected}")
    else:
        num_connected = nx.number_connected_components(G)
        print(f"  • 连通分量数: {num_connected}")
    
    # 最大连通分量
    if is_directed:
        largest_cc = max(nx.weakly_connected_components(G), key=len)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
    
    largest_cc_size = len(largest_cc)
    largest_cc_ratio = largest_cc_size / num_nodes
    
    print(f"\n最大连通分量:")
    print(f"  • 大小: {largest_cc_size:,} 节点 ({largest_cc_ratio*100:.2f}%)")
    
    # 孤立节点
    isolated_nodes = list(nx.isolates(G))
    print(f"  • 孤立节点: {len(isolated_nodes)} 个")
    
    # 最短路径信息（仅对小的连通分量计算）
    if largest_cc_size > 1 and largest_cc_ratio > 0.8:
        G_largest = G.subgraph(largest_cc).copy()
        
        # 只在连通分量较小时计算最短路径（超过3000节点后计算量过大）
        if largest_cc_size < 3000:
            try:
                # 计算平均最短路径长度（仅在连通图中）
                if is_directed:
                    # 对于有向图，计算弱连通的情况
                    avg_shortest_path = nx.average_shortest_path_length(G_largest.to_undirected())
                else:
                    avg_shortest_path = nx.average_shortest_path_length(G_largest)
                diameter = nx.diameter(G_largest.to_undirected())
                print(f"  • 平均最短路径长度: {avg_shortest_path:.4f}")
                print(f"  • 直径: {diameter}")
            except Exception as e:
                print(f"  • 平均最短路径长度: 无法计算（{str(e)[:30]}...）")
        else:
            print(f"  • 平均最短路径长度: 跳过计算（分量过大: {largest_cc_size} 节点）")
            # 使用采样估计直径
            try:
                sampled_distances = []
                import random
                sample_nodes = random.sample(list(largest_cc), min(100, largest_cc_size // 10))
                for node in sample_nodes:
                    lengths = nx.single_source_shortest_path_length(G_largest, node)
                    sampled_distances.extend(lengths.values())
                if sampled_distances:
                    estimated_diameter = max(sampled_distances)
                    print(f"  • 直径 (采样估计): ≈ {estimated_diameter}")
            except:
                print(f"  • 直径: 无法估计")
    
    # 聚集系数（只在图不太大时计算全局值）
    try:
        if num_nodes <= 5000:
            if is_directed:
                avg_clustering = nx.average_clustering(G.to_undirected())
            else:
                avg_clustering = nx.average_clustering(G)
            print(f"  • 平均聚集系数: {avg_clustering:.6f}")
        else:
            # 采样计算
            import random
            sample_nodes = random.sample(range(num_nodes), min(500, num_nodes // 10))
            clustering_values = [nx.clustering(G.to_undirected() if is_directed else G, node) for node in sample_nodes]
            avg_clustering_sample = np.mean(clustering_values)
            print(f"  • 平均聚集系数 (采样500节点): {avg_clustering_sample:.6f}")
    except Exception as e:
        print(f"  • 聚集系数计算失败: {str(e)[:30]}")
    
    # 过渡性
    try:
        transitivity = nx.transitivity(G.to_undirected())
        print(f"  • 图的过渡性: {transitivity:.6f}")
    except:
        pass


def analyze_degree(edge_index, num_nodes):
    """度数详细分析"""
    # 计算出入度
    in_degree = torch.zeros(num_nodes, dtype=torch.long)
    out_degree = torch.zeros(num_nodes, dtype=torch.long)
    
    in_degree.scatter_add_(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.long))
    out_degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.long))
    
    total_degree = in_degree + out_degree
    
    print(f"\n度数统计:")
    print(f"\n  出度:")
    print(f"    • 最小: {int(out_degree.min())}")
    print(f"    • 最大: {int(out_degree.max())}")
    print(f"    • 平均: {float(out_degree.float().mean()):.4f}")
    print(f"    • 中位数: {int(torch.median(out_degree.float()))}")
    
    print(f"\n  入度:")
    print(f"    • 最小: {int(in_degree.min())}")
    print(f"    • 最大: {int(in_degree.max())}")
    print(f"    • 平均: {float(in_degree.float().mean()):.4f}")
    print(f"    • 中位数: {int(torch.median(in_degree.float()))}")
    
    print(f"\n  总度数:")
    print(f"    • 最小: {int(total_degree.min())}")
    print(f"    • 最大: {int(total_degree.max())}")
    print(f"    • 平均: {float(total_degree.float().mean()):.4f}")
    print(f"    • 中位数: {int(torch.median(total_degree.float()))}")
    
    # 度数分布
    print(f"\n度数分布:")
    degree_dist = torch.bincount(total_degree)
    print(f"  • 度数为 0 的节点: {int((total_degree == 0).sum())}")
    print(f"  • 度数为 1 的节点: {int((total_degree == 1).sum())}")
    print(f"  • 度数为 2-5 的节点: {int(((total_degree >= 2) & (total_degree <= 5)).sum())}")
    print(f"  • 度数为 6-10 的节点: {int(((total_degree >= 6) & (total_degree <= 10)).sum())}")
    print(f"  • 度数 > 10 的节点: {int((total_degree > 10).sum())}")
    
    # 幂律检验（粗略）
    print(f"\n度数分布特性:")
    total_degree_float = total_degree.float()
    high_degree_nodes = (total_degree > total_degree_float.quantile(0.9)).sum().item()
    print(f"  • 高度节点 (top 10%): {high_degree_nodes}")
    print(f"  • 低度节点 (degree ≤ 2): {int((total_degree <= 2).sum())}")


def analyze_labels(y):
    """标签分布分析"""
    num_classes = int(y.max().item()) + 1
    label_counts = torch.bincount(y)
    
    print(f"\n类别信息:")
    print(f"  • 总类别数: {num_classes}")
    
    print(f"\n各类别大小:")
    for class_id in range(num_classes):
        count = int(label_counts[class_id])
        ratio = count / len(y) * 100
        print(f"    Class {class_id}: {count:,} 节点 ({ratio:.2f}%)")
    
    # 最大/最小类
    max_class = label_counts.argmax().item()
    min_class = label_counts.argmin().item()
    max_count = int(label_counts[max_class])
    min_count = int(label_counts[min_class])
    
    print(f"\n类别均衡性:")
    print(f"  • 最大类: Class {max_class} ({max_count} 节点)")
    print(f"  • 最小类: Class {min_class} ({min_count} 节点)")
    print(f"  • 比例: {max_count / min_count:.2f}:1")
    print(f"  • 熵: {calculate_entropy(label_counts / len(y)):.4f}")


def analyze_graph_structure(edge_index, num_nodes, x, y):
    """图的结构特性分析"""
    num_edges = edge_index.shape[1]
    
    # 密度
    max_edges = num_nodes * (num_nodes - 1)
    density = num_edges / max_edges
    
    print(f"\n图的密度:")
    print(f"  • 实际边数: {num_edges:,}")
    print(f"  • 最大可能边数: {max_edges:,}")
    print(f"  • 图密度: {density:.6f}")
    print(f"  • 稀疏程度: {'高度稀疏' if density < 0.001 else '中等稀疏' if density < 0.01 else '相对密集'}")
    
    # 同质性 (Homophily) - 同类邻接比例
    homophily_score = calculate_homophily(edge_index, y)
    print(f"\n同质性指标:")
    print(f"  • 同类邻接率: {homophily_score:.4f}")
    print(f"  • 解释: {'高同质性（同类节点倾向相邻）' if homophily_score > 0.5 else '异质性较强'}")
    
    # 平均邻接类别多样性（采样计算以加快速度）
    print(f"\n邻接特性:")
    edges_list = edge_index.t().tolist()
    
    # 只对前10000条边或所有边进行计算（限制计算量）
    sample_edges = edges_list[:min(10000, len(edges_list))]
    neighbor_class_diversity = []
    for src, dst in sample_edges:
        if int(y[src]) != int(y[dst]):
            neighbor_class_diversity.append(1)
        else:
            neighbor_class_diversity.append(0)
    
    if neighbor_class_diversity:
        diversity = np.mean(neighbor_class_diversity)
        print(f"  • 跨类边比例 ({len(sample_edges)} 条边采样): {diversity:.4f}")


def analyze_community(edge_index, num_nodes):
    """社区结构分析"""
    edges = edge_index.t().tolist()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)
    
    print(f"\n社区检测:")
    
    # 仅在图不太大时进行社区检测
    if num_nodes <= 10000:
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(G))
            
            print(f"  • 社区数: {len(communities)}")
            print(f"  • 社区大小分布:")
            sizes = sorted([len(c) for c in communities], reverse=True)
            
            for i, size in enumerate(sizes[:5]):  # 显示前5个
                ratio = size / num_nodes * 100
                print(f"    社区 {i+1}: {size:,} 节点 ({ratio:.2f}%)")
            
            if len(sizes) > 5:
                print(f"    ... 及其他 {len(sizes) - 5} 个社区")
            
            # 计算模块性
            modularity = calculate_modularity(G, communities)
            print(f"  • 模块性: {modularity:.4f}")
        except Exception as e:
            print(f"  • 社区检测失败: {str(e)[:50]}...")
    else:
        print(f"  • 跳过社区检测（图过大: {num_nodes} 节点）")
        print(f"  • 改用快速聚类系数估计:")
        try:
            # 采样计算聚集系数
            import random
            sample_nodes = random.sample(range(num_nodes), min(1000, num_nodes))
            clustering_values = [nx.clustering(G, node) for node in sample_nodes]
            avg_clustering = np.mean(clustering_values)
            print(f"    采样平均聚集系数: {avg_clustering:.6f}")
        except:
            pass


def analyze_neighborhood(edge_index, num_nodes, y):
    """邻接关系特性"""
    # 计算邻接表（仅采样构建，避免内存溢出）
    edges = edge_index.t().tolist()
    
    print(f"\n邻接关系分析:")
    
    # 快速计算度数统计
    in_degree = torch.zeros(num_nodes, dtype=torch.long)
    in_degree.scatter_add_(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.long))
    
    neighbor_sizes = in_degree.tolist()
    print(f"  • 邻接大小范围: [{min(neighbor_sizes)}, {max(neighbor_sizes)}]")
    print(f"  • 平均邻接大小: {np.mean(neighbor_sizes):.2f}")
    
    # 共同邻接（采样计算）
    if len(edges) > 1000:
        sampled_edges = random.sample(edges, 1000)
    else:
        sampled_edges = edges
    
    # 快速构建邻接表用于计算共同邻居
    adj_list = [set() for _ in range(num_nodes)]
    for src, dst in sampled_edges:
        adj_list[src].add(dst)
        adj_list[dst].add(src)
    
    common_neighbor_counts = []
    for src, dst in sampled_edges:
        common = len(adj_list[src] & adj_list[dst])
        common_neighbor_counts.append(common)
    
    if common_neighbor_counts:
        print(f"  • 平均共同邻接数: {np.mean(common_neighbor_counts):.2f}")
        print(f"  • 中位共同邻接数: {np.median(common_neighbor_counts):.2f}")


def analyze_sparsity(num_nodes, num_edges, feat_dim, x, edge_index):
    """稀疏性分析"""
    print(f"\n数据稀疏性:")
    
    # 边的稀疏性
    edge_sparsity = 1 - (num_edges / (num_nodes * num_nodes))
    print(f"  • 边的稀疏度: {edge_sparsity:.6f}")
    
    # 特征的稀疏性
    feature_sparsity = (x == 0).sum().float() / (x.shape[0] * x.shape[1])
    print(f"  • 特征的稀疏度: {feature_sparsity:.6f}")
    
    # 内存占用估计
    x_memory_mb = (x.element_size() * x.numel()) / 1024 / 1024
    edge_memory_mb = (edge_index.element_size() * edge_index.numel()) / 1024 / 1024
    total_memory_mb = x_memory_mb + edge_memory_mb
    
    print(f"\n内存占用估计:")
    print(f"  • 特征矩阵: {x_memory_mb:.2f} MB")
    print(f"  • 边索引: {edge_memory_mb:.2f} MB")
    print(f"  • 总计: {total_memory_mb:.2f} MB")


def calculate_entropy(probs):
    """计算熵"""
    probs = probs[probs > 0]
    return -(probs * torch.log(probs)).sum().item()


def calculate_homophily(edge_index, y):
    """计算同质性指标"""
    edges = edge_index.t().tolist()
    same_class = sum(1 for src, dst in edges if y[src] == y[dst])
    return same_class / len(edges) if edges else 0


def calculate_modularity(G, communities):
    """计算模块性"""
    from networkx.algorithms.community import modularity
    return modularity(G, communities)


if __name__ == '__main__':
    analyze_dataset()
