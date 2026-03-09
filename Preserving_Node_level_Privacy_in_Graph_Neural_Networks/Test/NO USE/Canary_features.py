import torch
from torch_geometric.data import Data
import os
import numpy as np

# ========== 0. 对抗性特征生成函数 ==========
def identify_vulnerable_dimensions(x, num_vulnerable=None, percentile=60):
    """
    第一阶段：识别脆弱维度 (Identify Vulnerable Dimensions)
    
    计算每个特征维度的均值距离边界的距离δ
    δ_j = max(|μ_j - α_j|, |β_j - μ_j|)
    
    Args:
        x: 特征矩阵 [N, F]
        num_vulnerable: 要选择的脆弱维度数，默认为总维度数的20%
        percentile: 百分位数，用于设定α和β的范围
    
    Returns:
        vulnerable_dims: 脆弱维度的索引列表（按δ降序）
        stats: 统计信息字典
    """
    N, F = x.shape
    
    # 计算每个维度的统计特性
    mu = x.mean(dim=0)  # 均值 [F]
    alpha = x.min(dim=0)[0]  # 下界
    beta = x.max(dim=0)[0]   # 上界
    
    # 计算δ（到边界的距离）
    delta = torch.maximum(torch.abs(mu - alpha), torch.abs(beta - mu))
    
    # 按δ降序排列
    sorted_indices = torch.argsort(delta, descending=True)
    
    # 选择前z个脆弱维度
    if num_vulnerable is None:
        num_vulnerable = max(1, int(F * percentile / 100))
    
    vulnerable_dims = sorted_indices[:num_vulnerable].tolist()
    
    print(f"\n{'='*60}")
    print(f"第一阶段：识别脆弱维度 (Identify Vulnerable Dimensions)")
    print(f"{'='*60}")
    print(f"总维度数: {F}")
    print(f"选择脆弱维度数: {num_vulnerable} ({percentile}%)")
    print(f"脆弱维度 (前5个): {vulnerable_dims[:5]}")
    print(f"δ值范围: [{delta.min():.4f}, {delta.max():.4f}]")
    print(f"{'='*60}\n")
    
    stats = {
        'mu': mu,
        'alpha': alpha,
        'beta': beta,
        'delta': delta,
        'vulnerable_dims': vulnerable_dims,
    }
    
    return vulnerable_dims, stats


def synthesize_adversarial_features(num_canaries, x, vulnerable_dims, stats, 
                                     m=None, lambda_M=None):
    """
    第二阶段：特征合成 (Feature Synthesis) - 最大化统计偏差
    
    攻击者直接构造位于LDP机制输出域中的数值，以最大化统计偏差。
    
    Args:
        num_canaries: 要生成的金丝雀特征数
        x: 原始特征矩阵 [N, F]，用于获取特征范围
        vulnerable_dims: 脆弱维度列表
        stats: 统计信息字典
        m: 每个canary要攻击的维度数，默认为脆弱维度总数的50%
        lambda_M: 默认填充值（对于非采样维度），默认为随机数
    
    Returns:
        canary_feats: 对抗性特征矩阵 [num_canaries, F]
    """
    N, F = x.shape
    
    # 获取统计信息
    mu = stats['mu']
    alpha = stats['alpha']
    beta = stats['beta']
    
    # LDP输出域范围
    L_M = alpha.min().item()  # 最小值
    H_M = beta.max().item()   # 最大值
    
    # 设置攻击维度数
    if m is None:
        m = max(1, len(vulnerable_dims) // 2)
    
    # 如果未指定默认值，使用LDP输出域内的随机数
    if lambda_M is None:
        use_random_default = True
    else:
        use_random_default = False
    
    canary_feats = torch.zeros(num_canaries, F, dtype=x.dtype)
    
    print(f"{'='*60}")
    print(f"第二阶段：特征合成 (Feature Synthesis)")
    print(f"{'='*60}")
    print(f"每个canary要攻击的维度数: {m} / {len(vulnerable_dims)}")
    print(f"LDP输出域范围: [{L_M:.4f}, {H_M:.4f}]")
    print(f"{'='*60}\n")
    
    # 为每个canary生成对抗性特征
    for i in range(num_canaries):
        # 初始化为默认值λ_M（随机数或指定值）
        if use_random_default:
            canary_feats[i, :] = torch.FloatTensor(F).uniform_(L_M, H_M)
        else:
            canary_feats[i, :] = lambda_M
        
        # 采样S_v：从脆弱维度中随机抽取m个维度
        selected_dims = np.random.choice(vulnerable_dims, size=min(m, len(vulnerable_dims)), 
                                         replace=False).tolist()
        
        # 反向极值注入：Distance Maximization
        for j in selected_dims:
            # 计算该维度的中间点
            mid_j = (alpha[j].item() + beta[j].item()) / 2
            mu_j = mu[j].item()
            
            # 如果原均值偏小，注入最大值；反之注入最小值
            if mu_j < mid_j:
                canary_feats[i, j] = H_M  # 最大值
            else:
                canary_feats[i, j] = L_M  # 最小值
    
    print(f"✓ 生成了 {num_canaries} 个对抗性canary特征")
    print(f"特征矩阵形状: {canary_feats.shape}\n")
    
    return canary_feats


def save_canary_features(canaries_data, save_path="./Test/canary_features1.pt"):
    """保存Canary特征以便复用（只保存特征张量）"""
    # 提取特征张量
    canaries_feats = torch.cat([feat for feat, _, _ in canaries_data], dim=0)
    torch.save(canaries_feats, save_path)
    print(f"✓ Canary features saved to: {save_path} ({canaries_feats.shape[0]} features)")

def load_canary_features(load_path="./Test/canary_features1.pt"):
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

    # 所有该类别节点（在添加canary前保存，用作锚点的候选）
    class_idx = (y == canary_class).nonzero(as_tuple=True)[0].tolist()  # 转为列表以避免后续问题
    print(f"Found {len(class_idx)} nodes in class {canary_class}")
    # 处理锚点不足的情况（允许循环选择）
    if len(class_idx) < num_base_canaries:
        print(f"Warning: Only {len(class_idx)} anchor nodes available, will use circular selection")

    # ========== 3. 尝试加载已保存的Canary特征 ==========
    canaries_file = "./Test/canary_features1.pt"
    cached_feats = load_canary_features(canaries_file)
    
    if cached_feats is None:
        print(f"Generating new canary features using adversarial strategy (not found in {canaries_file})...")
        
        # 第一阶段：识别脆弱维度
        vulnerable_dims, stats = identify_vulnerable_dimensions(x, percentile=50)
        
        # 第二阶段：特征合成 - 最大化统计偏差
        # 为所有canary生成对抗性特征（包括重复）
        num_all_canaries = num_base_canaries * num_repeats
        canaries_feats = synthesize_adversarial_features(
            num_canaries=num_base_canaries,  # 只需生成基础数量，重复时会复用
            x=x,
            vulnerable_dims=vulnerable_dims,
            stats=stats,
            m=None  # 默认使用脆弱维度的50%
        )
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
            
            # 为缺失部分生成新特征
            vulnerable_dims, stats = identify_vulnerable_dimensions(x, percentile=20)
            additional_feats = synthesize_adversarial_features(
                num_canaries=num_base_canaries - num_cached,
                x=x,
                vulnerable_dims=vulnerable_dims,
                stats=stats,
                m=None
            )
            canaries_feats = torch.cat([canaries_feats, additional_feats], dim=0)
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
    for i in range(num_base_canaries):
        # 处理锚点选择（按顺序选择）
        if len(class_idx) > 0:
            anchor_idx = i % len(class_idx)  # 循环选择
            anchor_node = int(class_idx[anchor_idx])
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
            # 所有重复的金丝雀都连接到原图中的同一锚点节点
            new_edges.append([canary_id, anchor_node])
            new_edges.append([anchor_node, canary_id])
        
        print(f"重复{repeat_idx}: ID范围 [{num_nodes + repeat_idx * num_base_canaries}, {num_nodes + (repeat_idx + 1) * num_base_canaries - 1}] -> 连接到原图锚点")
    
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
