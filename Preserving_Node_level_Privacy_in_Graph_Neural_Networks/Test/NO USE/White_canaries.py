import torch
from torch_geometric.data import Data
import os
import random

# ========== 特征管理函数 ==========
def save_canary_features(canaries_feats, save_path="./Test/canary_features_white.pt"):
    """保存Canary特征以便复用（只保存特征张量）"""
    # 如果输入是列表，提取特征张量
    if isinstance(canaries_feats, list) and len(canaries_feats) > 0:
        feats_tensor = torch.cat(canaries_feats, dim=0)
    else:
        feats_tensor = canaries_feats
    torch.save(feats_tensor, save_path)
    print(f"✓ Canary features saved to: {save_path} ({feats_tensor.shape[0]} features)")

def load_canary_features(load_path="./Test/canary_features_white.pt"):
    """加载已保存的Canary特征"""
    if os.path.exists(load_path):
        canaries_feats = torch.load(load_path)
        print(f"✓ Canary features loaded from: {load_path} ({len(canaries_feats)} features)")
        return canaries_feats
    return None

def get_canary_features(num_canaries, feat_dim, canaries_file="./Test/canary_features_white.pt"):
    """获取Canary特征，支持加载缓存或生成新特征"""
    cached_feats = load_canary_features(canaries_file)
    
    if cached_feats is None:
        print(f"Generating new canary features (not found in {canaries_file})...")
        # 生成新特征（全0）
        canaries_feats = torch.zeros(num_canaries, feat_dim)
        # 保存特征
        save_canary_features(canaries_feats, canaries_file)
    else:
        print(f"Using cached canary features...")
        # 处理数量不匹配的情况
        num_cached = len(cached_feats)
        if num_cached > num_canaries:
            print(f"  Warning: Cached {num_cached} features > needed {num_canaries}")
            canaries_feats = cached_feats[:num_canaries]
        elif num_cached < num_canaries:
            print(f"  Warning: Cached {num_cached} features < needed {num_canaries}")
            print(f"  Generating {num_canaries - num_cached} new ones (全0)")
            new_feats = torch.zeros(num_canaries - num_cached, feat_dim)
            canaries_feats = torch.cat([cached_feats, new_feats], dim=0)
        else:
            canaries_feats = cached_feats
    
    return canaries_feats

def build_audit_dataset(original_path="amazon_subgraph_all.pt", save_path="./Test/white_audit_dataset.pt", canaries_file="./Test/canary_features_white.pt"):
    print(f"审计数据集: {original_path} -> {save_path} ===")
    
    data = torch.load(original_path)
    x, y, edge_index = data.x.clone(), data.y.clone(), data.edge_index.clone()

    num_nodes_original, feat_dim = x.size()
    print(f"原始节点数: {num_nodes_original}")

    # 2. 金丝雀配置
    num_canaries = 3000
    num_in = num_canaries//2   # 参与训练 (Members)
    num_out = num_canaries - num_in  # 不参与训练 (Non-Members)
    canary_class = 6

    # 【新增】预加载金丝雀特征
    print("\n=== 加载金丝雀特征 ===")
    canaries_feats = get_canary_features(num_canaries, feat_dim, canaries_file)
    print(f"特征形状: {canaries_feats.shape}")

    # 3. 构造金丝雀节点
    x_new = x
    y_new = y
    new_edges = []
    
    # 初始化 ID 数组：正常节点 ID >= 0
    sample_ids = torch.arange(num_nodes_original + num_canaries, dtype=torch.long)
    
    print("\n正在植入金丝雀...")
    for i in range(num_canaries):
        real_idx = num_nodes_original + i
        
        # 【关键】金丝雀 ID 设为负数: -1, -2, ...
        sample_ids[real_idx] = -1 - i
        
        # 【修改】使用预加载的特征而不是临时生成
        canary_feat = canaries_feats[i:i+1]  # [1, feat_dim]
        canary_label = torch.tensor([canary_class], dtype=y.dtype)
        
        x_new = torch.cat([x_new, canary_feat], dim=0)
        y_new = torch.cat([y_new, canary_label], dim=0)
        
        # 连接到随机锚点
        anchor = torch.randint(0, num_nodes_original, (1,)).item()
        new_edges.append([real_idx, anchor])
        new_edges.append([anchor, real_idx])

    # 4. 更新边
    if len(new_edges) > 0:
        new_edges = torch.tensor(new_edges, dtype=torch.long).t()
        edge_index_new = torch.cat([edge_index, new_edges], dim=1)
    else:
        edge_index_new = edge_index

    # 5. 设置 Masks
    num_all = x_new.shape[0]
    train_mask = torch.zeros(num_all, dtype=torch.bool)
    val_mask = torch.zeros(num_all, dtype=torch.bool)
    test_mask = torch.zeros(num_all, dtype=torch.bool)

    # 正常节点划分 (示例: 前80%训练，后20%测试)
    train_end = int(num_nodes_original * 0.8)
    train_mask[0:train_end] = True
    test_mask[train_end:num_nodes_original] = True  # 【关键】test_mask 只包含原始节点，不包含金丝雀

    # 【关键】设置金丝雀的 IN/OUT
    # 前 num_in 个金丝雀加入训练集
    canary_indices = torch.arange(num_nodes_original, num_all)
    train_mask[canary_indices[:num_in]] = True
    
    # 后 num_out 个金丝雀保持 False (OUT) 
    # 金丝雀不参与 train 或 test，仅在白盒审计时使用
    
    print(f"总节点数: {num_all}")
    print(f"训练集大小: {train_mask.sum().item()} (含 {num_in} 个金丝雀)")
    print(f"OUT金丝雀数: {num_out}")

    # 6. 保存
    data_new = Data(
        x=x_new, 
        edge_index=edge_index_new, 
        y=y_new, 
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        sample_ids=sample_ids # 必须保存!
    )
    
    torch.save(data_new, save_path)
    print(f"数据集已保存至: {save_path}")

if __name__ == "__main__":
    build_audit_dataset()