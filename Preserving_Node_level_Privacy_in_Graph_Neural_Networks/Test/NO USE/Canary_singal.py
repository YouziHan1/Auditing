import torch
from torch_geometric.data import Data

# ========== 1. 加载原始子图 ==========
data = torch.load("amazon_subgraph_10000.pt")

x = data.x.clone()                 # [N, F]
y = data.y.clone()                 # [N]
edge_index = data.edge_index.clone()  # [2, E]

num_nodes, feat_dim = x.size()
num_classes = int(y.max().item()) + 1

print(f"Original graph: {num_nodes} nodes")

#特征很像4，但是类别和邻居为6
canary_class = 6
target_class = 4
class_idx = (y == target_class).nonzero(as_tuple=True)[0]#类别4的所有节点
class_idx_6 = (y == canary_class).nonzero(as_tuple=True)[0]#类别6的所有节点
  # 找一个真实节点作为锚点
anchor_node = int(class_idx_6[0])  
# ========== 3. 构造 canary 特征 ==========
# # 使用4类别的均值特征 + 极小扰动
# class_mean = x[class_idx].mean(dim=0)
# epsilon = 0.5
# canary_feats = class_mean + epsilon * torch.randn(feat_dim)
# canary_feats = canary_feats.unsqueeze(0)   # [1, F]

# 使用随机特征
canary_feats = torch.randn(feat_dim)
canary_feats = canary_feats.unsqueeze(0)   # [1, F]

# ========== 4. 构造标签 ==========
canary_labels = torch.full((1,), canary_class, dtype=y.dtype)
# ========== 5. 拼接节点 ==========
x_new = torch.cat([x, canary_feats], dim=0)
y_new = torch.cat([y, canary_labels], dim=0)
canary_id = num_nodes
# ========== 6. 构造 clique 结构 ==========
new_edges = []
# 与 anchor 节点连接（双向）
new_edges.append([canary_id, anchor_node])
new_edges.append([anchor_node, canary_id])
new_edges = torch.tensor(new_edges, dtype=torch.long).t()  # [2, E_new]
edge_index_new = torch.cat([edge_index, new_edges], dim=1)
# ========== 7. 保存新图 ==========
data_new = Data(
    x=x_new,
    edge_index=edge_index_new,
    y=y_new
)
torch.save(data_new, "amazon_subgraph_10000_singal_canary.pt")
print(f"New graph with canary saved: {data_new.num_nodes} nodes")

