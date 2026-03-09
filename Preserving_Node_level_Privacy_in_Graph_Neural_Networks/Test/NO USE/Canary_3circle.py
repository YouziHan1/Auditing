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

# ========== 2. 选择 canary 的目标类别 ==========
canary_class = 4   # 你可以改成 1~9 中任意一个
class_idx = (y == canary_class).nonzero(as_tuple=True)[0]
anchor_node = int(class_idx[0])    # 找一个真实节点作为锚点

print(f"Canary class = {canary_class}, anchor node = {anchor_node}")

# ========== 3. 构造 canary 特征 ==========
# 使用该类别的均值特征 + 极小扰动
class_mean = x[class_idx].mean(dim=0)

epsilon = 0.01
canary_feats = []
for _ in range(3):
    noise = epsilon * torch.randn(feat_dim)
    canary_feats.append(class_mean + noise)

canary_feats = torch.stack(canary_feats)   # [3, F]

# ========== 4. 构造标签 ==========
canary_labels = torch.full((3,), 4, dtype=y.dtype)

# ========== 5. 拼接节点 ==========
x_new = torch.cat([x, canary_feats], dim=0)
y_new = torch.cat([y, canary_labels], dim=0)

canary_ids = list(range(num_nodes, num_nodes + 3))

# ========== 6. 构造 clique 结构 ==========
new_edges = []

# clique 内部全连接
for i in canary_ids:
    for j in canary_ids:
        if i != j:
            new_edges.append([i, j])

# 与 anchor 节点连接（双向）
for i in canary_ids:
    new_edges.append([i, anchor_node])
    new_edges.append([anchor_node, i])

new_edges = torch.tensor(new_edges, dtype=torch.long).t()  # [2, E_new]

edge_index_new = torch.cat([edge_index, new_edges], dim=1)

# ========== 7. 生成新 Data ==========
data_canary = Data(
    x=x_new,
    y=y_new,
    edge_index=edge_index_new
)

print(f"New graph: {data_canary.num_nodes} nodes")
print(f"Inserted canary node ids: {canary_ids}")

# ========== 8. 保存 ==========
torch.save(data_canary, "amazon_subgraph_10000_with_canary.pt")
print("Saved as amazon_subgraph_10000_with_canary.pt")
