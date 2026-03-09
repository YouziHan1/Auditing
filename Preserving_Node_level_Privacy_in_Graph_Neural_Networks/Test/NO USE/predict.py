# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import SAGEConv
# from tqdm import trange

# # -------------------- GraphSAGE --------------------
# class GraphSAGE(torch.nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim):
#         super().__init__()
#         self.conv1 = SAGEConv(in_dim, hidden_dim)
#         self.conv2 = SAGEConv(hidden_dim, out_dim)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         return x


# def train(model, data, train_mask, optimizer, epochs=50):
#     model.train()
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         out = model(data.x, data.edge_index)
#         loss = F.cross_entropy(out[train_mask], data.y[train_mask])
#         loss.backward()
#         optimizer.step()


# # -------------------- 主程序 --------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 1. 加载子图
# data = torch.load("amazon_subgraph_10000_singal_canary.pt", map_location=device)
# data = data.to(device)

# num_nodes = data.num_nodes
# num_classes = int(data.y.max().item()) + 1

# # 2. 训练集 mask（后 90%）
# train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
# train_mask[1000:] = True

# # 3. 统计变量
# num_runs = 1000
# canary_idx = 10000  # 第10000个节点
# count_class4 = 0
# count_class6 = 0
# prob_class6_list = []

# for run in trange(num_runs, desc="Training & Inference", ncols=90):
#     # 初始化模型
#     model = GraphSAGE(
#         in_dim=data.x.size(1),
#         hidden_dim=128,
#         out_dim=num_classes
#     ).to(device)

#     optimizer = torch.optim.Adam(
#         model.parameters(), lr=0.005, weight_decay=5e-4
#     )

#     # 训练
#     train(model, data, train_mask, optimizer, epochs=50)

#     # 推理 canary 节点
#     model.eval()
#     with torch.no_grad():
#         logits = model(data.x, data.edge_index)[canary_idx]
#         probs = F.softmax(logits, dim=0)

#         pred_class = probs.argmax().item()
#         prob_class6 = probs[6].item()

#         prob_class6_list.append(prob_class6)

#         if pred_class == 4:
#             count_class4 += 1
#         elif pred_class == 6:
#             count_class6 += 1


# # 5. 输出统计结果
# mean_prob_class6 = sum(prob_class6_list) / num_runs
# print(f"After {num_runs} runs:")
# print(f"Predicted class 4 count: {count_class4}")
# print(f"Predicted class 6 count: {count_class6}")
# print(f"Mean Class 6 probability: {mean_prob_class6:.4f}")
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from tqdm import trange

# -------------------- GraphSAGE --------------------
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def train(model, data, train_mask, optimizer, epochs=50):
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()


# -------------------- 主程序 --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载子图
data = torch.load("amazon_subgraph_10000_100_canaries.pt", map_location=device)
data = data.to(device)

num_nodes = data.num_nodes
num_classes = int(data.y.max().item()) + 1

# 2. 训练集 mask（后 90%）
train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
train_mask[1000:] = True

# -------------------- Canary 设置 --------------------
num_canaries = 100
canary_start = 10000
canary_indices = list(range(canary_start, canary_start + num_canaries))

# -------------------- 统计变量 --------------------
num_runs = 1000

count_class4 = 0
count_class6 = 0
prob_class6_all = []          # 所有 canary × 所有 run 的概率
train_acc_list = []           # 每次 run 的训练集准确率

# -------------------- 多次训练 --------------------
for run in trange(num_runs, desc="Training & Inference", ncols=90):

    model = GraphSAGE(
        in_dim=data.x.size(1),
        hidden_dim=128,
        out_dim=num_classes
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.005, weight_decay=5e-4
    )

    # 训练
    train(model, data, train_mask, optimizer, epochs=50)

    # -------------------- 推理 --------------------
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)

        # ===== 训练集准确率 =====
        train_pred = out[train_mask].argmax(dim=1)
        train_acc = (train_pred == data.y[train_mask]).float().mean().item()
        train_acc_list.append(train_acc)

        # ===== Canary 统计 =====
        for idx in canary_indices:
            logits = out[idx]
            probs = F.softmax(logits, dim=0)

            pred_class = probs.argmax().item()
            prob_class6 = probs[6].item()

            prob_class6_all.append(prob_class6)

            if pred_class == 4:
                count_class4 += 1
            elif pred_class == 6:
                count_class6 += 1


# -------------------- 输出统计结果 --------------------
total_canary_predictions = num_runs * num_canaries

mean_prob_class6 = sum(prob_class6_all) / total_canary_predictions
mean_train_acc = sum(train_acc_list) / num_runs

print("=" * 60)
print(f"Total runs: {num_runs}")
print(f"Canary nodes per run: {num_canaries}")
print(f"Total canary predictions: {total_canary_predictions}")
print("-" * 60)
print(f"Predicted class 4 count: {count_class4}")
print(f"Predicted class 6 count: {count_class6}")
print(f"Mean Class 6 probability (100 canaries avg): {mean_prob_class6:.4f}")
print(f"Mean training accuracy: {mean_train_acc:.4f}")
print("=" * 60)
