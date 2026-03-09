import torch
from torch_geometric.datasets import Amazon

# 加载原始数据集
dataset = Amazon(root='./data/Amazon', name='Computers')
graph = dataset[0]  # 这就是一个 Data 对象，包含了 x, edge_index, y

# 确保数据在 CPU 上（方便跨设备加载）
graph = graph.cpu()

# 直接保存整个 Data 对象
torch.save(graph, './Test/amazon_subgraph_all.pt')

print(f"备份完成！")
print(f"节点数: {graph.num_nodes}")
print(f"边数: {graph.num_edges}")
print(f"特征维度: {graph.num_node_features}")