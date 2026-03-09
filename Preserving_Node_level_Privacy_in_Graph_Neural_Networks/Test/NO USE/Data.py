import torch
import numpy as np

data = torch.load("amazon_subgraph_10000.pt")
x = data.x.cpu()        # [N, F]
y = data.y.cpu()        # [N]
num_classes = int(y.max().item()) + 1

class_means = []

for c in range(num_classes):
    idx = (y == c)
    class_x = x[idx]
    mean_vec = class_x.mean(dim=0)
    class_means.append(mean_vec)

class_means = torch.stack(class_means)  # [C, F]

print("Class mean feature matrix shape:", class_means.shape)
print("Class mean feature matrix:", class_means)

#类内方差
for c in range(num_classes):
    idx = (y == c)
    var = x[idx].var(dim=0).mean()
    print(f"Class {c} intra-class variance: {var:.4f}")

from itertools import combinations

#类间距离
for i, j in combinations(range(num_classes), 2):
    dist = torch.norm(class_means[i] - class_means[j], p=2)
    print(f"Distance between class {i} and {j}: {dist:.4f}")
