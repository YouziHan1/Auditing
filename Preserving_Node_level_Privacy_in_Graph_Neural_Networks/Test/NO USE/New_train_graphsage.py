import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv



# GraphSAGE

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
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            pred = out.argmax(dim=1)
            acc = (pred[train_mask] == data.y[train_mask]).float().mean()
            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Train Acc {acc:.4f}")



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # 1. 加载子图
    data = torch.load("amazon_subgraph_10000_singal_canary.pt", map_location=device)
    data = data.to(device)

    num_nodes = data.num_nodes
    num_classes = int(data.y.max().item()) + 1

    print(f"Loaded subgraph with {num_nodes} nodes, {num_classes} classes")


    # 2. 后90%个节点作为训练集
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[1000:] = True


    # 3. 初始化模型
    model = GraphSAGE(
        in_dim=data.x.size(1),
        hidden_dim=128,
        out_dim=num_classes
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.005, weight_decay=5e-4
    )


    # 4. 训练
    print("\nStart training...")
    train(model, data, train_mask, optimizer, epochs=50)
    print("Training finished.")

    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask[:1000] = True

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_acc = (pred[test_mask] == data.y[test_mask]).float().mean()

    print(f"\nTest Accuracy (nodes 0-999): {test_acc:.4f}")
    
    # 5. 预测
    model.eval()
    print("\nInteractive prediction mode")
    print("Enter node index (0–9999), or 'q' to quit\n")

    while True:
        s = input("Query node index > ").strip()

        if s.lower() in ("q", "quit", "exit"):
            print("Bye.")
            break

        if not s.isdigit():
            print("Please input a valid integer.")
            continue

        idx = int(s)
        if idx < 0 or idx >= num_nodes:
            print(f"Index out of range [0, {num_nodes - 1}]")
            continue

        with torch.no_grad():
            logits = model(data.x, data.edge_index)[idx]
            probs = F.softmax(logits, dim=0)

        pred_class = probs.argmax().item()
        true_class = data.y[idx].item()

        top3_prob, top3_class = torch.topk(probs, 3)

        node_type = "TRAIN" if idx > 999 else "NON-TRAIN"

        print("\n--- Prediction Result ---")
        print(f"Node index      : {idx} ({node_type})")
        print(f"True label      : {true_class}")
        print(f"Predicted label : {pred_class}")
        print("Top-3 probabilities:")
        for i in range(3):
            print(
                f"  Class {top3_class[i].item():2d} "
                f"Probability {top3_prob[i].item():.4f}"
            )
        print()


if __name__ == "__main__":
    main()
