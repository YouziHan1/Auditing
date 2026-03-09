#本文方案 加载数据、构建模型、启动默认训练流程
import torch
import time
import datasets.SETUP as SETUP

import datasets.utils as dms_utils
import datasets.model as dms_model

import utils
#import train_scheduler as tsch
import train_scheduler_with_audit as tsch  # 改为审计版本

def evaluate_canary_nodes(model, dataset, device, num_canaries=100):
    """
    在训练结束后评估最后100个Canary节点的预测正确率和Loss值
    """
    print("\n" + "="*60)
    print("Evaluating Canary Nodes (Last 100 nodes)")
    print("="*60)
    
    # 获取完整图数据
    graph_data = dataset[0]
    
    # 最后100个节点的索引
    total_nodes = graph_data.num_nodes
    canary_start = total_nodes - num_canaries
    canary_indices = list(range(canary_start, total_nodes))
    
    # 准备数据 - 只需要特征
    x = graph_data.x.to(device)
    y_true = graph_data.y[canary_indices].to(device)
    
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    # 模型推理
    model.eval()
    with torch.no_grad():
        # 只对Canary节点的特征进行预测
        canary_x = x[canary_indices]
        # 前向传播
        canary_pred = model(canary_x)
        # 获取预测类别
        y_pred = canary_pred.argmax(dim=1)
        # 计算每个样本的Loss
        losses = criterion(canary_pred, y_true)
    
    # 计算准确率
    correct = (y_pred == y_true).sum().item()
    accuracy = correct / num_canaries * 100
    
    # 计算Loss统计
    avg_loss = losses.mean().item()
    min_loss = losses.min().item()
    max_loss = losses.max().item()
    std_loss = losses.std().item()
    
    # 统计每个类别的预测情况
    print(f"\nCanary nodes range: [{canary_start}, {total_nodes-1}]")
    print(f"Total canary nodes: {num_canaries}")
    print(f"Correct predictions: {correct}")
    print(f"Canary Accuracy: {accuracy:.2f}%")
    
    # Loss统计
    print(f"\n{'='*60}")
    print("Loss Statistics for Canary Nodes")
    print(f"{'='*60}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Min Loss: {min_loss:.4f}")
    print(f"Max Loss: {max_loss:.4f}")
    print(f"Std Loss: {std_loss:.4f}")
    
    # 详细统计
    print(f"\n{'='*60}")
    print("Label Distribution")
    print(f"{'='*60}")
    print(f"True labels distribution:")
    unique_true, counts_true = torch.unique(y_true, return_counts=True)
    for label, count in zip(unique_true.cpu().numpy(), counts_true.cpu().numpy()):
        print(f"  Class {label}: {count} nodes")
    
    print(f"\nPredicted labels distribution:")
    unique_pred, counts_pred = torch.unique(y_pred, return_counts=True)
    for label, count in zip(unique_pred.cpu().numpy(), counts_pred.cpu().numpy()):
        print(f"  Class {label}: {count} nodes")
    
    # 每个类别的平均Loss
    print(f"\n{'='*60}")
    print("Per-Class Loss and Accuracy")
    print(f"{'='*60}")
    for cls in unique_true.cpu().numpy():
        cls_mask = (y_true == cls)
        cls_correct = ((y_pred == y_true) & cls_mask).sum().item()
        cls_total = cls_mask.sum().item()
        cls_acc = cls_correct / cls_total * 100 if cls_total > 0 else 0
        
        # 计算该类别的平均Loss
        cls_losses = losses[cls_mask]
        cls_avg_loss = cls_losses.mean().item() if cls_total > 0 else 0
        cls_std_loss = cls_losses.std().item() if cls_total > 0 else 0
        
        print(f"Class {cls}:")
        print(f"  Accuracy: {cls_correct}/{cls_total} ({cls_acc:.2f}%)")
        print(f"  Avg Loss: {cls_avg_loss:.4f} (±{cls_std_loss:.4f})")
    
    print("="*60 + "\n")
    
    return accuracy

if __name__ == '__main__':
    s_time  = time.time()
    #读取参数与环境
    args = utils.get_args()
    SETUP.setup_seed(args.seed)
    device = SETUP.get_device()

    #读取数据集，划分训练/测试集，构造 DataLoader
    train_loader, val_loader, test_loader, dataset, x = dms_utils.form_loaders(args)
    args.num_classes = dataset.num_classes#类别数

    model = dms_model.G_net(K = args.K, feat_dim = x.shape[1], num_classes = args.num_classes, hidden_channels = 128)#模型构建
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, )
    # optimizer = torch.optim.SGD(model.parameters(), lr = 1, momentum = 0.0)

    # args.optimizer = str(optimizer)
    #DP-SGD训练
    train_master = tsch.trainer(
                        model = model,
                        optimizer = optimizer,
                        loaders = [train_loader, None, test_loader],
                        device = device,
                        criterion = dms_model.criterion,
                        args = args,
                    )
    
    train_master.run()
    print(f'\n==> ToTaL TiMe FoR OnE RuN: {time.time() - s_time:.4f}\n')
    
    # # ========== 新增：评估Canary节点 ==========
    # canary_accuracy = evaluate_canary_nodes(model, dataset, device, num_canaries=100)
    # print(f'\n==> Canary Node Accuracy: {canary_accuracy:.2f}%\n\n\n')


