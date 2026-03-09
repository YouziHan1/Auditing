#!/usr/bin/env python3
"""
诊断工具：分析Canary Loss分布异常的原因
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def analyze_canary_loss_distribution(data_path="./Test/amazon_subgraph_black.pt"):
    """
    分析Canary Loss分布，诊断为什么未训练数据Loss也很低
    """
    # 加载数据和模型
    data = torch.load(data_path)
    print("="*70)
    print("Canary Loss分布诊断")
    print("="*70)
    
    # 基本信息
    num_nodes = data.num_nodes
    train_mask = data.train_mask
    test_mask = data.test_mask
    
    num_train = train_mask.sum().item()
    num_test = test_mask.sum().item()
    num_canaries_in = (train_mask & ~test_mask).sum().item() if hasattr(data, 'y') else 0
    
    print(f"\n【数据统计】")
    print(f"总节点数: {num_nodes}")
    print(f"训练集: {num_train} (包含IN Canary)")
    print(f"测试集: {num_test} (不含任何Canary)")
    print(f"推断的IN Canary数: {num_nodes - num_train - num_test}")
    
    # 检查特征分布
    print(f"\n【特征分析】")
    x = data.x
    print(f"特征维度: {x.shape}")
    print(f"特征范围: [{x.min():.4f}, {x.max():.4f}]")
    print(f"特征均值: {x.mean():.4f}, 标准差: {x.std():.4f}")
    
    # 分析边的分布
    print(f"\n【边的分析】")
    edge_index = data.edge_index
    print(f"总边数: {edge_index.shape[1]}")
    
    # 计算节点度数分布
    degrees = torch.zeros(num_nodes)
    degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1]))
    degrees.scatter_add_(0, edge_index[1], torch.ones(edge_index.shape[1]))
    
    print(f"平均度数: {degrees.mean():.2f}")
    print(f"度数范围: [{degrees.min():.0f}, {degrees.max():.0f}]")
    print(f"度数标准差: {degrees.std():.2f}")
    
    # 分析Canary节点的度数
    canary_start_idx = num_train + num_test
    if canary_start_idx < num_nodes:
        canary_degrees = degrees[canary_start_idx:]
        print(f"\nCanary节点度数统计:")
        print(f"  平均度数: {canary_degrees.mean():.2f}")
        print(f"  度数范围: [{canary_degrees.min():.0f}, {canary_degrees.max():.0f}]")
    
    # 分析标签分布
    if hasattr(data, 'y'):
        print(f"\n【标签分布】")
        y = data.y
        unique_labels = torch.unique(y)
        print(f"类别数: {len(unique_labels)}")
        for label in unique_labels:
            count = (y == label).sum().item()
            print(f"  类{label}: {count}个节点")
    
    print("\n" + "="*70)
    print("【诊断建议】")
    print("="*70)
    print("""
为什么未训练数据(OUT)的Loss也很低？可能原因：

1. 【模型置信度过高】
   - 模型对大部分输入都输出高置信度预测
   - 即使预测错误，如果某个类的概率很高，CrossEntropyLoss也会很低
   - 解决: 检查模型的预测概率分布 (softmax输出)

2. 【特征分布相似】
   - 原始节点和Canary节点的特征分布接近
   - 模型难以区分新旧数据
   - 解决: 使用更不同的Canary特征或添加噪声

3. 【邻接结构相似】
   - Canary连接到的随机邻居与训练邻接类似
   - 使用GNN时，邻接信息可能比特征更重要
   - 解决: 考虑连接到更不同的邻居

4. 【IN/OUT划分无效】
   - 前一半和后一半的Canary本质没有区别
   - Loss差异太小不足以进行有效的MIA
   - 解决: 调整Canary生成策略或划分方式

建议的诊断步骤：
☐ 输出模型对Canary的预测概率分布
☐ 检查不同类别的预测置信度
☐ 对比IN/OUT Canary的邻接度数分布
☐ 尝试修改Canary特征标准差或添加异常值
☐ 检查Loss计算是否正确
    """)

def generate_loss_visualization(canary_losses, canary_mask, output_path="canary_loss_distribution.png"):
    """
    生成Loss分布直方图
    """
    mid_point = len(canary_mask) // 2
    loss_in = canary_losses[:mid_point]
    loss_out = canary_losses[mid_point:]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 直方图对比
    axes[0, 0].hist(loss_in, bins=50, alpha=0.7, label='IN', color='blue', edgecolor='black')
    axes[0, 0].hist(loss_out, bins=50, alpha=0.7, label='OUT', color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Loss值')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title('IN vs OUT Loss分布对比 (直方图)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 箱线图
    axes[0, 1].boxplot([loss_in, loss_out], labels=['IN', 'OUT'])
    axes[0, 1].set_ylabel('Loss值')
    axes[0, 1].set_title('IN vs OUT Loss分布对比 (箱线图)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 累积分布函数
    sorted_in = np.sort(loss_in)
    sorted_out = np.sort(loss_out)
    axes[1, 0].plot(sorted_in, np.arange(1, len(sorted_in)+1) / len(sorted_in), 
                    label='IN', linewidth=2)
    axes[1, 0].plot(sorted_out, np.arange(1, len(sorted_out)+1) / len(sorted_out), 
                    label='OUT', linewidth=2)
    axes[1, 0].set_xlabel('Loss值')
    axes[1, 0].set_ylabel('累积概率')
    axes[1, 0].set_title('累积分布函数 (CDF)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 分位数对比
    quantiles = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
    q_in = [np.percentile(loss_in, q*100) for q in quantiles]
    q_out = [np.percentile(loss_out, q*100) for q in quantiles]
    
    x = np.arange(len(quantiles))
    width = 0.35
    axes[1, 1].bar(x - width/2, q_in, width, label='IN', color='blue', alpha=0.7)
    axes[1, 1].bar(x + width/2, q_out, width, label='OUT', color='red', alpha=0.7)
    axes[1, 1].set_xlabel('分位数')
    axes[1, 1].set_ylabel('Loss值')
    axes[1, 1].set_title('分位数对比')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f'{q:.0%}' for q in quantiles])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 可视化已保存: {output_path}")

if __name__ == '__main__':
    analyze_canary_loss_distribution()
    print("\n如需生成Loss分布图表，请使用:")
    print("  from analyze_canary_loss import generate_loss_visualization")
    print("  generate_loss_visualization(canary_losses, canary_mask)")
