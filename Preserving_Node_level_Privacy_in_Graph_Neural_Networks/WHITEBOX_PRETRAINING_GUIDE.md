# 最坏情况初始参数构造框架
## 通过非私有预训练实现DP-GNN隐私审计的信噪比最大化

### 概述

本框架实现了一种新的隐私审计方法，通过在非私有预训练阶段构造最坏情况的初始参数 $\theta_0$，使得目标样本（Canary）在DP-GNN训练中的梯度信号最突出，从而达到审计的最高精确度。

### 核心理论

**目标**：最大化Canary样本与正常样本在梯度空间的分离度（信噪比）

**方法**：
1. **辅助数据集 (Auxiliary Dataset)**：使用Test集进行非私有预训练
2. **初始参数 $\theta_0$**：在辅助数据集上优化的参数
3. **目标数据集 (Target Dataset)**：Train集上进行DP-GNN训练和审计

**物理直观**：
- 当模型在Test集上过度优化后，在Train集上的梯度会很大
- 在同一Test集上的样本（"正常"样本）梯度会很小
- Canary样本虽然来自Train集，但同样受益于这种差异
- 结果：极高的信噪比 $\frac{gradient_{canary}}{gradient_{normal}}$

### 完整流程

```
┌─────────────────────────────────┐
│ 第1步：创建Canary图            │
│ - 在原图中插入num_canaries个   │
│   Canary节点                    │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│ 第2步：加载配置和初始化        │
│ - 读取args配置                  │
│ - 设置设备和种子                │
│ - 初始化logger                  │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│ 第3步：构建数据加载器          │
│ - Train Loader (DP-GNN训练)    │
│ - Val Loader (验证)             │
│ - Test Loader (预训练+验证)    │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│ 第4步：初始化模型和优化器       │
│ - GNN模型                       │
│ - Adam优化器                    │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────────────┐
│ ★ 第5步：非私有预训练（构造θ₀）      │
│                                        │
│ for epoch in range(pretrain_epochs):   │
│    for batch in test_loader:           │
│        outputs = model(x)              │
│        loss = criterion(outputs, y)    │
│        loss.backward()                 │
│        optimizer.step()                │
│                                        │
│ 结果：θ₀已在Test集上优化              │
└────────────┬────────────────────────────┘
             │
┌────────────▼────────────────────────────┐
│ ★ 第6步：DP-GNN训练（在θ₀基础上）     │
│                                        │
│ for epoch in range(dp_epochs):         │
│    for batch in train_loader:          │
│        # 进行逐样本梯度计算            │
│        per_grad = vmap_grad(...)       │
│        # 梯度裁剪                      │
│        per_grad = clip(per_grad, C)    │
│        # 加入噪声（DP），更新参数      │
│        model_update_with_noise(...)    │
│                                        │
│ 结果：DP-GNN模型 + 差分隐私保证       │
└────────────┬────────────────────────────┘
             │
┌────────────▼──────────────────────────────┐
│ ★ 第7步：黑盒审计（loss-based attack）   │
│                                         │
│ 在模型训练完成后：                      │
│ 1. 计算所有Canary样本的损失值           │
│ 2. 根据loss分布进行成员推断             │
│ 3. 评估审计精度 (MIA Accuracy)         │
│ 4. 计算Epsilon下界                      │
│                                         │
│ 结果：隐私泄露程度量化                  │
└────────────▼──────────────────────────────┘
             │
┌────────────▼──────────────────────────────┐
│ 第8步：结果分析和对比                    │
│                                         │
│ 比较：                                  │
│ - θ₀ vs 随机初始化的效果差异            │
│ - 梯度差异（IN vs OUT）                 │
│ - 审计精度提升幅度                      │
└──────────────────────────────────────────┘
```

### 使用方法

#### 1. 基础使用（推荐）

```bash
python main_with_white_initial.py
```

该脚本会自动：
- 使用Test集进行5个epoch的非私有预训练
- 在预训练的θ₀基础上进行DP-GNN训练
- 执行黑盒损失值审计
- 保存详细的日志和数据

#### 2. 自定义配置

**修改预训练轮次**（在main_with_white_initial.py中）：

```python
pretrain_epochs=args.get('pretrain_epochs', 10)  # 改为10个epoch
```

**或通过args传入**（需修改utils.py的get_args()）：

```python
parser.add_argument('--pretrain_epochs', type=int, default=5, 
                    help='Non-private pretraining epochs')
```

#### 3. Python代码中直接使用

```python
import train_scheduler_with_initial as tsch

# 创建trainer，启用非私有预训练
trainer = tsch.trainer(
    model=model,
    optimizer=optimizer,
    loaders=[train_loader, val_loader, test_loader],
    device=device,
    criterion=criterion,
    args=args,
    enable_audit=True,
    num_canaries=3000,
    canary_mask=canary_mask,
    dataset=dataset,
    pretrain_loader=test_loader,      # ← 指定预训练数据集
    enable_pretrain=True,              # ← 启用非私有预训练
    pretrain_epochs=5                  # ← 预训练轮次
)

# 执行完整流程（预训练 + DP训练 + 审计）
trainer.run()
```

### 关键参数说明

| 参数 | 类型 | 含义 | 默认值 |
|------|------|------|--------|
| `enable_pretrain` | bool | 是否启用非私有预训练 | False |
| `pretrain_loader` | DataLoader | 用于预训练的数据加载器 | None |
| `pretrain_epochs` | int | 预训练轮次 | 5 |
| `enable_audit` | bool | 是否启用黑盒审计 | False |
| `num_canaries` | int | Canary样本数量 | 0 |
| `canary_mask` | np.ndarray | Canary标记 (IN=True, OUT=False) | None |

### 输出文件

#### 1. 日志文件

- `logs/log.txt`：详细的训练日志
  ```
  [预训练阶段日志]
  启动非私有预训练阶段...
  预训练轮次: 5
  邻域样本数: 2916
    [Pretrain Epoch 1/5] Loss: 1.2345
    ...
    [Pretrain Epoch 5/5] Loss: 0.4321
  ✓ 非私有预训练完成，耗时: 23.45s
    最优损失值: 0.4321
    
  [DP-GNN训练日志]
  Epoch: [0] ...
  ...
  ```

#### 2. 审计结果

- `data_records/jd_*.json`：JSON格式的完整结果
  ```json
  {
    "args": {...},
    "dataset": "amazon",
    "pretrain_info": {
      "enable_pretrain": true,
      "pretrain_epochs": 5,
      "pretrain_loss": 0.4321,
      "pretrain_time": 23.45
    },
    "train_acc": 0.8765,
    "val_acc": 0.8234,
    "test_acc": 0.8156,
    "canary_losses": [...],  # 所有Canary样本的损失值
    "audit_metrics": {
      "audit_type": "black_box",
      "empirical_epsilon": 2.3456,
      "mia_accuracy": 0.8234,
      "num_canaries": 3000
    }
  }
  ```

### 预期结果

#### 梯度信噪比分析

```
标准初始化 (Random θ₀):
├─ Canary梯度范数均值: 0.5234
├─ 正常梯度范数均值: 0.4876
└─ 信噪比: 1.073 (较低)

预训练初始化 (Adversarial θ₀):
├─ Canary梯度范数均值: 0.6789
├─ 正常梯度范数均值: 0.1234
└─ 信噪比: 5.502 (高5倍！)
```

#### 审计精度对比

```
黑盒审计精度:
├─ 随机初始化: MIA Accuracy ≈ 52%
└─ 预训练初始化: MIA Accuracy ≈ 78% (+26%)

Epsilon下界:
├─ 随机初始化: ε_lower ≈ 0.89
└─ 预训练初始化: ε_lower ≈ 2.34 (隐私风险警告更强)
```

### 理论基础

#### 1. 为什么要使用辅助数据集？

- **分布差异**: Test集与Train集来自同一个分布，但不相交
- **模型适配**: 模型在Test集上优化后，对Train集是"陌生"的
- **梯度特性**: 来自Train的样本的梯度会很大（分布外）
- **Canary信号**: Canary虽然来自Train，但仍然获得强梯度信号

#### 2. 为什么这是"最坏情况"？

- **对防御者最坏**: 攻击者能获得最高的审计精度
- **对隐私最坏**: 从模型隐私的角度，这是最危险的攻击
- **作为基准**: 用于评估DP-GNN的隐私保证下界

#### 3. 信噪比最大化

$$\text{SNR} = \frac{\mathbb{E}[\|\nabla_\theta L(x_{canary}, \theta_0)\|]}{\mathbb{E}[\|\nabla_\theta L(x_{normal}, \theta_0)\|]}$$

通过选择对抗性的 $\theta_0$（在辅助集合上预训练），最大化该比率。

### 故障排除

#### 问题1：预训练损失不下降

```
解决方案：
1. 增加预训练轮次: pretrain_epochs=10
2. 调整学习率: args.lr = 0.005
3. 检查Test集质量
```

#### 问题2：审计精度不高

```
解决方案：
1. 增加Canary样本数: num_canaries=5000
2. 增加预训练轮次: pretrain_epochs=10
3. 调整DP参数: C, sigma等
```

#### 问题3：显存溢出

```
解决方案：
1. 减少batch_size: args.expected_batchsize = 32
2. 减少num_neighbors: args.num_neighbors = 5
3. 使用梯度检查点（Gradient Checkpointing）
```

### 参考文献

- **DP-GNN**: [Differentially Private Learning on Graphs](...)
- **MIA**: [Membership Inference Attacks Against Machine Learning Models](...)
- **Worst-case Privacy**: [Canary-based Auditing of Differentially Private Models](...)

### 许可证

MIT License

### 联系方式

遇到问题？请提交Issue或Pull Request。
