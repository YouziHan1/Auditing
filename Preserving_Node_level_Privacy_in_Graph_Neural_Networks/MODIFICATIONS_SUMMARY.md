# 分位数回归器改进总结

## 修改概述

根据论文推荐的方法，修改了分位数回归器（Quantile Regressor）的训练部分，使其使用得分`s(x)`而不是直接使用Loss。

---

## 主要改动

### 1. **分位数回归器训练方法改进** (`train_scheduler_black_score.py`)

#### 原方法（旧实现）
- 直接使用**Loss值**作为训练目标
- 分位数回归器预测Loss的分布 N(μ_loss, σ_loss²)

#### 新方法（改进实现）
- 使用**得分 s(x)**作为训练目标
- **得分计算公式**：
  $$s(x) = \text{logit}(\text{correct\_class}) - \sum_{\text{other}} \text{logit}(\text{other\_class})$$

- **工作流程**：
  1. 获取Test数据集中的样本
  2. 输入DP GNN模型，获取logits（预测值）
  3. 对每个样本计算得分 s(x)
  4. 训练MLP预测得分的分布参数 (μ, σ)
  5. 使用NLL Loss进行优化

#### 关键代码变更
```python
# 旧方法：直接计算Loss
losses = self.criterion(out, batch.y)
all_losses.append(losses.cpu())

# 新方法：计算得分s(x)
logits = self.model(x)  # [batch_size, num_classes]
correct_logits = logits[torch.arange(logits.size(0)), y]
logits_masked = logits.clone()
logits_masked[torch.arange(logits.size(0)), y] = 0
other_logits_sum = logits_masked.sum(dim=1)
scores = correct_logits - other_logits_sum
all_scores.append(scores.cpu())
```

---

### 2. **CDF得分计算器实现** (`score.py`)

新增`CDFScoreCalculator`类，用于计算CDF-based审计得分。

#### 功能说明
- **输入**：样本特征和实际值（Loss或Score）
- **输出**：CDF得分 [0, 1]
- **原理**：
  1. 分位数回归器预测分布 N(μ, σ)
  2. 计算标准化值：z = (value - μ) / σ
  3. 计算CDF：Φ(z)（高斯分布的累积分布函数）

#### CDF得分解释
- **CDF ≈ 0**：值远小于预期 → 强烈成员信号（低Loss）
- **CDF ≈ 0.5**：值符合预期 → 非成员信号（正常Loss）
- **CDF ≈ 1**：值远大于预期 → 异常值（高Loss）

#### 代码实现
```python
class CDFScoreCalculator:
    def compute_scores(self, features, values):
        """计算CDF得分"""
        mu, sigma = self.model(features)  # 预测分布参数
        z = (values - mu) / (sigma + 1e-8)  # 标准化
        cdf_scores = norm.cdf(z.cpu().numpy())  # 计算CDF
        return cdf_scores
```

---

## 训练流程

### Test数据集处理
```
Test Set
  ↓
[逐batch处理]
  ├─ 获取特征 x
  ├─ 获取标签 y
  ├─ 前向传播获取logits
  ├─ 计算得分 s(x) = logit(correct) - sum(logit(others))
  └─ 收集 (x, s(x)) 对
  ↓
[将收集的数据分为8:2训练验证集]
  ↓
[训练MLP分位数回归器]
  ├─ 输入：特征 x
  ├─ 输出：预测均值 μ(x)、标准差 σ(x)
  ├─ 目标：学习得分 s(x) 的分布
  ├─ 损失函数：NLL = 0.5*log(2πσ²) + (s(x)-μ)²/(2σ²)
  └─ 训练50个epoch
  ↓
[保存训练完成的分位数回归器]
```

---

## 训练参数

### MLP架构
| 参数 | 值 |
|------|-----|
| 输入维度 | 特征维度（如Amazon=500） |
| 隐藏层维度 | 256 |
| 隐藏层数 | 2 |
| 激活函数 | ReLU |
| Dropout | 0.5 |
| σ激活函数 | Softplus |

### 优化器配置
| 参数 | 值 |
|------|-----|
| 优化器 | Adam |
| 学习率 | 1e-3 |
| 权重衰减 | 1e-5 |
| 梯度裁剪 | max_norm=1.0 |

### 数据划分
- 训练集：80%的Test数据
- 验证集：20%的Test数据
- 总Epoch：50

---

## 黑盒审计流程

### 审计步骤
1. **提取金丝雀损失值**：从最后 num_canaries 个样本获取损失
2. **计算CDF得分**：使用分位数回归器预测分布，计算CDF
3. **双重循环攻击**：
   - 猜测前 k+ 个最可能的成员（CDF最小）
   - 猜测前 k- 个最可能的非成员（CDF最大或接近0.5）
4. **计算隐私下界**：基于Clopper-Pearson二项式测试

### 输出指标
- **Epsilon下界**：实证隐私泄露量
- **MIA准确率**：成员推断攻击成功率
- **最佳配置**：(k+, k-)值

---

## 优势对比

| 方面 | 旧方法（Loss） | 新方法（Score s(x)） |
|------|----------------|-------------------|
| **训练信号** | Loss值（容易受DP噪声影响） | 得分s(x)（更稳健） |
| **分布特性** | Loss分布可能多峰 | 得分分布更接近单峰高斯 |
| **审计准确性** | 受噪声干扰较大 | 审计信号更清晰 |
| **论文一致性** | ✗ | ✓（完全遵循论文推荐） |

---

## 使用建议

1. **确保模型输出**：模型必须能输出raw logits（未softmax）
2. **批量大小**：建议≥32以获得稳定的分布估计
3. **Test集大小**：越大越好，建议≥1000个样本
4. **金丝雀数量**：通常使用Test集后1000-3000个样本

---

## 文件修改清单

| 文件 | 修改 | 行数 |
|------|------|------|
| `train_scheduler_black_score.py` | 修改 `_train_quantile_regressor()` 方法 | 445-546 |
| `score.py` | 新增 `CDFScoreCalculator` 类 | 18-75 |

---

## 验证方式

运行以下命令验证修改：
```bash
python train_scheduler_black_score.py \
  --enable_audit \
  --num_canaries 1000 \
  --priv_epsilon 1.0
```

预期输出：
- ✓ Quantile Regressor 在Test集上训练完成
- ✓ 输出得分 s(x) 的统计信息（范围、均值、标准差）
- ✓ CDF-based审计结果（epsilon下界、MIA准确率）

---

## 参考资源

- **论文**: Logit Difference作为得分度量的优势
- **Loss**: NLL = 0.5*log(2πσ²) + (target-μ)²/(2σ²)
- **CDF**: Φ(z) = P(Z ≤ z)，其中Z~N(0,1)
