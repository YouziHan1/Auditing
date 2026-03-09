"""
分位数回归器 (Quantile Regressor) - 基于论文方法
用于预测节点Loss分布的均值 μ 和标准差 σ

架构：
- 输入：节点特征 (feat_dim)
- 隐藏层：2-3层，隐藏单元数 128-256
- 输出：两个头，分别输出 μ 和 σ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm


class CDFScoreCalculator:
    """
    CDF得分计算器 - 使用分位数回归器计算CDF-based得分
    
    原理：
    1. 分位数回归器预测给定特征的Loss/Score分布 N(μ, σ²)
    2. 计算真实值在该分布下的CDF: Φ((value - μ) / σ)
    3. CDF接近0或1表示异常值，CDF接近0.5表示正常值
    
    Args:
        quantile_regressor: 训练好的QuantileRegressor模型
    """
    
    def __init__(self, quantile_regressor, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = quantile_regressor.to(device)
        self.device = device
        self.model.eval()
    
    def compute_scores(self, features, values):
        """
        计算CDF得分
        
        Args:
            features: 输入特征 [N, feat_dim]，numpy array或torch tensor
            values: 对应的真实值（Loss或Score）[N]，numpy array或torch tensor
        
        Returns:
            cdf_scores: CDF得分 [N]，值在[0, 1]之间
                - 接近0：值远小于预期（成员信号强）
                - 接近0.5：值符合预期（正常/非成员信号）
                - 接近1：值远大于预期（异常）
        """
        # 转换为tensor
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values).float()
        
        features = features.to(self.device)
        values = values.to(self.device)
        
        # 预测分布参数
        with torch.no_grad():
            mu, sigma = self.model(features)  # [N, 1], [N, 1]
            mu = mu.squeeze(-1)  # [N]
            sigma = sigma.squeeze(-1)  # [N]
        
        # 计算标准化值: z = (value - μ) / σ
        z = (values - mu) / (sigma + 1e-8)
        
        # 计算CDF: Φ(z) 使用高斯分布累积分布函数
        # scipy的norm.cdf计算P(Z <= z)，其中Z~N(0,1)
        z_np = z.cpu().numpy()
        cdf_scores = norm.cdf(z_np)  # [N]
        
        return cdf_scores


class QuantileRegressor(nn.Module):
    """
    分位数回归器：预测Loss分布参数
    
    Args:
        feat_dim (int): 输入特征维度
        hidden_dim (int): 隐藏层维度，默认256
        num_layers (int): 隐藏层层数，默认2 (可选2-3)
        dropout_p (float): Dropout概率，默认0.5
        activation (str): 激活函数，'relu' 或 'gelu'，默认'relu'
        sigma_activation (str): σ的激活函数，'softplus' 或 'exp'，默认'softplus'
        sigma_eps (float): σ的极小值防止数值不稳定，默认1e-6
    """
    
    def __init__(
        self,
        feat_dim,
        hidden_dim=256,
        num_layers=2,
        dropout_p=0.5,
        activation='relu',
        sigma_activation='softplus',
        sigma_eps=1e-6
    ):
        super(QuantileRegressor, self).__init__()
        
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.sigma_eps = sigma_eps
        
        # 激活函数选择
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # σ激活函数选择
        if sigma_activation.lower() == 'softplus':
            self.sigma_activation = nn.Softplus()
        elif sigma_activation.lower() == 'exp':
            self.sigma_activation = lambda x: torch.exp(x) + sigma_eps
        else:
            raise ValueError(f"Unknown sigma activation: {sigma_activation}")
        
        # ========== 构建网络 ==========
        layers = []
        
        # 第一层：特征 -> 隐藏
        layers.append(nn.Linear(feat_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(self.activation)
        layers.append(nn.Dropout(dropout_p))
        
        # 中间隐藏层 (如果 num_layers > 2)
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_p))
        
        self.backbone = nn.Sequential(*layers)
        
        # ========== 两个输出头 ==========
        # 头1：预测 μ (均值)
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 头2：预测 σ (标准差)
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征张量，形状 [batch_size, feat_dim]
        
        Returns:
            mu: 预测的Loss均值，形状 [batch_size, 1]
            sigma: 预测的Loss标准差，形状 [batch_size, 1]
        """
        # 通过backbone提取特征
        h = self.backbone(x)  # [batch_size, hidden_dim]
        
        # 两个头分别预测μ和σ
        mu = self.mu_head(h)  # [batch_size, 1]
        sigma_raw = self.sigma_head(h)  # [batch_size, 1]
        
        # 对σ应用激活函数确保其为正
        sigma = self.sigma_activation(sigma_raw) + self.sigma_eps
        
        return mu, sigma
    
    def predict_loss_distribution(self, x):
        """
        预测Loss分布（返回均值、标准差和采样值）
        
        Args:
            x: 输入特征张量
        
        Returns:
            dict: 包含 'mu', 'sigma', 'samples' 的字典
        """
        with torch.no_grad():
            mu, sigma = self.forward(x)
            
            # 从高斯分布采样（可选）
            dist = torch.distributions.Normal(mu, sigma)
            samples = dist.sample()  # [batch_size, 1]
            
            return {
                'mu': mu.squeeze(-1),
                'sigma': sigma.squeeze(-1),
                'samples': samples.squeeze(-1),
                'distribution': dist
            }
    
    def compute_nll_loss(self, predictions, targets):
        """
        计算负对数似然 (NLL) 损失
        
        假设预测的Loss遵循高斯分布 N(μ, σ²)
        NLL = -log p(y|x) = 0.5 * log(2πσ²) + (y-μ)²/(2σ²)
        
        Args:
            predictions: tuple (mu, sigma)
            targets: 真实Loss值，形状 [batch_size]
        
        Returns:
            loss: 标量损失值
        """
        mu, sigma = predictions
        
        # 重塑以匹配目标
        mu = mu.squeeze(-1)
        sigma = sigma.squeeze(-1)
        
        # 高斯分布的NLL
        variance = sigma ** 2
        loss = 0.5 * torch.log(2 * np.pi * variance) + (targets - mu) ** 2 / (2 * variance)
        
        return loss.mean()


class QuantileRegressorTrainer:
    """
    分位数回归器的训练器
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.loss_history = []
    
    def setup_optimizer(self, lr=1e-3, weight_decay=1e-5):
        """设置优化器"""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device).float()
            
            # 前向传播
            mu, sigma = self.model(x)
            
            # 计算损失
            loss = self.model.compute_nll_loss((mu, sigma), y)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        self.loss_history.append(avg_loss)
        return avg_loss
    
    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets_list = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device).float()
                
                mu, sigma = self.model(x)
                loss = self.model.compute_nll_loss((mu, sigma), y)
                
                total_loss += loss.item()
                predictions.append(mu.cpu())
                targets_list.append(y.cpu())
        
        avg_loss = total_loss / len(test_loader)
        predictions = torch.cat(predictions)
        targets = torch.cat(targets_list)
        
        # 计算R²分数
        ss_res = torch.sum((targets - predictions) ** 2)
        ss_tot = torch.sum((targets - targets.mean()) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        
        return avg_loss, r2_score.item()


# ========== 使用示例 ==========
if __name__ == '__main__':
    print("="*70)
    print("分位数回归器 (Quantile Regressor) 示例")
    print("="*70)
    
    # 参数
    feat_dim = 500  # Amazon数据集特征维度示例
    batch_size = 32
    num_samples = 1000
    
    # 生成模拟数据
    print(f"\n1. 生成模拟数据...")
    X = torch.randn(num_samples, feat_dim)
    # 模拟Loss：与某些特征相关
    y_true = torch.abs(X[:, :10].sum(dim=1)) * 0.1 + 0.05 + 0.02 * torch.randn(num_samples)
    y_true = torch.clamp(y_true, min=0)  # Loss应该非负
    
    print(f"   特征维度: {feat_dim}")
    print(f"   样本数: {num_samples}")
    print(f"   Loss范围: [{y_true.min():.4f}, {y_true.max():.4f}]")
    print(f"   Loss均值: {y_true.mean():.4f}")
    
    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(X, y_true)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    print(f"\n2. 创建模型...")
    model = QuantileRegressor(
        feat_dim=feat_dim,
        hidden_dim=256,
        num_layers=2,
        dropout_p=0.5,
        activation='relu',
        sigma_activation='softplus'
    )
    print(f"   模型架构:")
    print(f"   - 输入层: {feat_dim}")
    print(f"   - 隐藏层: 256 x 2")
    print(f"   - 输出: μ (均值) + σ (标准差)")
    print(f"   总参数数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    print(f"\n3. 训练模型...")
    trainer = QuantileRegressorTrainer(model)
    trainer.setup_optimizer(lr=1e-3, weight_decay=1e-5)
    
    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        test_loss, r2 = trainer.evaluate(test_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Test Loss: {test_loss:.6f} | "
                  f"R²: {r2:.4f}")
    
    # 评估
    print(f"\n4. 最终评估...")
    model.eval()
    with torch.no_grad():
        sample_x = X[:5]
        mu, sigma = model(sample_x)
        pred_dist = model.predict_loss_distribution(sample_x)
        
        print(f"   样本预测 (前5个):")
        for i in range(5):
            print(f"   {i+1}. μ={mu[i].item():.4f}, σ={sigma[i].item():.4f}, "
                  f"样本Loss={pred_dist['samples'][i].item():.4f}")
    
    print(f"\n✓ 模型训练完成!")
    print("="*70)
