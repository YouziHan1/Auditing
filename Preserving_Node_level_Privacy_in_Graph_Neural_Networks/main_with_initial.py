#本文方案: 构造最坏情况初始参数 (θ₀) 通过非私有预训练
# 流程：
# 1. 使用Test集（辅助数据集）进行非私有预训练，构造θ₀
# 2. 在θ₀基础上进行DP-GNN训练
# 3. 进行黑盒审计分析

import torch
import time
import datasets.SETUP as SETUP

import datasets.utils as dms_utils
import datasets.model as dms_model
import numpy as np
import os
import utils
import train_scheduler_with_initial as tsch  # 支持非私有预训练的trainer版本
import Test.Canary_random_features
import Test.Canary_Compare
import Test.Canary_compare_3

if __name__ == '__main__':
    """
    最坏情况初始参数构造框架
    
    核心思想：
    - 辅助数据集（Auxiliary Dataset）：使用Test集进行非私有预训练
    - 目标数据集（Target Dataset）：使用Train集进行DP-GNN训练和审计
    
    这样构造的初始参数θ₀会使得：
    1. "正常"样本产生的梯度最小（因为在test集上预训练好了）
    2. Canary样本的梯度信号最突出（与分布不一致）
    3. 从而最大化信噪比，达到最坏情况检测精度
    """
    for epoch in [0]:  
        for num_canaries_uni, num_repeats in [(3000, 3)]:
            s_time = time.time()
            num_canaries = num_canaries_uni * num_repeats
            
            # ========== 第1步：创建带金丝雀节点的新图 ==========
            print("\n" + "="*70)
            print("第1步：创建金丝雀图")
            print("="*70)
            # Test.Canary_random_features.create_canary_graph(num_canaries_uni, num_repeats)
            # Test.Canary_Compare.create_canary_graph(num_canaries_uni, num_repeats)
            Test.Canary_compare_3.create_canary_graph(num_canaries_uni, num_repeats)
            # ========== 第2步：读取参数与环境 ==========
            print("\n" + "="*70)
            print("第2步：加载配置和数据")
            print("="*70)
            args = utils.get_args()
            SETUP.setup_seed(args.seed)
            device = SETUP.get_device()
            print(f"设备: {device}")
            print(f"种子: {args.seed}")

            # ========== 第3步：读取数据集，划分训练/验证/测试集 ==========
            print("\n" + "="*70)
            print("第3步：构建数据加载器")
            print("="*70)
            train_loader, val_loader, test_loader, dataset, x = dms_utils.form_loaders(args)
            args.num_classes = dataset.num_classes  # 类别数
            
            print(f"训练集大小: {len(train_loader.dataset)}")
            print(f"验证集大小: {len(val_loader.dataset) if val_loader else 'None'}")
            print(f"测试集大小: {len(test_loader.dataset)}")
            print(f"特征维度: {x.shape[1]}")
            print(f"类别数: {args.num_classes}")

            # ========== 第4步：构建模型 ==========
            print("\n" + "="*70)
            print("第4步：初始化模型")
            print("="*70)
            model = dms_model.G_net(K=args.K, feat_dim=x.shape[1], num_classes=args.num_classes, hidden_channels=128)
            model.to(device)
            print(f"模型构建完成")

            # ========== 第5步：配置优化器 ==========
            print("\n" + "="*70)
            print("第5步：配置优化器")
            print("="*70)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            print(f"优化器: Adam, 学习率: {args.lr}")

            # ========== 第6步：创建Canary掩码 ==========
            print("\n" + "="*70)
            print("第6步：设置Canary样本标记")
            print("="*70)
            
            # ⚠️  加载掩码（在create_canary_graph中生成，大小为3000）
            canary_mask_file = "./Test/canary_mask.npy"
            if os.path.exists(canary_mask_file):
                # 加载已生成的掩码
                canary_mask = np.load(canary_mask_file)
                print(f"✓ 已加载生成的mask: {canary_mask_file}")
                print(f"  掩码大小: {len(canary_mask)}")
            else:
                # 备选方案：如果mask不存在，创建一个新的掩码
                print(f"⚠️  未找到 {canary_mask_file}，创建新的掩码")
                # 只在原始金丝雀数上随机分配
                in_indices = np.random.choice(range(num_canaries_uni), size=num_canaries_uni // 2, replace=False)
                canary_mask = np.zeros(num_canaries_uni, dtype=bool)
                canary_mask[in_indices] = True
                # 保存
                np.save(canary_mask_file, canary_mask)
                print(f"✓ 已保存新生成的mask: {canary_mask_file} (大小={len(canary_mask)})")
            
            print(f"\nCanary掩码统计（用于审计）:")
            print(f"  掩码大小: {len(canary_mask)} (仅原始金丝雀数，不包括重复版本)")
            print(f"  IN（成员）样本: {canary_mask.sum()}")
            print(f"  OUT（非成员）样本: {(~canary_mask).sum()}")
            print(f"  示例IN索引: {np.where(canary_mask)[0][:10]}")
            print(f"  示例OUT索引: {np.where(~canary_mask)[0][:10]}")

            # ========== 第7步：创建Trainer + 非私有预训练 ==========
            print("\n" + "="*70)
            print("第7步：启动非私有预训练阶段（构造θ₀）")
            print("="*70)
            print(f"[信息] 使用Test集作为辅助数据集进行非私有预训练")
            print(f"[目标] 构造最坏情况初始参数，使得梯度信噪比最大化")
            
            # 准备非私有预训练的trainer
            train_master = tsch.trainer(
                model=model,
                optimizer=optimizer,
                loaders=[train_loader, val_loader, test_loader],
                device=device,
                criterion=dms_model.criterion,
                args=args,
                enable_audit=True,
                num_canaries=num_canaries_uni,
                canary_mask=canary_mask,
                dataset=dataset,
                pretrain_loader=test_loader,  # 使用Test集作为辅助数据集
                enable_pretrain=False,
                pretrain_epochs=getattr(args, 'pretrain_epochs', epoch)  # 默认预训练5个epoch
            )
            
            # ========== 第8步：执行完整训练流程 ==========
            print("\n" + "="*70)
            print("第8步：执行完整训练流程")
            print("="*70)
            print("[流程] 非私有预训练 -> DP-GNN训练 -> 黑盒审计")
            
            # 执行预训练 + DP-GNN训练 + 审计
            train_master.run()
            
            # ========== 第9步：统计和输出结果 ==========
            print("\n" + "="*70)
            print("第9步：训练完成，统计结果")
            print("="*70)
            total_time = time.time() - s_time
            print(f'\n==> 总耗时: {total_time:.4f}秒')
            print(f'==> 单次运行完成\n')

            # ========== 第10步：分析和对比 ==========
            print("\n" + "="*70)
            print("第10步：审计分析")
            print("="*70)
            print("""
    [预期结果分析]
    由于使用了辅助数据集（Test集）进行非私有预训练：

    1. 初始参数θ₀特性：
    - 在Test集上已经过优化，对Test分布很适配
    - 梯度特征与Train集差异最大

    2. DP-GNN训练阶段：
    - "正常"样本（Train集）的梯度：较小（因为起点已优化，仅需微调）
    - Canary样本的梯度：较大（分布外，差异明显）
    - 结果：信噪比最大化，审计精度最高

    3. 黑盒审计结果：
    - 应观察到较高的审计准确率
    - 较低的Epsilon下界（相对较强的隐私保证警告）
    
    [对比对象]
    - 标准初始参数（随机初始化）：信噪比较低
    - 本方法（最坏情况θ₀）：信噪比最高
            """)
            
            print("\n" + "="*70)
            print("流程完成！")
            print("="*70)
            print(f"\n详细日志已保存到: {args.log_dir}/log.txt")
            print(f"审计结果已保存到: data_records/")
