#本文方案 加载数据、构建模型、启动DP-SGD训练 + 白盒审计
import torch
import time
import datasets.SETUP as SETUP

import datasets.utils as dms_utils
import datasets.model as dms_model
import numpy as np
import utils
import train_scheduler_with_audit as tsch  # 改为审计版本
#import train_scheduler_with_white_pro as tsch

if __name__ == '__main__':
    for num in [1]:
        print(f"\n\n\n===== Running with {num} directions =====\n")
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
        canary_mask = np.zeros(3000, dtype=bool)
        canary_mask[:3000//2] = True
        # ========== DP-SGD 训练 + 白盒审计 ==========
        train_master = tsch.trainer(
                            model = model,
                            optimizer = optimizer,
                            loaders = [train_loader, None, test_loader],
                            device = device,
                            criterion = dms_model.criterion,
                            args = args,
                            enable_audit=True,           # ✓ 启用白盒审计
                            num_canaries=3000,            # ✓ 金丝雀数量
                            canary_mask=canary_mask,
                            num_directions=num,
                            #enable_two_stage=True
                        )
        
        train_master.run()
        print(f'\n==> ToTaL TiMe FoR OnE RuN: {time.time() - s_time:.4f}\n')
        
        # 评估Canary节点（可选）
        # canary_accuracy = evaluate_canary_nodes(model, dataset, device, num_canaries=100)
        # print(f'\n==> Canary Node Accuracy: {canary_accuracy:.2f}%\n\n\n')
