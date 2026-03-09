#本文方案 加载数据、构建模型、启动DP-SGD训练 + 黑盒审计
import torch
import time
import datasets.SETUP as SETUP

import datasets.utils as dms_utils
import datasets.model as dms_model
import numpy as np
import utils
import train_scheduler_with_blackbox_audit as tsch  # 黑盒审计版本
import Test.Canary_random_features_v2
import Test.Canary_random_features
import Test.Canary_features
import Test.find_features



if __name__ == '__main__':
    for num_canaries_uni, num_repeats in [(3000,3)]:
        s_time  = time.time()
        num_canaries = num_canaries_uni * num_repeats
        #创建带金丝雀节点的新图
        Test.find_features.create_canary_graph(num_canaries_uni, num_repeats)
        #Test.Canary_random_features_v2.create_canary_graph(num_canaries_uni, num_repeats)
        #Test.Canary_features.create_canary_graph(num_canaries_uni, num_repeats)
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
        canary_mask = np.zeros(num_canaries_uni, dtype=bool)
        canary_mask[:num_canaries_uni//2] = True
        # ========== DP-SGD 训练==========
        train_master = tsch.trainer(
                            model = model,
                            optimizer = optimizer,
                            loaders = [train_loader, None, test_loader],
                            device = device,
                            criterion = dms_model.criterion,
                            args = args,
                            enable_audit=True,           
                            num_canaries=num_canaries_uni,           
                            canary_mask=canary_mask,
                            dataset=dataset,  # 传递数据集用于黑盒审计
                        )
        
        train_master.run()
        print(f'\n==> ToTaL TiMe FoR OnE RuN: {time.time() - s_time:.4f}\n')
        
        # # 评估Canary节点
        # canary_accuracy = evaluate_canary_nodes(model, dataset, device, num_canaries=100)
        # print(f'\n==> Canary Node Accuracy: {canary_accuracy:.2f}%\n\n\n')
