#本文方案 加载数据、构建模型、启动DP-SGD训练 + CDF黑盒审计
import torch
import time
import datasets.SETUP as SETUP
import datasets.utils as dms_utils
import datasets.model as dms_model
import numpy as np
import utils
import train_scheduler_black_score as tsch

if __name__ == '__main__':
    s_time = time.time()
    num_canaries_uni = 3000
    num_repeats = 3
    
    args = utils.get_args()
    SETUP.setup_seed(args.seed)
    device = SETUP.get_device()

    train_loader, val_loader, test_loader, dataset, x = dms_utils.form_loaders(args)
    args.num_classes = dataset.num_classes

    model = dms_model.G_net(K=args.K, feat_dim=x.shape[1], num_classes=args.num_classes, hidden_channels=128)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    canary_mask = np.zeros(num_canaries_uni, dtype=bool)
    canary_mask[:num_canaries_uni//2] = True
    
    train_master = tsch.trainer(
        model=model,
        optimizer=optimizer,
        loaders=[train_loader, None, test_loader],
        device=device,
        criterion=dms_model.criterion,
        args=args,
        enable_audit=True,
        num_canaries=num_canaries_uni,
        canary_mask=canary_mask,
        dataset=dataset,
    )
    
    train_master.run()
    
    print(f'\n==> Total Time: {time.time() - s_time:.2f}s')

