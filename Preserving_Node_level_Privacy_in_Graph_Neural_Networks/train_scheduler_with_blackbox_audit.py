#DP-SGD 实现 + 黑盒损失值审计
#包含 DP 相关的梯度裁剪、噪声注入、隐私预算统计
#管理 train/val/test 三阶段 + 金丝雀黑盒审计
import enum
import torch
import time
from tqdm import tqdm
import random
import torchvision
import torchvision.transforms as T
from functorch import make_functional_with_buffers
from functorch import vmap, grad
from copy import deepcopy
import os
import numpy as np
import math
import logging
import json
from pathlib import Path

import scipy.stats
import math
import numpy as np

''' '''
import utils

from privacy import accounting_analysis as aa
from privacy import sampling as sampling
import datasets.SETUP as SETUP
from analyze_canary_loss import generate_loss_visualization
''' helpers '''
TRAIN_SETUP_LIST = ('epoch', 'device', 'optimizer', 'loss_metric', 'enable_per_grad')

class Phase(enum.Enum):
    TRAIN = enum.auto()
    VAL = enum.auto()
    TEST = enum.auto()
    PHASE_to_PHASE_str = { TRAIN: "Training", VAL: "Validation", TEST: "Testing"}


def p_value_DP_audit(m, r, v, eps, delta):
    """计算审计的 P-value"""
    if eps < 0: return 1.0 
    q = 1 / (1 + math.exp(-eps))
    beta = scipy.stats.binom.sf(v - 1, r, q)
    alpha = 0
    sum_prob = 0
    for i in range(1, v + 1):
        sum_prob = sum_prob + scipy.stats.binom.pmf(v - i, r, q)
        if sum_prob > i * alpha:
            alpha = sum_prob / i
    p = beta + alpha * delta * 2 * m
    return min(p, 1)

def get_eps_audit(m, r, v, delta, p_val_target):
    """二分查找 Epsilon 下界"""
    eps_min = 0
    eps_max = 20 # 搜索上限
    for _ in range(50):
        eps = (eps_min + eps_max) / 2
        if p_value_DP_audit(m, r, v, eps, delta) < p_val_target:
            eps_min = eps
        else:
            eps_max = eps
    return eps_min

def analyze_privacy_leakage_blackbox(losses, mask_in, delta=1e-5, confidence=0.1, write_log=None):
    """
    黑盒损失值审计：基于损失值进行 Clopper-Pearson 攻击
    
    假设：
    - 低损失值 → 可能是成员（IN）
    - 高损失值 → 可能是非成员（OUT）
    
    Args:
        losses: 金丝雀损失值 (numpy array，低值=成员信号)
        mask_in: 真实成员标记 (Boolean array)
        delta: 隐私参数
        confidence: 置信度 (默认0.05，即95%置信水平)
        write_log: 日志写入函数 (可选，用于同时保存到日志文件)
    """
    M = len(losses)  # 总数
    
    # 定义日志输出函数
    def log_output(msg):
        if write_log is not None:
            write_log(msg, verbose=True)
        else:
            print(msg)
    
    # 1. 排序索引
    # 升序索引：用于找损失值最低的 (Top-K+, 猜测为成员)
    indices_asc = np.argsort(losses)
    # 降序索引：用于找损失值最高的 (Top-K-, 猜测为非成员)
    indices_desc = np.argsort(losses)[::-1]
    
    # 2. 准备猜测范围
    step = max(10, M // 20) 
    range_k_plus = [0] + list(range(step, M, step))
    range_k_minus = [0] + list(range(step, M, step))
    
    log_output(f"\n>>> Running Black-Box Loss-Based Audit (Confidence: {1-confidence:.0%})")
    log_output(f"{'k+ (Low Loss)':<12} {'k- (High Loss)':<14} {'Total(r)':<10} {'Correct(v)':<12} {'Acc':<10} {'Eps Lower Bound'}")
    log_output("-" * 80)

    best_eps = 0.0
    best_config = (0, 0) # (k_plus, k_minus)
    best_acc = 0.0

    # 3. 双重循环遍历 (Grid Search)
    for k_p in range_k_plus:
        for k_m in range_k_minus:
            
            # 边界检查
            if k_p + k_m > M or (k_p == 0 and k_m == 0):
                continue
                
            # --- 统计正向猜测 (Guess IN based on low loss) ---
            if k_p > 0:
                # 取损失值最低的 k_p 个
                idx_p = indices_asc[:k_p]
                # 猜对了多少个 (真值为 True 的个数)
                v_p = np.sum(mask_in[idx_p])
            else:
                v_p = 0
                
            # --- 统计负向猜测 (Guess OUT based on high loss) ---
            if k_m > 0:
                # 取损失值最高的 k_m 个
                idx_m = indices_desc[:k_m]
                # 猜对了多少个 (真值为 False 的个数)
                v_m = np.sum(~mask_in[idx_m])
            else:
                v_m = 0
            
            # --- 汇总结果 ---
            r_total = k_p + k_m
            v_total = v_p + v_m
            
            acc = v_total / r_total
            
            # --- 计算审计下界 ---
            if acc > 0.5:
                eps_lb = get_eps_audit(M, r_total, v_total, delta, confidence)
            else:
                eps_lb = 0.0
            
            # 打印筛选
            if eps_lb > best_eps or (k_p % (step*5) == 0 and k_m % (step*5) == 0):
                 log_output(f"{k_p:<12} {k_m:<14} {r_total:<10} {v_total:<12} {acc:.4f}     {eps_lb:.4f}")

            if eps_lb > best_eps:
                best_eps = eps_lb
                best_config = (k_p, k_m)
                best_acc = acc

    log_output("-" * 80)
    log_output(f"FINAL BLACK-BOX AUDIT RESULT:")
    log_output(f"Max Epsilon Lower Bound: {best_eps:.4f}")
    log_output(f"Best Configuration: k+={best_config[0]} (Low Loss), k-={best_config[1]} (High Loss)")
    log_output(f"Attacker Accuracy at Best: {best_acc:.2%}")
    log_output("=" * 30)
    
    return best_eps, best_acc


# ==========================================
# 黑盒损失值审计 - BlackBoxAuditor
# ==========================================
class BlackBoxAuditor:
    """
    黑盒审计器：在训练完成后对金丝雀进行损失值计算和成员推断
    """
    def __init__(self, model, device, criterion):
        self.model = model
        self.device = device
        self.criterion = criterion
        # 创建一个用于逐样本计算的损失函数
        self.criterion_per_sample = torch.nn.CrossEntropyLoss(reduction='none')
        self.canary_losses = None
    
    def predict_and_extract_losses_from_dataset(self, dataset, num_canaries=3000):
        """
        从完整图数据集中提取最后 num_canaries 个样本的损失值
        
        Args:
            dataset: 图数据集
            num_canaries: 要审计的样本数（通常是最后1000个）
        
        Returns:
            losses: 金丝雀的损失值数组
        """
        self.model.eval()
        
        # 获取完整图数据
        graph_data = dataset[0]
        total_nodes = graph_data.num_nodes
        
        # 最后 num_canaries 个样本的索引
        canary_start = total_nodes - num_canaries
        canary_indices = list(range(canary_start, total_nodes))
        
        print(f"Extracting losses from nodes [{canary_start}, {total_nodes-1}]")
        
        losses = []
        with torch.no_grad():
            # 获取特征和标签
            x = graph_data.x[canary_indices].to(self.device)  # [num_canaries, feat_dim]
            y = graph_data.y[canary_indices].to(self.device)  # [num_canaries]
            
            # 前向传播
            outputs = self.model(x)  # [num_canaries, num_classes] 或其他形状
            
            # 处理输出维度
            if outputs.dim() > 2:
                if outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)
                else:
                    outputs = outputs.reshape(outputs.size(0), -1)
            
            # 计算逐样本损失
            batch_losses = self.criterion_per_sample(outputs, y.view(-1))
            losses = batch_losses.cpu().numpy()
        
        self.canary_losses = np.array(losses)
        print(f"✓ 提取了 {len(losses)} 个金丝雀的损失值")
        print(f"  - 损失值范围: [{self.canary_losses.min():.4f}, {self.canary_losses.max():.4f}]")
        print(f"  - 平均损失值: {self.canary_losses.mean():.4f}")
        
        return self.canary_losses
        
    def predict_and_extract_losses(self, data_loader):
        """
        在金丝雀数据上执行前向传播并提取损失值
        
        Args:
            data_loader: 包含金丝雀的数据加载器 (返回 x, y 或 x, y, sample_ids)
        
        Returns:
            losses: 金丝雀的损失值数组
        """
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, desc="Extracting Losses")):
                # 处理 2-tuple 或 3-tuple
                if len(batch) == 3:
                    x, targets, _ = batch
                else:
                    x, targets = batch
                
                x, targets = x.to(self.device), targets.to(self.device)
                
                # 前向传播
                outputs = self.model(x)
                
                # 确保 targets 是正确的形状（标量标签）
                # 强制转换为 long (int64) 类型
                targets = targets.view(-1).long()   
                
                # 处理输出维度：需要确保是 [batch, num_classes] 形状
                # 可能的形状：[batch, 1, num_classes] -> squeeze -> [batch, num_classes]
                #           或 [batch, num_classes] -> 保持不变
                #           或其他情况
                if outputs.dim() > 2:
                    # 如果维度大于2，尝试squeeze中间维度
                    if outputs.size(1) == 1:
                        outputs = outputs.squeeze(1)
                    else:
                        # 其他情况，打印警告
                        print(f"[Warning] Unexpected outputs shape: {outputs.shape}, targets shape: {targets.shape}")
                        outputs = outputs.reshape(outputs.size(0), -1)  # 展平到2D
                
                # 调试：仅在第一个batch打印形状
                if i == 0:
                    print(f"  >> outputs shape: {outputs.shape}, targets shape: {targets.shape}")
                
                # 计算损失（逐样本）- 使用 reduction='none'
                try:
                    batch_losses = self.criterion_per_sample(outputs, targets)
                    losses.extend(batch_losses.cpu().numpy())
                except Exception as e:
                    print(f"[Error] Loss calculation failed: {e}")
                    print(f"  outputs shape: {outputs.shape}, targets shape: {targets.shape}")
                    raise
        
        self.canary_losses = np.array(losses)
        print(f"✓ 提取了 {len(losses)} 个金丝雀的损失值")
        print(f"  - 损失值范围: [{self.canary_losses.min():.4f}, {self.canary_losses.max():.4f}]")
        print(f"  - 平均损失值: {self.canary_losses.mean():.4f}")
        print(f"  - 中位数损失值: {np.median(self.canary_losses):.4f}")
        print(f"  - 标准差: {self.canary_losses.std():.4f}")
        
        return self.canary_losses


class trainer:
    def __init__(self, *, model, optimizer, loaders, device, criterion, args, enable_audit=False, num_canaries=0, canary_mask=None, dataset=None):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = loaders[0]
        self.val_loader = loaders[1]
        self.test_loader = loaders[2]
        self.device = device
        self.criterion = criterion
        self.args = args
        self.canary_mask = canary_mask
        self.dataset = dataset  # 保存数据集用于黑盒审计
        
        # 黑盒审计相关
        self.enable_audit = enable_audit
        self.num_canaries = num_canaries
        self.blackbox_auditor = None
        if enable_audit and num_canaries > 0:
            self.blackbox_auditor = BlackBoxAuditor(model, device, criterion)
            print(f"\n✓ 已启用黑盒审计，金丝雀数量: {num_canaries}\n")

        self.worker_model_func, self.worker_param_func, self.worker_buffers_func = make_functional_with_buffers(deepcopy(model), disable_autograd_tracking=True)
        self.record_data_type = 'weighted_recall'

        q = args.q = self.args.expected_batchsize / len(self.train_loader.dataset)

        
        print(f'{"="*40}\n')
        #隐私预算与噪声计算
        self.args.delta = 1/len(self.train_loader.dataset)**1.1
        self.std = self.args.std = aa.get_std_node_dp(
                                        q = q,
                                        EPOCH = int(self.args.epoch),
                                        D_out = len(self.train_loader.dataset),
                                        M_train = self.args.num_neighbors,
                                        epsilon = self.args.priv_epsilon,
                                        delta = self.args.delta,
                                        saving_path = Path( SETUP.get_dataset_data_path() ) / 'privacy_node_dp'
                                    )
        # exit() 

        self.dataset_size = len(self.train_loader.dataset)

        ''' prepare for logging '''
        self.is_info_initialized = False
        self.init_log(dir = args.log_dir)
        para_info, self.args.num_params = utils.show_param(model)
        self.write_log( para_info, verbose=False) 
        arg_info = self._arg_readable(self.args)
        self.write_log(f'{arg_info}')
        self.write_log( f'dataset: {self.train_loader.dataset.graph_data_name}' )

        ''' '''
        self.data_logger = utils.data_recorder(root = self.args.log_dir)
        self.data_logger.record_data(f'weighted_recall.csv', arg_info)
        self.data_logger.record_data(f'weighted_recall.csv', para_info)
        self.data_logger.record_data(f'weighted_recall.csv', self.train_loader.dataset.graph_data_name)
        self.data_logger.record_data(f'weighted_recall.csv', self.train_loader.dataset.graph_data)

        self.write_log(self.train_loader.dataset.graph_data_name)
        self.write_log(self.train_loader.dataset.graph_data)

        ''' json data recorder template '''
        self.json_recorder = json_data_recorder(f'jd_{self.train_loader.dataset.graph_data_name}_{self.args.graph_setting}_eps{args.priv_epsilon}.json')
        self.json_recorder.add_record('args', vars(self.args))
        self.json_recorder.add_record('num_params', self.args.num_params)
        self.json_recorder.add_record('dataset', self.train_loader.dataset.graph_data_name)
        self.json_recorder.add_record('train_acc', None)
        self.json_recorder.add_record('val_acc', None)
        self.json_recorder.add_record('test_acc', None)
        self.json_recorder.add_record('test_pre', None)
        
        # 审计结果记录
        if enable_audit:
            self.json_recorder.add_record('audit_info', {
                'audit_type': 'black_box',
                'num_canaries': num_canaries,
                'enable_audit': True
            })

    def _arg_readable(self, arg):
        arg_dict = vars(self.args)
        arg_info = ['args:'] + [f'{key} -- {value}' for key, value in arg_dict.items()]
        arg_info = f'\n{"="*40}\n' + '\n'.join(arg_info) + f'\n{"="*40}\n'
        return arg_info

    def init_log(self, dir = 'logs'):
        self.is_info_initialized = True
        file_name = '/'.join(__file__.split('/')[:-1]) + f'/{dir}/log.txt'
        # print('==> log file name is at: ', file_name)
        logging.basicConfig(
            filename = file_name, 
            filemode = 'a',
            datefmt = "%H:%M:%S", 
            level = logging.INFO,
            format = '%(asctime)s[%(levelname)s] ~ %(message)s'
        )
        self.write_log('\n\n' + "VV" * 40 + '\n' + " " * 37 + 'NEW LOG\n' + "^^" * 40)  
    
    def write_log(self, info = None, c_tag = None, verbose = True):
        if not self.is_info_initialized:
            self.init_log()
            self.is_info_initialized = True
        if verbose:
            print(str(info))
        if c_tag is not None:
            logging.info(str(c_tag) + ' ' + str(info))
        else:
            logging.info(str(info))

    def run(self):
        start_time = time.time()

        for epoch in range(self.args.epoch):
            self.write_log(f'\nEpoch: [{epoch}] '.ljust(11) + '#' * 35)
            ''' lr rate scheduler '''
            self.epoch = epoch

            train_metrics, val_metrics, test_metrics = None, None, None
            ''' training '''
            if self.train_loader is not None:
                train_metrics = self.one_epoch(train_or_val = Phase.TRAIN, loader = self.train_loader)
                self.json_recorder.add_record('train_acc', float(train_metrics.__getattr__(self.record_data_type)))

            ''' validation '''
            if self.val_loader is not None:
                val_metrics = self.one_epoch(train_or_val = Phase.VAL, loader = self.val_loader)
                self.json_recorder.add_record('val_acc', float(val_metrics.__getattr__(self.record_data_type)))

            ''' testing '''
            if self.test_loader is not None:
                test_metrics = self.one_epoch(train_or_val = Phase.TEST, loader = self.test_loader)
                self.json_recorder.add_record('test_acc', float(test_metrics.__getattr__(self.record_data_type)))
                self.json_recorder.add_record('test_pre', float(test_metrics.__getattr__('weighted_precis')))

            '''logging data '''
            data_str = (' '*3).join([
                                f'{epoch}',
                                f'{ float( train_metrics.__getattr__(self.record_data_type) ) * 100:.2f}%'.rjust(7)
                                if train_metrics else 'NAN',

                                f'{ float( val_metrics.__getattr__(self.record_data_type) ) * 100:.2f}%'.rjust(7)
                                if val_metrics else 'NAN',

                                f'{ float( test_metrics.__getattr__(self.record_data_type) ) * 100:.2f}%'.rjust(7)
                                if test_metrics else 'NAN',
                                ])
            
            self.data_logger.record_data(f'{self.record_data_type}.csv', data_str)
      
        ''' ending '''
        self.write_log(f'\n\n=> TIME for ALL: {time.time() - start_time:.2f}  secs')
        
        # 保存黑盒审计结果
        if self.enable_audit and self.blackbox_auditor:
            self.write_log(f'\n=== 黑盒审计结果分析 ===')
            
            # 提取损失值
            try:
                # 直接从数据集的最后 num_canaries 个样本进行审计
                canary_losses = self.blackbox_auditor.predict_and_extract_losses_from_dataset(
                    self.dataset, 
                    num_canaries=self.num_canaries
                )
                # 计算IN/OUT平均损失和分布统计
                import numpy as np
                
                mid_point = self.num_canaries // 2
                loss_in = canary_losses[:mid_point]
                loss_out = canary_losses[mid_point:]
                
                # IN样本统计
                avg_loss_in = loss_in.mean()
                median_loss_in = np.median(loss_in)
                std_loss_in = loss_in.std()
                min_loss_in = loss_in.min()
                max_loss_in = loss_in.max()
                q25_in = np.percentile(loss_in, 25)
                q75_in = np.percentile(loss_in, 75)
                q90_in = np.percentile(loss_in, 90)
                q95_in = np.percentile(loss_in, 95)
                
                # OUT样本统计
                avg_loss_out = loss_out.mean()
                median_loss_out = np.median(loss_out)
                std_loss_out = loss_out.std()
                min_loss_out = loss_out.min()
                max_loss_out = loss_out.max()
                q25_out = np.percentile(loss_out, 25)
                q75_out = np.percentile(loss_out, 75)
                q90_out = np.percentile(loss_out, 90)
                q95_out = np.percentile(loss_out, 95)
                
                # 计算Loss分布的离散度指标
                # 低Loss占比 (Loss < 中位数)
                low_loss_in_ratio = (loss_in <= median_loss_in).sum() / len(loss_in) * 100
                low_loss_out_ratio = (loss_out <= median_loss_out).sum() / len(loss_out) * 100
                # 高Loss占比 (Loss > Q95)
                high_loss_in_ratio = (loss_in > q95_in).sum() / len(loss_in) * 100
                high_loss_out_ratio = (loss_out > q95_out).sum() / len(loss_out) * 100
                
                # 输出详细统计信息
                self.write_log('='*60)
                self.write_log('IN样本 (前一半金丝雀) Loss分布:')
                self.write_log(f'  平均值: {avg_loss_in:.4f}, 中位数: {median_loss_in:.4f}, 标准差: {std_loss_in:.4f}')
                self.write_log(f'  范围: [{min_loss_in:.4f}, {max_loss_in:.4f}]')
                self.write_log(f'  分位数 Q25: {q25_in:.4f}, Q50: {median_loss_in:.4f}, Q75: {q75_in:.4f}')
                self.write_log(f'  分位数 Q90: {q90_in:.4f}, Q95: {q95_in:.4f}')
                self.write_log(f'  低Loss占比(≤中位数): {low_loss_in_ratio:.1f}%, 高Loss占比(>Q95): {high_loss_in_ratio:.1f}%')
                self.write_log('='*60)
                self.write_log('OUT样本 (后一半金丝雀) Loss分布:')
                self.write_log(f'  平均值: {avg_loss_out:.4f}, 中位数: {median_loss_out:.4f}, 标准差: {std_loss_out:.4f}')
                self.write_log(f'  范围: [{min_loss_out:.4f}, {max_loss_out:.4f}]')
                self.write_log(f'  分位数 Q25: {q25_out:.4f}, Q50: {median_loss_out:.4f}, Q75: {q75_out:.4f}')
                self.write_log(f'  分位数 Q90: {q90_out:.4f}, Q95: {q95_out:.4f}')
                self.write_log(f'  低Loss占比(≤中位数): {low_loss_out_ratio:.1f}%, 高Loss占比(>Q95): {high_loss_out_ratio:.1f}%')
                self.write_log('='*60)
                self.write_log('分离度分析:')
                self.write_log(f'  IN vs OUT平均Loss差异: {(avg_loss_out - avg_loss_in):.4f} ({(avg_loss_out/avg_loss_in - 1)*100:.1f}%)')
                self.write_log(f'  IN vs OUT中位数差异: {(median_loss_out - median_loss_in):.4f}')
                self.write_log('='*60)
                

                # 记录原始损失值
                self.json_recorder.add_record('canary_losses', canary_losses.tolist())
                self.json_recorder.add_record('canary_mask', self.canary_mask.tolist())
                
                # 调用黑盒审计分析
                emp_eps, mia_acc = analyze_privacy_leakage_blackbox(
                    canary_losses, 
                    self.canary_mask, 
                    delta=self.args.delta if hasattr(self.args, 'delta') else 1e-5,
                    write_log=self.write_log
                )
                
                # 记录分析指标
                self.write_log(f'Black-Box Empirical Epsilon: {emp_eps:.4f}')
                self.write_log(f'Black-Box MIA Accuracy: {mia_acc:.2%}')
                

                self.json_recorder.add_record('audit_metrics', {
                    'audit_type': 'black_box',
                    'empirical_epsilon': float(emp_eps),
                    'mia_accuracy': float(mia_acc),
                    'num_canaries': self.num_canaries,
                    'avg_loss_in': float(avg_loss_in),
                    'avg_loss_out': float(avg_loss_out)
                })
            except Exception as e:
                self.write_log(f"黑盒审计分析出错: {e}")
                import traceback
                traceback.print_exc()
        
        self._clear_loader()
        self.json_recorder.save()

    def _clear_loader(self):
        ''' when persistent worker enabled, this following allows the worker to be terminated '''
        if self.train_loader is not None:
            self.train_loader._iterator._shutdown_workers()
        if self.val_loader is not None:
            self.val_loader._iterator._shutdown_workers()
        if self.test_loader is not None:
            self.test_loader._iterator._shutdown_workers()

    def one_epoch(self, train_or_val, loader):
        metrics = utils.ClassificationMetrics(num_classes = self.args.num_classes)
        metrics.num_images = metrics.loss = 0 
        is_training = train_or_val is Phase.TRAIN
        self.model.train(is_training)

        ''' per example method '''
        def compute_loss(model_para, buffers, x, targets):
            predictions = self.worker_model_func(model_para, buffers, x)
            predictions, targets = predictions[:1], targets[:1]
            loss = self.criterion(predictions, targets.flatten())
            return loss
        
        def per_forward(model_para, buffers, x):
            predictions = self.worker_model_func(model_para, buffers, x)
            return predictions

        def manual_forward(x, targets):
            out = vmap(per_forward, in_dims=(None, None, 0) )(self.worker_param_func, self.worker_buffers_func, x)
            out = out[:,0,:]
            targets = targets[:,0]
            loss = self.criterion(out, targets.view(-1))
            return loss, out, targets

        s = time.time()
        if is_training: 
            print(f'==> have {len(loader)} iterations in this epoch')
            s_time = time.time()
            for i, batch in enumerate(loader):
                # 标准处理：非黑盒审计，不需要特殊逻辑
                if len(batch) == 3:
                    x, targets, sample_ids = batch
                else:
                    x, targets = batch
                    sample_ids = torch.arange(len(x))

                x, targets = x.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                
                # 逐样本梯度计算
                per_grad = vmap( grad(compute_loss), in_dims=(None, None, 0, 0) )(
                    self.worker_param_func, self.worker_buffers_func, x, targets
                )
                
                ''' 前向获取loss '''
                loss, out, targets = manual_forward(x, targets)
                metrics.batch_update(loss.cuda(), out.cuda(), targets.cuda())
                
                # 黑盒审计不需要梯度注入，直接执行标准的DP-SGD步骤
                self.other_routine(per_grad)

        else:
            for i, batch in enumerate(loader):
                if len(batch) == 3:
                    x, targets, _ = batch
                else:
                    x, targets = batch
                    
                x, targets = x.to(self.device), targets.to(self.device)
                loss, out, targets = manual_forward(x, targets)
                metrics.batch_update(loss.cuda(), out.cuda(), targets.cuda())

        metrics.loss /= metrics.num_images
        self.write_log(f'    {train_or_val}: {time.time()-s:.3f} S, {self.record_data_type} = {float(metrics.__getattr__(self.record_data_type))*100:.2f}%')

        return metrics

    def other_routine(self, per_grad):

        per_grad = self.clip_per_grad(per_grad)

        ''' forming gradients'''
        for p_model, p_per in zip(self.model.parameters(), per_grad):
            p_model.grad = torch.mean(p_per, dim=0)
            ''' add noise to gradients'''
            p_model.grad = p_model.grad + torch.randn_like(p_model.grad) * self.std * self.args.C / p_per.shape[0] 
            
            ''' clamp extreme gradients'''
            cth = self.std * self.args.C / p_per.shape[0] / 1e2
            p_model.grad = torch.clamp(p_model.grad, -cth, cth)

        self.model_update()
    
    def model_update(self):
        '''update parameters'''
        self.optimizer.step()  
        ''' copy parameters to worker '''
        for p_model, p_worker in zip(self.model.parameters(), self.worker_param_func):
            p_worker.copy_(p_model.data)
    
    def clip_per_grad(self, per_grad):
        per_grad = list(per_grad)
        per_grad_norm = 2 * self._compute_per_grad_norm(per_grad) + 1e-6 

        ''' clipping/normalizing '''
        multiplier = torch.clamp(self.args.C / per_grad_norm, max = 1)
        for index, p in enumerate(per_grad):
            per_grad[index] = p * self._make_broadcastable( multiplier, p ) 
        return per_grad

    def _compute_per_grad_norm(self, iterator, which_norm = 2):
        all_grad = torch.cat([p.reshape(p.shape[0], -1) for p in iterator], dim = 1)
        per_grad_norm = torch.norm(all_grad, dim = 1, p = which_norm)

        return per_grad_norm
    
    def _make_broadcastable(self, tensor_to_be_reshape, target_tensor):
        broadcasting_shape = (-1, *[1 for _ in target_tensor.shape[1:]])
        return tensor_to_be_reshape.reshape(broadcasting_shape)
    
class json_data_recorder:
    def __init__(self, filename):
        self.data_dict = {}
        self.dir_name = 'data_records'
        
        self._set_record_filename(filename)
        
    def _set_record_filename(self, filename='data_record.json'):
        self.record_name = filename
        dir_path = '/'.join(__file__.split('/')[:-1]) + f'/{self.dir_name}'
        os.mkdir(dir_path) if not os.path.exists(dir_path) else None
        self.file_path =  dir_path + f'/{filename}'

    def add_record(self, name, data):
        if name not in self.data_dict:
            self.data_dict[name] = []
        if data is not None:
            self.data_dict[name].append(data)

    def save(self):
        data = []
        
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                data = json.load(f)
        
        with open(self.file_path, 'w') as f:
            json.dump(data + [self.data_dict], f, indent=4) 

def get_data_from_record(filename):
    path =  '/'.join(__file__.split('/')[:-1]) + f'/data_records/{filename}'
    with open(path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    dr = json_data_recorder()
    dr.set_record_filename('sampling_noise.json')
    dr.add_record('a', 1)
    dr.add_record('a', 2)
    dr.save()
    data = get_data_from_record('sampling_noise.json')
    print(data)
