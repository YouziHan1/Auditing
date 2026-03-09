#DP-SGD 实现 + 白盒梯度注入审计
#包含 DP 相关的梯度裁剪、噪声注入、隐私预算统计
#管理 train/val/test 三阶段 + 金丝雀审计
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

def analyze_privacy_leakage(scores, mask_in, delta=1e-5, confidence=0.05):
    """
    双边 Clopper-Pearson 审计：同时猜测最高分的 K+ 个为成员，最低分的 K- 个为非成员。
    
    Args:
        scores: 金丝雀得分 (numpy array)
        mask_in: 真实成员标记 (Boolean array)
        delta: 隐私参数
        confidence: 置信度 (默认0.05)
    """
    M = len(scores)  # 总数
    
    # 1. 排序索引
    # 降序索引：用于找分数最高的 (Top-K+)
    indices_desc = np.argsort(scores)[::-1]
    # 升序索引：用于找分数最低的 (Top-K-)
    indices_asc = np.argsort(scores) # 默认升序
    
    # 2. 准备猜测范围
    # 我们可以设置一个步长，比如总数的 2% 或 5%
    step = max(10, M // 20)
    # 【修改】猜测上限设置为金丝雀总数的一半
    max_guess = M // 2
    
    # 生成 K+ 和 K- 的候选列表（不超过上限）
    # 包含 0 是为了兼容"只猜正向"或"只猜负向"的情况
    range_k_plus = [0] + list(range(step, max_guess + 1, step))
    range_k_minus = [0] + list(range(step, max_guess + 1, step))
    
    print(f"\n>>> Running Dual-Side Top-K Audit (Confidence: {1-confidence:.0%})")
    print(f">>> Guess limit: {max_guess} (half of total {M} canaries)")
    print(f"{'k+ (High)':<10} {'k- (Low)':<10} {'Total(r)':<10} {'Correct(v)':<12} {'Acc':<10} {'Eps Lower Bound'}")
    print("-" * 75)

    best_eps = 0.0
    best_config = (0, 0) # (k_plus, k_minus)
    best_acc = 0.0

    # 3. 双重循环遍历 (Grid Search)
    # 注意：为了性能，如果数据量巨大，可以只扫描对角线或特定区域
    for k_p in range_k_plus:
        for k_m in range_k_minus:
            
            # [关键] 边界检查：猜测总数不能超过样本总数
            if k_p + k_m > M or (k_p == 0 and k_m == 0):
                continue
                
            # --- 统计正向猜测 (Guess IN) ---
            if k_p > 0:
                # 取分数最高的 k_p 个
                idx_p = indices_desc[:k_p]
                # 猜对了多少个 (真值为 True 的个数)
                v_p = np.sum(mask_in[idx_p])
            else:
                v_p = 0
                
            # --- 统计负向猜测 (Guess OUT) ---
            if k_m > 0:
                # 取分数最低的 k_m 个
                idx_m = indices_asc[:k_m]
                # 猜对了多少个 (真值为 False 的个数)
                # mask_in 为 False 代表是非成员，~mask_in 为 True
                v_m = np.sum(~mask_in[idx_m])
            else:
                v_m = 0
            
            # --- 汇总结果 ---
            # 总尝试次数 (Total Guesses)
            r_total = k_p + k_m
            # 总成功次数 (Correct Guesses: IN猜IN + OUT猜OUT)
            v_total = v_p + v_m
            
            acc = v_total / r_total
            
            # --- 计算审计下界 ---
            # 只有当准确率 > 50% 时才计算，节省时间
            if acc > 0.5:
                eps_lb = get_eps_audit(M, r_total, v_total, delta, confidence)
            else:
                eps_lb = 0.0
            
            # 打印筛选：只打印稍微有意义的结果，避免刷屏
            if eps_lb > best_eps or (k_p % (step*5) == 0 and k_m % (step*5) == 0):
                 print(f"{k_p:<10} {k_m:<10} {r_total:<10} {v_total:<12} {acc:.4f}     {eps_lb:.4f}")

            if eps_lb > best_eps:
                best_eps = eps_lb
                best_config = (k_p, k_m)
                best_acc = acc

    print("-" * 75)
    print(f"FINAL AUDIT RESULT:")
    print(f"Max Epsilon Lower Bound: {best_eps:.4f}")
    print(f"Best Configuration: k+={best_config[0]} (High Scores), k-={best_config[1]} (Low Scores)")
    print(f"Attacker Accuracy at Best: {best_acc:.2%}")
    print("=" * 30)
    
    return best_eps, best_acc
# ==========================================
# 白盒梯度注入审计 - OptimizedCanaryManager
# ==========================================
class OptimizedCanaryManager:
    """索引准备工作，将金丝雀的非零位置对应到模型参数上（多方向注入）"""
    def __init__(self, model, num_canaries, num_directions=5):
        self.model = model  # 【修复】保存模型引用用于后续访问参数
        self.num_canaries = num_canaries
        self.num_directions = num_directions  # 每个金丝雀的方向数
        # {layer_name: {canary_id: [local_param_idx_0, local_param_idx_1, ...]}}
        self.layer_canary_map = {}
        self.canary_id_to_locs = {}  # {canary_id: [(layer_name, local_idx), ...]}

        total_params = sum(p.numel() for p in model.parameters())
        print(f"> 模型参数总数: {total_params}")
        print(f"> 金丝雀数量: {num_canaries}")
        print(f"> 每个金丝雀的方向数: {num_directions}")
        
        # 随机选择参数位置植入金丝雀（总共 num_canaries * num_directions 个位置）
        total_positions = num_canaries * num_directions
        # 优先不重复，但如果位置不足则允许重复
        if total_positions <= total_params:
            all_indices = np.random.choice(total_params, total_positions, replace=False)
        else:
            all_indices = np.random.choice(total_params, total_positions, replace=True)
            print(f"  [Warning] 位置数 ({total_positions}) 超过总参数数 ({total_params})，使用重复抽样")
        
        # 按索引排序，并分配给每个金丝雀
        sorted_indices = sorted(all_indices)
        canary_directions = {}  # {canary_id: [idx_0, idx_1, ...]}
        for i, idx in enumerate(sorted_indices):
            cid = i // num_directions
            if cid not in canary_directions:
                canary_directions[cid] = []
            canary_directions[cid].append(idx)
        
        current_offset = 0
        
        for name, param in model.named_parameters():
            # 记录原始名字 (不带 _module.)
            clean_name = name.replace("_module.", "")
            
            num_el = param.numel()
            layer_end = current_offset + num_el
            
            if clean_name not in self.layer_canary_map:
                self.layer_canary_map[clean_name] = {}
            
            # 遍历所有金丝雀及其多个方向
            for cid in range(num_canaries):
                if cid not in canary_directions:
                    continue
                    
                local_indices = []
                for global_idx in canary_directions[cid]:
                    if current_offset <= global_idx < layer_end:
                        local_idx = global_idx - current_offset
                        local_indices.append(local_idx)
                
                if local_indices:
                    if cid not in self.layer_canary_map[clean_name]:
                        self.layer_canary_map[clean_name][cid] = []
                    self.layer_canary_map[clean_name][cid].extend(local_indices)
                    
                    # 记录到 canary_id_to_locs
                    if cid not in self.canary_id_to_locs:
                        self.canary_id_to_locs[cid] = []
                    for local_idx in local_indices:
                        self.canary_id_to_locs[cid].append((clean_name, local_idx))
            
            current_offset += num_el
        
        print(f"> 金丝雀分布在 {len(self.layer_canary_map)} 层")

    def init_from_sensitive_dims(self, sensitive_dims, num_canaries, num_directions=1):
        """
        【新增】基于敏感维度初始化金丝雀
        将敏感维度作为金丝雀的注入位置，确保金丝雀完全在敏感维度上
        
        Args:
            sensitive_dims: [(layer_name, local_idx), ...] 敏感维度列表
            num_canaries: 金丝雀数量
            num_directions: 每个金丝雀的方向数（通常为1）
        """
        self.num_canaries = num_canaries
        self.num_directions = num_directions
        self.layer_canary_map = {}  # 重置
        self.canary_id_to_locs = {}
        
        print(f"\n✓ 基于 {len(sensitive_dims)} 个敏感维度初始化金丝雀")
        print(f"> 金丝雀数量: {num_canaries}")
        print(f"> 每个金丝雀的方向数: {num_directions}")
        
        # 将敏感维度分配给金丝雀
        total_positions = num_canaries * num_directions
        
        if len(sensitive_dims) < total_positions:
            print(f"  [Warning] 敏感维度数 ({len(sensitive_dims)}) 不足 ({total_positions})，将重复使用")
            # 重复使用敏感维度
            selected_dims = [sensitive_dims[i % len(sensitive_dims)] for i in range(total_positions)]
        else:
            # 随机选择 total_positions 个敏感维度
            indices = np.random.choice(len(sensitive_dims), total_positions, replace=False)
            selected_dims = [sensitive_dims[i] for i in sorted(indices)]
        
        # 分配给金丝雀
        canary_id = 0
        direction_count = 0
        
        for layer_name, local_idx in selected_dims:
            if canary_id not in self.canary_id_to_locs:
                self.canary_id_to_locs[canary_id] = []
            
            self.canary_id_to_locs[canary_id].append((layer_name, local_idx))
            
            # 记录到 layer_canary_map
            if layer_name not in self.layer_canary_map:
                self.layer_canary_map[layer_name] = {}
            if canary_id not in self.layer_canary_map[layer_name]:
                self.layer_canary_map[layer_name][canary_id] = []
            self.layer_canary_map[layer_name][canary_id].append(local_idx)
            
            direction_count += 1
            if direction_count == num_directions:
                canary_id += 1
                direction_count = 0
        
        print(f"> 金丝雀分布在 {len(self.layer_canary_map)} 层")
        print(f"✓ 已初始化 {canary_id} 个金丝雀")

    def inject_gradients(self, per_grad_list, batch_sample_ids, device, clipping_threshold):
        """
        白盒梯度注入：在逐样本梯度的多个方向上注入裁剪阈值C
        
        Args:
            per_grad_list: 逐样本梯度列表 (来自vmap)
            batch_sample_ids: 样本ID (负数表示金丝雀)
            device: 计算设备
            clipping_threshold: 梯度裁剪阈值 (注入的最大梯度值)
        """
        ids_cpu = batch_sample_ids.cpu().numpy() if isinstance(batch_sample_ids, torch.Tensor) else batch_sample_ids
        active_canaries = {}  # {canary_id: batch_pos}
        
        # 找出当前Batch里的金丝雀
        for batch_pos, sid in enumerate(ids_cpu):
            if sid < 0:
                cid = -sid - 1  # 恢复金丝雀ID
                active_canaries[cid] = batch_pos
        
        if not active_canaries:
            return per_grad_list
        
        #print(f"  >> 检测到 {len(active_canaries)} 个金丝雀在当前批次")
        
        # 【修复1】Clone出来，断开计算图，避免In-place修改问题
        per_grad_list = [p.clone() for p in per_grad_list]
        
        # 全局清零：将所有层中这些金丝雀的梯度置零
        for layer_idx, per_grad in enumerate(per_grad_list):
            batch_indices = list(active_canaries.values())
            b_idx = torch.tensor(batch_indices, device=device)
            per_grad[b_idx] = 0.0
        
        # 【新增】多方向注入：在所有分配的位置上注入 clipping_threshold
        param_idx = 0
        for name, param in self.model.named_parameters():
            if param_idx < len(per_grad_list):
                clean_name = name.replace("_module.", "")
                
                if clean_name in self.layer_canary_map:
                    layer_map = self.layer_canary_map[clean_name]
                    valid_cids = set(active_canaries.keys()) & set(layer_map.keys())
                    
                    if valid_cids:
                        for cid in valid_cids:
                            batch_pos = active_canaries[cid]
                            # 获取该金丝雀在本层的所有位置（多个方向）
                            local_indices = layer_map[cid]
                            if not isinstance(local_indices, list):
                                local_indices = [local_indices]
                            
                            # 在所有位置上注入梯度
                            flat_view = per_grad_list[param_idx][batch_pos].view(-1)
                            for local_idx in local_indices:
                                flat_view[local_idx] = clipping_threshold
                
                param_idx += 1
        
        return per_grad_list
    
    def inject_gradients_on_sensitive_dims_fast(self, per_grad_list, batch_sample_ids, device, clipping_threshold, canary_sensitive_map):
        """
        【新增】使用预先构造的敏感维度映射快速注入梯度
        这比每次都遍历快得多，因为已经预先计算了每个金丝雀的敏感维度注入位置
        
        Args:
            per_grad_list: 逐样本梯度列表
            batch_sample_ids: 样本ID (负数表示金丝雀)
            device: 计算设备
            clipping_threshold: 梯度裁剪阈值
            canary_sensitive_map: {canary_id: [(layer_name, local_idx), ...]}
                                 预先构造的每个金丝雀的敏感维度位置
        """
        ids_cpu = batch_sample_ids.cpu().numpy() if isinstance(batch_sample_ids, torch.Tensor) else batch_sample_ids
        active_canaries = {}  # {canary_id: batch_pos}
        
        # 找出当前Batch里的金丝雀
        for batch_pos, sid in enumerate(ids_cpu):
            if sid < 0:
                cid = -sid - 1
                active_canaries[cid] = batch_pos
        
        if not active_canaries:
            return per_grad_list
        
        per_grad_list = [p.clone() for p in per_grad_list]
        
        # 全局清零：将所有金丝雀的梯度置零
        for layer_idx, per_grad in enumerate(per_grad_list):
            batch_indices = list(active_canaries.values())
            b_idx = torch.tensor(batch_indices, device=device)
            per_grad[b_idx] = 0.0
        
        # 【优化】使用预先构造的映射，直接注入（避免重复搜索）
        # 构建 layer_name -> param_idx 的映射（只构建一次）
        layer_to_idx = {}
        param_idx = 0
        for name, param in self.model.named_parameters():
            clean_name = name.replace("_module.", "")
            layer_to_idx[clean_name] = param_idx
            param_idx += 1
        
        # 遍历活跃金丝雀，使用预先构造的敏感维度位置
        for cid in active_canaries.keys():
            if cid not in canary_sensitive_map:
                continue  # 该金丝雀在敏感维度上没有注入位置
            
            batch_pos = active_canaries[cid]
            sensitive_locs = canary_sensitive_map[cid]  # [(layer_name, local_idx), ...]
            
            # 在每个敏感维度位置注入梯度
            for layer_name, local_idx in sensitive_locs:
                if layer_name in layer_to_idx:
                    param_idx = layer_to_idx[layer_name]
                    if param_idx < len(per_grad_list):
                        flat_view = per_grad_list[param_idx][batch_pos].view(-1)
                        if local_idx < len(flat_view):
                            flat_view[local_idx] = clipping_threshold
        
        return per_grad_list
    
    def calculate_scores(self, model, old_params_dict, scores_array):
        """
        计算审计得分：基于全局参数更新审计（与 batch_sample_ids 无关）
        
        这样可以同时审计 IN 和 OUT 金丝雀：
        - IN 金丝雀：得分较高（受梯度下降影响）
        - OUT 金丝雀：得分接近0（仅受噪声影响）
        
        新增：每个金丝雀有多个方向，求和所有方向的更新量
        """
        for name, new_param in model.named_parameters():
            clean_name = name.replace("_module.", "")
            if clean_name not in self.layer_canary_map: 
                continue
            
            old_param = old_params_dict.get(clean_name)
            if old_param is None: 
                continue
            
            # 计算全层更新量: delta = w_old - w_new
            layer_update = (old_param - new_param.data.cpu()).view(-1)
            
            # 【新增】遍历该层所有金丝雀及其多个方向
            for cid, local_indices in self.layer_canary_map[clean_name].items():
                # 处理 local_indices 为列表或单个值
                if not isinstance(local_indices, list):
                    local_indices = [local_indices]
                
                # 汇总所有方向上的参数更新
                for local_idx in local_indices:
                    val = layer_update[local_idx].item()
                    scores_array[cid] += val

    def prepare_sensitive_canary_gradients(self, sensitive_dims):
        """
        【新增】预先为每个金丝雀构造敏感维度梯度注入方案
        这样可以在训练时快速查表，而不需要每次都搜索
        
        Args:
            sensitive_dims: [(layer_name, local_idx), ...] 敏感维度列表
        
        Returns:
            canary_sensitive_map: {canary_id: [(layer_name, local_idx), ...]}
                                 表示每个金丝雀在敏感维度上的注入位置
        """
        # 构建敏感维度的快速查询集合（便于O(1)查找）
        sensitive_dims_set = set(sensitive_dims)
        
        # 为每个金丝雀找出其在敏感维度上的注入位置
        canary_sensitive_map = {}
        
        for cid in range(self.num_canaries):
            if cid not in self.canary_id_to_locs:
                continue
            
            # 该金丝雀的所有注入位置
            all_locs = self.canary_id_to_locs[cid]  # [(layer_name, local_idx), ...]
            
            # 筛选出在敏感维度上的注入位置
            sensitive_locs = [loc for loc in all_locs if loc in sensitive_dims_set]
            
            if sensitive_locs:
                canary_sensitive_map[cid] = sensitive_locs
        
        print(f"\n✓ 已为 {len(canary_sensitive_map)} 个金丝雀预先构造敏感维度梯度")
        print(f"  平均每个金丝雀在敏感维度上有 {sum(len(locs) for locs in canary_sensitive_map.values()) / max(1, len(canary_sensitive_map)):.2f} 个注入位置")
        
        return canary_sensitive_map

    def get_sensitive_dimensions(self, gradient_accumulator, num_dims):
        """
        获取隐私敏感维度（除了0之外最小的维度）
        
        选择策略：
        - 排除梯度绝对值为 0 的位置
        - 从剩余的非零梯度中选择绝对值最小的 num_dims 个
        
        Args:
            gradient_accumulator: 累积梯度 {layer_name: accumulated_grad_tensor}
            num_dims: 要选择的维度数
            
        Returns:
            sensitive_dims: (layer_name, local_idx) 的列表，共num_dims个
            update_scores: 每个维度的累积更新量
        """
        # 合并所有梯度为平坦向量
        all_updates = []
        layer_info = []  # 记录每个梯度块对应的 (layer_name, start_idx, end_idx)
        
        current_idx = 0
        for name in sorted(gradient_accumulator.keys()):
            grad = gradient_accumulator[name]
            if isinstance(grad, torch.Tensor):
                grad_flat = grad.cpu().numpy().flatten()
            else:
                grad_flat = np.array(grad).flatten()
            
            all_updates.append(np.abs(grad_flat))  # 取绝对值
            layer_info.append((name, current_idx, current_idx + len(grad_flat)))
            current_idx += len(grad_flat)
        
        # 合并为一个数组
        all_updates = np.concatenate(all_updates)
        
        # 排除梯度绝对值为 0 的位置
        non_zero_mask = all_updates > 1e-8  # 使用小阈值而非严格的 0
        non_zero_indices = np.where(non_zero_mask)[0]
        non_zero_values = all_updates[non_zero_mask]
        
        if len(non_zero_indices) == 0:
            print(f"\n⚠ 警告：所有梯度更新都接近 0，无法选择敏感维度")
            return [], all_updates
        
        # 从非零梯度中选择绝对值最小的 num_dims 个
        sorted_indices_in_non_zero = np.argsort(non_zero_values)
        selected_count = min(num_dims, len(non_zero_indices))
        selected_indices_in_non_zero = sorted_indices_in_non_zero[:selected_count]
        
        # 映射回全局索引
        selected_indices = non_zero_indices[selected_indices_in_non_zero]
        
        # 转换回 (layer_name, local_idx) 格式
        sensitive_dims = []
        for global_idx in selected_indices:
            for layer_name, start_idx, end_idx in layer_info:
                if start_idx <= global_idx < end_idx:
                    local_idx = global_idx - start_idx
                    sensitive_dims.append((layer_name, int(local_idx)))
                    break
        
        # 统计信息
        selected_values = all_updates[selected_indices]
        print(f"\n✓ 选择了 {len(sensitive_dims)} 个隐私敏感维度（除零最小策略）")
        print(f"  梯度更新量范围: [{selected_values.min():.8f}, {selected_values.max():.8f}]")
        print(f"  平均更新量: {selected_values.mean():.8f}")
        print(f"  非零维度总数: {len(non_zero_indices)}")
        
        return sensitive_dims, all_updates
    
    def inject_gradients_on_sensitive_dims(self, per_grad_list, batch_sample_ids, device, clipping_threshold, sensitive_dims, layer_to_param_idx=None):
        """
        【修改】只在敏感维度上注入梯度
        
        Args:
            per_grad_list: 逐样本梯度列表
            batch_sample_ids: 样本ID (负数表示金丝雀)
            device: 计算设备
            clipping_threshold: 梯度裁剪阈值
            sensitive_dims: [(layer_name, local_idx), ...] 敏感维度列表
            layer_to_param_idx: 缓存的 layer_name -> param_idx 映射
        """
        ids_cpu = batch_sample_ids.cpu().numpy() if isinstance(batch_sample_ids, torch.Tensor) else batch_sample_ids
        active_canaries = {}  # {canary_id: batch_pos}
        
        # 找出当前Batch里的金丝雀
        for batch_pos, sid in enumerate(ids_cpu):
            if sid < 0:
                cid = -sid - 1
                active_canaries[cid] = batch_pos
        
        if not active_canaries:
            return per_grad_list
        
        per_grad_list = [p.clone() for p in per_grad_list]
        
        # 全局清零：将所有金丝雀的梯度置零
        for layer_idx, per_grad in enumerate(per_grad_list):
            batch_indices = list(active_canaries.values())
            b_idx = torch.tensor(batch_indices, device=device)
            per_grad[b_idx] = 0.0
        
        # 【修改】只在敏感维度上注入梯度
        # 如果未提供 layer_to_param_idx，则构建它
        if layer_to_param_idx is None:
            layer_to_param_idx = {}
            param_idx = 0
            for name, param in self.model.named_parameters():
                clean_name = name.replace("_module.", "")
                layer_to_param_idx[clean_name] = param_idx
                param_idx += 1
        
        # 遍历敏感维度，在这些位置注入梯度
        for layer_name, local_idx in sensitive_dims:
            if layer_name in layer_to_param_idx:
                param_idx = layer_to_param_idx[layer_name]
                if param_idx < len(per_grad_list):
                    for cid in active_canaries.keys():
                        batch_pos = active_canaries[cid]
                        flat_view = per_grad_list[param_idx][batch_pos].view(-1)
                        if local_idx < len(flat_view):
                            flat_view[local_idx] = clipping_threshold
        
        return per_grad_list


class trainer:
    def __init__(self, *, model, optimizer, loaders, device, criterion, args, enable_audit=False, num_canaries=0, canary_mask=None, num_directions=5, enable_two_stage=False):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = loaders[0]
        self.val_loader = loaders[1]
        self.test_loader = loaders[2]
        self.device = device
        self.criterion = criterion
        self.args = args
        self.canary_mask = canary_mask
        
        # 【新增】两阶段训练模式
        self.enable_two_stage = enable_two_stage
        self.stage = 1  # 当前阶段 (1=模拟, 2=正常)
        self.gradient_accumulator = {}  # 累积梯度用于计算敏感维度
        self.sensitive_dims = None  # 敏感维度列表
        
        if enable_two_stage:
            print(f"\n✓ 启用两阶段训练")
            print(f"  阶段1: 模拟训练（计算隐私敏感维度）")
            print(f"  阶段2: 正常训练（在敏感维度注入金丝雀梯度）\n")
        
        # 审计相关
        self.enable_audit = enable_audit
        self.num_canaries = num_canaries
        self.num_directions = num_directions
        self.canary_manager = None
        self.canary_sensitive_map = None  # 【新增】预先构造的敏感维度映射
        if enable_audit and num_canaries > 0:
            # 【修改】两阶段模式下延迟初始化（在计算敏感维度后）
            if enable_two_stage:
                # 延迟初始化：先创建空的管理器
                self.canary_manager = OptimizedCanaryManager.__new__(OptimizedCanaryManager)
                self.canary_manager.model = model
                self.canary_manager.num_canaries = num_canaries
                self.canary_manager.num_directions = num_directions
                self.canary_manager.layer_canary_map = {}
                self.canary_manager.canary_id_to_locs = {}
            else:
                # 非两阶段模式：立即初始化
                self.canary_manager = OptimizedCanaryManager(model, num_canaries, num_directions=num_directions)
                self.canary_manager.model_ref = model
            
            self.canary_scores = np.zeros(num_canaries)
            print(f"\n✓ 已启用白盒审计，金丝雀数量: {num_canaries}，方向数: {num_directions}\n")

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
            self.json_recorder.add_record('canary_scores', None)
            self.json_recorder.add_record('audit_info', {
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
        
        # 【新增】两阶段流程
        if self.enable_two_stage:
            # ===== 【关键】保存初始模型参数和缓冲区 =====
            initial_params = {name.replace("_module.", ""): p.clone().detach()
                            for name, p in self.model.named_parameters()}
            # 【新增】保存初始缓冲区（如 BatchNorm 的 running_mean/running_var）
            initial_buffers = {name.replace("_module.", ""): b.clone().detach()
                             for name, b in self.model.named_buffers()}
            
            # ===== 阶段 1：模拟训练 =====
            print("\n" + "="*70)
            print("【 阶段 1】模拟训练 - 计算隐私敏感维度")
            print("="*70)
            self.stage = 1
            self.model.eval()  # 评估模式
            
            for epoch in range(self.args.epoch):
                self.write_log(f'\n【模拟】Epoch: [{epoch}] '.ljust(15) + '#' * 30)
                
                # 只运行训练loop来累积梯度
                if self.train_loader is not None:
                    self._simulate_training(loader=self.train_loader)
            
            # 计算敏感维度
            if self.canary_manager and len(self.gradient_accumulator) > 0:
                self.sensitive_dims, _ = self.canary_manager.get_sensitive_dimensions(
                    self.gradient_accumulator, 
                    num_dims=self.num_canaries
                )
                self.write_log(f"\n✓ 阶段1完成：找到 {len(self.sensitive_dims)} 个敏感维度")
                
                # 【新增】基于敏感维度初始化金丝雀管理器
                self.canary_manager.init_from_sensitive_dims(
                    self.sensitive_dims,
                    num_canaries=self.num_canaries,
                    num_directions=self.num_directions
                )
                self.write_log(f"✓ 已基于敏感维度初始化金丝雀")
            
            # ===== 【关键】将模型参数和缓冲区恢复到初始状态，保证两阶段初始梯度一致 =====
            # 1. 恢复参数
            for name, param in self.model.named_parameters():
                clean_name = name.replace("_module.", "")
                if clean_name in initial_params:
                    param.data.copy_(initial_params[clean_name])
            
            # 2. 【新增】恢复缓冲区（如 BatchNorm 的 running_mean/running_var）
            for name, buffer in self.model.named_buffers():
                clean_name = name.replace("_module.", "")
                if clean_name in initial_buffers:
                    buffer.data.copy_(initial_buffers[clean_name])
            
            # 3. 同时更新 worker 模型参数
            for p_model, p_worker in zip(self.model.parameters(), self.worker_param_func):
                p_worker.copy_(p_model.data)
            
            # 4. 【新增】同时更新 worker 模型缓冲区
            buffer_idx = 0
            for buffer_name, buffer in self.model.named_buffers():
                if buffer_idx < len(self.worker_buffers_func):
                    self.worker_buffers_func[buffer_idx] = buffer.clone().detach()
                buffer_idx += 1
            
            # 5. 【修复】清空 gradient_accumulator，避免与阶段2混淆
            self.gradient_accumulator.clear()
            
            # 6. 重置优化器（注意：参数已恢复到初始状态）
            self.model.train()
            self.optimizer = type(self.optimizer)(self.model.parameters(), **self.optimizer.defaults)
            self.write_log(f"✓ 已将模型参数和缓冲区恢复到初始状态，梯度积累器已清空")
            
            print("\n" + "="*70)
            print("【 阶段 2】正常训练 - 在敏感维度注入金丝雀梯度")
            print("="*70)
            self.stage = 2
        
        # ===== 阶段 2 或正常模式：进行常规训练 =====
        for epoch in range(self.args.epoch):
            if self.enable_two_stage:
                self.write_log(f'\n【审计】Epoch: [{epoch}] '.ljust(15) + '#' * 30)
            else:
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
        
        # 保存审计结果 [修改这里]
        if self.enable_audit and self.canary_manager:
            self.write_log(f'\n=== 审计结果分析 ===')
            self.write_log(f'金丝雀平均得分: {self.canary_scores.mean():.6f}')
            
            # 1. 记录原始分数
            self.json_recorder.add_record('canary_scores', self.canary_scores.tolist())
            self.json_recorder.add_record('canary_mask', self.canary_mask.tolist()) # 记录 Truth 以便复现

            # 2. [新增] 统计成员和非成员的得分分布
            member_scores = self.canary_scores[self.canary_mask]
            non_member_scores = self.canary_scores[~self.canary_mask]
            
            # 输出详细的统计数据
            self.write_log(f'\n--- 成员得分统计 (Members, n={len(member_scores)}) ---')
            if len(member_scores) > 0:
                self.write_log(f'  平均值: {member_scores.mean():.6f}')
                self.write_log(f'  中位数: {np.median(member_scores):.6f}')
                self.write_log(f'  标准差: {member_scores.std():.6f}')
                self.write_log(f'  最小值: {member_scores.min():.6f}')
                self.write_log(f'  最大值: {member_scores.max():.6f}')
                self.write_log(f'  Q1(25%): {np.percentile(member_scores, 25):.6f}')
                self.write_log(f'  Q3(75%): {np.percentile(member_scores, 75):.6f}')
            
            self.write_log(f'\n--- 非成员得分统计 (Non-Members, n={len(non_member_scores)}) ---')
            if len(non_member_scores) > 0:
                self.write_log(f'  平均值: {non_member_scores.mean():.6f}')
                self.write_log(f'  中位数: {np.median(non_member_scores):.6f}')
                self.write_log(f'  标准差: {non_member_scores.std():.6f}')
                self.write_log(f'  最小值: {non_member_scores.min():.6f}')
                self.write_log(f'  最大值: {non_member_scores.max():.6f}')
                self.write_log(f'  Q1(25%): {np.percentile(non_member_scores, 25):.6f}')
                self.write_log(f'  Q3(75%): {np.percentile(non_member_scores, 75):.6f}')
            
            # 得分差异分析
            if len(member_scores) > 0 and len(non_member_scores) > 0:
                self.write_log(f'\n--- 得分分离度分析 ---')
                mean_diff = member_scores.mean() - non_member_scores.mean()
                self.write_log(f'  成员平均 - 非成员平均: {mean_diff:.6f}')
                self.write_log(f'  得分重叠范围: [{max(member_scores.min(), non_member_scores.min()):.6f}, {min(member_scores.max(), non_member_scores.max()):.6f}]')
                
                # 计算两组分布的分离程度 (Cohen's d)
                pooled_std = np.sqrt(((len(member_scores)-1)*member_scores.std()**2 + (len(non_member_scores)-1)*non_member_scores.std()**2) / (len(member_scores) + len(non_member_scores) - 2))
                if pooled_std > 0:
                    cohens_d = mean_diff / pooled_std
                    self.write_log(f"  Cohen's d (效应大小): {cohens_d:.6f}")

            # 3. 调用分析函数计算指标
            try:
                emp_eps, mia_acc = analyze_privacy_leakage(
                    self.canary_scores, 
                    self.canary_mask, 
                    delta=self.args.delta if hasattr(self.args, 'delta') else 1e-5
                )
                
                # 4. 记录分析指标到日志和 JSON
                self.write_log(f'\n--- 隐私泄露评估 ---')
                self.write_log(f'Empirical Epsilon: {emp_eps:.4f}')
                self.write_log(f'MIA Accuracy: {mia_acc:.2%}')
                
                # 记录详细的统计数据到 JSON
                self.json_recorder.add_record('audit_metrics', {
                    'empirical_epsilon': float(emp_eps),
                    'mia_accuracy': float(mia_acc),
                    'num_canaries': self.num_canaries,
                    'member_stats': {
                        'count': int(len(member_scores)),
                        'mean': float(member_scores.mean()) if len(member_scores) > 0 else None,
                        'median': float(np.median(member_scores)) if len(member_scores) > 0 else None,
                        'std': float(member_scores.std()) if len(member_scores) > 0 else None,
                        'min': float(member_scores.min()) if len(member_scores) > 0 else None,
                        'max': float(member_scores.max()) if len(member_scores) > 0 else None,
                    },
                    'non_member_stats': {
                        'count': int(len(non_member_scores)),
                        'mean': float(non_member_scores.mean()) if len(non_member_scores) > 0 else None,
                        'median': float(np.median(non_member_scores)) if len(non_member_scores) > 0 else None,
                        'std': float(non_member_scores.std()) if len(non_member_scores) > 0 else None,
                        'min': float(non_member_scores.min()) if len(non_member_scores) > 0 else None,
                        'max': float(non_member_scores.max()) if len(non_member_scores) > 0 else None,
                    }
                })
            except Exception as e:
                print(f"审计分析出错: {e}")
        
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

    def _simulate_training(self, loader):
        """
        【新增】阶段1：模拟训练
        - 不使用金丝雀，只处理正常样本
        - 完成完整的训练过程（包括参数更新）
        - 不进行DP处理（无梯度裁剪、无噪声注入）
        - 【修复】逐元素累积梯度敏感度（保留空间信息以精确定位敏感维度）
        """
        is_training = True
        self.model.train(is_training)
        
        def compute_loss(model_para, buffers, x, targets):
            predictions = self.worker_model_func(model_para, buffers, x)
            predictions, targets = predictions[:1], targets[:1]
            loss = self.criterion(predictions, targets.flatten())
            return loss
        
        s = time.time()
        for i, batch in enumerate(loader):
            # 忽略金丝雀，只处理正常样本
            if len(batch) == 3:
                x, targets, sample_ids = batch
                # 过滤掉金丝雀（负数ID）
                mask = sample_ids >= 0
                if not mask.any():
                    continue
                x = x[mask]
                targets = targets[mask]
            else:
                x, targets = batch
            
            x, targets = x.to(self.device), targets.to(self.device)
            
            # 保存更新前的参数（用于计算参数变化）
            old_params = {name.replace("_module.", ""): p.clone().detach() 
                         for name, p in self.model.named_parameters()}
            
            self.optimizer.zero_grad()
            
            # 计算梯度
            per_grad = vmap(grad(compute_loss), in_dims=(None, None, 0, 0))(
                self.worker_param_func, self.worker_buffers_func, x, targets
            )
            
            # 【关键】将梯度赋给模型参数（不进行DP处理）
            for p_model, p_per in zip(self.model.parameters(), per_grad):
                # 直接使用平均梯度，不进行裁剪、不添加噪声
                p_model.grad = torch.mean(p_per, dim=0)
            
            # 【修复】累积梯度敏感度（逐元素保留空间信息）
            param_idx = 0
            for name, param in self.model.named_parameters():
                clean_name = name.replace("_module.", "")
                if clean_name not in self.gradient_accumulator:
                    self.gradient_accumulator[clean_name] = torch.zeros_like(param.data)
                
                # 逐元素累积梯度绝对值（不是整层norm，保留空间信息以精确定位敏感维度）
                mean_grad = torch.mean(per_grad[param_idx], dim=0)  # [param_shape]
                self.gradient_accumulator[clean_name] += torch.abs(mean_grad.detach())
                param_idx += 1
            
            # 【关键】进行参数更新
            self.optimizer.step()
            
            # 复制更新后的参数到worker模型
            for p_model, p_worker in zip(self.model.parameters(), self.worker_param_func):
                p_worker.copy_(p_model.data)
        
        print(f"    模拟训练完成: {time.time()-s:.3f} S")

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
            if self.enable_audit:
                old_params = {name.replace("_module.", ""): p.clone().detach().cpu() 
                           for name, p in self.model.named_parameters()}
            for i, batch in enumerate(loader):
                if self.enable_audit:
                    if len(batch) != 3:
                        raise ValueError(
                            "【Audit Error】审计模式下 DataLoader 必须返回 (x, y, sample_ids)，"
                            f"但实际返回了 {len(batch)} 元素。请检查 collate_subgraphs 函数！"
                        )
                    x, targets, sample_ids = batch
                else:
                    # 非审计模式：兼容 2-tuple 或 3-tuple
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
                
                # ========== 新增：白盒梯度注入 ==========
                if self.enable_audit and self.canary_manager:
                    # 【优化】全维度注入（金丝雀已位于敏感维度上）
                    per_grad = self.canary_manager.inject_gradients(
                        per_grad, sample_ids, self.device, self.args.C
                    )
                
                self.other_routine(per_grad)
                
                # ========== 新增：审计计分（在每个batch后计算参数更新） ==========
                if self.enable_audit and self.canary_manager:
                    self.canary_manager.calculate_scores(
                        self.model, old_params, self.canary_scores
                    )
                    # 更新 old_params 为当前参数（用于下一个batch的计算）
                    old_params = {name.replace("_module.", ""): p.clone().detach().cpu() 
                               for name, p in self.model.named_parameters()}

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
