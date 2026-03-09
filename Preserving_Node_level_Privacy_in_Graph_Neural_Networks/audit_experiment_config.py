#!/usr/bin/env python3
"""
审计配置与对比实验脚本

用于系统地对比不同epsilon和canaries下的隐私泄露风险
"""

import json
import os
from pathlib import Path
import subprocess
import sys

# ========== 实验配置 ==========

# 隐私参数配置
PRIVACY_CONFIGS = {
    "strong_privacy": {
        "priv_epsilon": 0.5,
        "description": "强隐私保护"
    },
    "medium_privacy": {
        "priv_epsilon": 2.0,
        "description": "中等隐私保护"
    },
    "weak_privacy": {
        "priv_epsilon": 8.0,
        "description": "弱隐私保护"
    },
}

# 审计配置
AUDIT_CONFIGS = {
    "light_audit": {
        "num_canaries": 50,
        "description": "轻量审计"
    },
    "medium_audit": {
        "num_canaries": 100,
        "description": "中等审计"
    },
    "heavy_audit": {
        "num_canaries": 200,
        "description": "重量级审计"
    },
}

# ========== 实验矩阵 ==========

EXPERIMENTS = [
    # 强隐私 + 不同审计强度
    {
        "name": "strong_privacy_light",
        "privacy": "strong_privacy",
        "audit": "light_audit",
        "epochs": 50,
        "batch_size": 256,
    },
    {
        "name": "strong_privacy_medium",
        "privacy": "strong_privacy",
        "audit": "medium_audit",
        "epochs": 50,
        "batch_size": 256,
    },
    {
        "name": "strong_privacy_heavy",
        "privacy": "strong_privacy",
        "audit": "heavy_audit",
        "epochs": 50,
        "batch_size": 256,
    },
    
    # 中等隐私 + 不同审计强度
    {
        "name": "medium_privacy_light",
        "privacy": "medium_privacy",
        "audit": "light_audit",
        "epochs": 50,
        "batch_size": 256,
    },
    {
        "name": "medium_privacy_medium",
        "privacy": "medium_privacy",
        "audit": "medium_audit",
        "epochs": 50,
        "batch_size": 256,
    },
    {
        "name": "medium_privacy_heavy",
        "privacy": "medium_privacy",
        "audit": "heavy_audit",
        "epochs": 50,
        "batch_size": 256,
    },
    
    # 弱隐私 + 不同审计强度
    {
        "name": "weak_privacy_light",
        "privacy": "weak_privacy",
        "audit": "light_audit",
        "epochs": 50,
        "batch_size": 256,
    },
    {
        "name": "weak_privacy_medium",
        "privacy": "weak_privacy",
        "audit": "medium_audit",
        "epochs": 50,
        "batch_size": 256,
    },
    {
        "name": "weak_privacy_heavy",
        "privacy": "weak_privacy",
        "audit": "heavy_audit",
        "epochs": 50,
        "batch_size": 256,
    },
]


def generate_experiment_report():
    """生成实验设计报告"""
    
    report = """
# DP-SGD 白盒审计实验设计
    
## 实验矩阵 (3x3)

|                  | Light (50) | Medium (100) | Heavy (200) |
|------------------|-----------|-------------|-----------|
| **Strong ε=0.5** | ✓ 1       | ✓ 2        | ✓ 3       |
| **Medium ε=2.0** | ✓ 4       | ✓ 5        | ✓ 6       |
| **Weak ε=8.0**   | ✓ 7       | ✓ 8        | ✓ 9       |

## 预期结果

### 隐私泄露风险矩阵

- **强隐私 + 轻审计**：低风险（可能无法检测泄露）
- **强隐私 + 重审计**：低风险（检测灵敏度高）
- **弱隐私 + 轻审计**：中风险（可能漏检）
- **弱隐私 + 重审计**：高风险（清晰检测泄露）

### 关键指标

1. **Canary Scores**: 监测金丝雀梯度变化
   - 高分(>0.01) → 存在隐私泄露
   - 低分(<0.001) → 隐私保护充分

2. **Model Accuracy**: 验证隐私不影响效用
   - 应该保持相近

3. **Training Time**: 审计的计算开销
   - num_canaries ↑ → time ↑ (线性关系)

## 运行命令

### 单个实验
```bash
python audit_experiment.py --exp strong_privacy_medium
```

### 全部实验
```bash
python audit_experiment.py --all
```

### 生成报告
```bash
python audit_experiment.py --analyze
```
"""
    return report


def generate_config_args(exp_config):
    """根据实验配置生成训练参数"""
    
    privacy = PRIVACY_CONFIGS[exp_config["privacy"]]
    audit = AUDIT_CONFIGS[exp_config["audit"]]
    
    args = {
        "priv_epsilon": privacy["priv_epsilon"],
        "num_canaries": audit["num_canaries"],
        "epochs": exp_config["epochs"],
        "batch_size": exp_config["batch_size"],
        "exp_name": exp_config["name"],
        "audit_enabled": True,
    }
    
    return args


def analyze_results():
    """分析所有实验结果"""
    
    results_dir = Path("data_records")
    
    if not results_dir.exists():
        print("❌ 未找到结果目录")
        return
    
    print("\n" + "="*70)
    print("审计实验结果分析")
    print("="*70)
    
    # 查找所有audit结果
    audit_results = {}
    
    for json_file in results_dir.glob("jd_*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # data 是列表，每个元素是一次运行的结果
                if isinstance(data, list) and len(data) > 0:
                    last_run = data[-1]  # 最后一次运行
                    
                    if 'canary_scores' in last_run and last_run['canary_scores']:
                        scores = last_run['canary_scores'][-1]  # 最后一个epoch的得分
                        if isinstance(scores, list):
                            scores = scores
                        
                        audit_info = last_run.get('audit_info', {})
                        num_canaries = audit_info.get('num_canaries', 'Unknown')
                        
                        # 提取epsilon和canary信息
                        filename = json_file.stem
                        
                        audit_results[filename] = {
                            "num_canaries": num_canaries,
                            "canary_scores": scores if isinstance(scores, list) else [scores],
                            "avg_score": sum(scores) / len(scores) if isinstance(scores, list) else scores,
                            "max_score": max(scores) if isinstance(scores, list) else scores,
                            "min_score": min(scores) if isinstance(scores, list) else scores,
                            "test_acc": last_run.get('test_acc', [None])[-1] if last_run.get('test_acc') else None,
                        }
        except Exception as e:
            print(f"⚠️  无法读取 {json_file}: {e}")
    
    # 打印结果表格
    if audit_results:
        print("\n📊 审计结果总结\n")
        print(f"{'实验名':<30} {'金丝雀数':<12} {'平均得分':<15} {'最大得分':<15} {'模型准确率':<12}")
        print("-" * 84)
        
        for exp_name, results in sorted(audit_results.items()):
            print(f"{exp_name:<30} {results['num_canaries']:<12} "
                  f"{results['avg_score']:<15.6f} {results['max_score']:<15.6f} "
                  f"{str(results['test_acc']):<12}")
    else:
        print("⚠️  没有找到审计结果")
    
    print("\n📈 解释\n")
    print("- 平均得分 > 0.01：存在明显隐私泄露风险")
    print("- 平均得分 0.001-0.01：存在潜在隐私泄露")
    print("- 平均得分 < 0.001：隐私保护充分")
    print()


def print_experiment_matrix():
    """打印实验矩阵"""
    
    print("\n" + "="*70)
    print("实验矩阵 (epsilon × canaries)")
    print("="*70 + "\n")
    
    print("隐私强度配置:")
    for key, config in PRIVACY_CONFIGS.items():
        print(f"  {key:20s}: epsilon={config['priv_epsilon']:<4} ({config['description']})")
    
    print("\n审计强度配置:")
    for key, config in AUDIT_CONFIGS.items():
        print(f"  {key:20s}: canaries={config['num_canaries']:<4} ({config['description']})")
    
    print("\n实验列表:")
    for i, exp in enumerate(EXPERIMENTS, 1):
        privacy = PRIVACY_CONFIGS[exp["privacy"]]
        audit = AUDIT_CONFIGS[exp["audit"]]
        print(f"  [{i}] {exp['name']:30s} | eps={privacy['priv_epsilon']:<4} "
              f"canaries={audit['num_canaries']:<4}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DP-SGD白盒审计实验配置工具")
    parser.add_argument("--matrix", action="store_true", help="显示实验矩阵")
    parser.add_argument("--report", action="store_true", help="生成实验设计报告")
    parser.add_argument("--analyze", action="store_true", help="分析所有实验结果")
    parser.add_argument("--config", type=str, help="显示特定实验配置")
    
    args = parser.parse_args()
    
    if args.matrix:
        print_experiment_matrix()
    
    if args.report:
        report = generate_experiment_report()
        print(report)
        # 保存到文件
        with open("EXPERIMENT_DESIGN.md", "w") as f:
            f.write(report)
        print("\n✓ 报告已保存到 EXPERIMENT_DESIGN.md")
    
    if args.analyze:
        analyze_results()
    
    if args.config:
        # 查找匹配的实验
        matching = [e for e in EXPERIMENTS if args.config in e["name"]]
        if matching:
            exp = matching[0]
            config_args = generate_config_args(exp)
            print(json.dumps(config_args, indent=2))
        else:
            print(f"❌ 找不到实验 '{args.config}'")
    
    if not any([args.matrix, args.report, args.analyze, args.config]):
        print_experiment_matrix()
        print("\n💡 提示: 使用 --help 查看更多选项")
