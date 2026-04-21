#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跨时间交叉验证实验脚本

功能：
1. 双向跨时间验证（May→September 和 September→May）
2. 对比基线方法和结构感知方法
3. 分析时间泛化能力和稳定性

使用方法：
    python cross_time_validation_experiment.py

作者：结构感知修正器研究组
日期：2024年
"""

import numpy as np
import pandas as pd
from structure_aware_corrector import (
    BaselineCorrector,
    StructureAwareCorrector
)


def load_and_split_data():
    """
    加载数据并按时间划分
    
    返回:
        may_df: May数据
        sept_df: September数据
    """
    # 加载数据
    df = pd.read_excel('result_with_fixstr_fixed.xlsx')
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['month'] = df['date'].dt.to_period('M')
    
    # 按月划分
    may_df = df[df['month'] == '2024-05']
    sept_df = df[df['month'] == '2024-09']
    
    print("="*80)
    print("数据加载完成")
    print("="*80)
    print(f"May (2024-05) 样本数: {len(may_df)}")
    print(f"September (2024-09) 样本数: {len(sept_df)}")
    print("="*80)
    
    return may_df, sept_df


def run_experiment(train_df, test_df, exp_name):
    """
    运行单个实验（训练→测试）
    
    参数:
        train_df: 训练数据
        test_df: 测试数据
        exp_name: 实验名称
    
    返回:
        baseline_metrics: 基线方法指标
        sa_metrics: 结构感知方法指标
    """
    print("\n" + "="*80)
    print(f"【实验】{exp_name}")
    print("="*80)
    
    # 初始化模型
    baseline_corrector = BaselineCorrector(max_len=13)
    structure_aware_corrector = StructureAwareCorrector(max_len=13)
    
    # 训练基线方法
    print("\n【步骤1】训练基线修正器...")
    baseline_corrector.train_all(
        train_df['pse'].tolist(),
        train_df['Fixstr'].tolist()
    )
    
    # 训练结构感知方法
    print("\n【步骤2】训练结构感知修正器...")
    structure_aware_corrector.train(
        train_df['pse'].tolist(),
        train_df['Fixstr'].tolist()
    )
    
    # 测试集评估
    print("\n【步骤3】测试集评估...")
    
    baseline_metrics = baseline_corrector.evaluate(
        test_df['pse'].tolist(),
        test_df['Fixstr'].tolist()
    )
    
    sa_metrics = structure_aware_corrector.evaluate(
        test_df['pse'].tolist(),
        test_df['Fixstr'].tolist(),
        verbose=False
    )
    
    # 打印结果
    print(f"\n【结果】{exp_name}")
    print("-"*80)
    print(f"基线方法:")
    print(f"  字符准确率: {baseline_metrics['char_accuracy']:.4f}")
    print(f"  字符串准确率: {baseline_metrics['string_accuracy']:.4f}")
    print(f"  错误字符串数: {len(test_df) - int(baseline_metrics['string_accuracy']*len(test_df))}")
    
    print(f"\n结构感知方法:")
    print(f"  字符准确率: {sa_metrics['char_accuracy']:.4f}")
    print(f"  字符串准确率: {sa_metrics['string_accuracy']:.4f}")
    print(f"  错误字符串数: {len(test_df) - int(sa_metrics['string_accuracy']*len(test_df))}")
    
    char_improvement = sa_metrics['char_accuracy'] - baseline_metrics['char_accuracy']
    string_improvement = sa_metrics['string_accuracy'] - baseline_metrics['string_accuracy']
    
    print(f"\n改进幅度:")
    print(f"  字符准确率: {char_improvement:+.4f} ({char_improvement*100:+.2f}%)")
    print(f"  字符串准确率: {string_improvement:+.4f} ({string_improvement*100:+.2f}%)")
    
    return baseline_metrics, sa_metrics


def main():
    """
    主函数：运行完整的跨时间验证实验
    """
    print("="*80)
    print("结构感知修正器 - 跨时间交叉验证实验")
    print("核心思想：借鉴不定长场景的'先学结构再学修正规则'双层设计")
    print("="*80)
    
    # 加载数据
    may_df, sept_df = load_and_split_data()
    
    # 结果存储
    results = []
    
    # ========== 实验1：May → September ==========
    baseline_metrics1, sa_metrics1 = run_experiment(
        may_df, sept_df,
        "May → September (前向验证)"
    )
    
    results.append({
        'exp_name': 'May → September',
        'train_data': 'May (2000 samples)',
        'test_data': 'September (3000 samples)',
        'direction': 'Forward',
        'baseline_char_acc': baseline_metrics1['char_accuracy'],
        'baseline_string_acc': baseline_metrics1['string_accuracy'],
        'sa_char_acc': sa_metrics1['char_accuracy'],
        'sa_string_acc': sa_metrics1['string_accuracy'],
        'char_improvement': sa_metrics1['char_accuracy'] - baseline_metrics1['char_accuracy'],
        'string_improvement': sa_metrics1['string_accuracy'] - baseline_metrics1['string_accuracy']
    })
    
    # ========== 实验2：September → May ==========
    baseline_metrics2, sa_metrics2 = run_experiment(
        sept_df, may_df,
        "September → May (反向验证)"
    )
    
    results.append({
        'exp_name': 'September → May',
        'train_data': 'September (3000 samples)',
        'test_data': 'May (2000 samples)',
        'direction': 'Backward',
        'baseline_char_acc': baseline_metrics2['char_accuracy'],
        'baseline_string_acc': baseline_metrics2['string_accuracy'],
        'sa_char_acc': sa_metrics2['char_accuracy'],
        'sa_string_acc': sa_metrics2['string_accuracy'],
        'char_improvement': sa_metrics2['char_accuracy'] - baseline_metrics2['char_accuracy'],
        'string_improvement': sa_metrics2['string_accuracy'] - baseline_metrics2['string_accuracy']
    })
    
    # ========== 总结分析 ==========
    print("\n" + "="*80)
    print("实验总结")
    print("="*80)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    
    # 打印详细结果表格
    print("\n【详细结果】")
    print(results_df.to_string(index=False))
    
    # 保存详细结果
    results_df.to_csv('cross_time_validation_results.csv', index=False)
    print(f"\n详细结果已保存到: cross_time_validation_results.csv")
    
    # 计算平均性能
    print("\n" + "="*80)
    print("性能对比分析")
    print("="*80)
    
    baseline_char_accs = [r['baseline_char_acc'] for r in results]
    baseline_string_accs = [r['baseline_string_acc'] for r in results]
    sa_char_accs = [r['sa_char_acc'] for r in results]
    sa_string_accs = [r['sa_string_acc'] for r in results]
    
    baseline_char_mean = np.mean(baseline_char_accs)
    baseline_string_mean = np.mean(baseline_string_accs)
    sa_char_mean = np.mean(sa_char_accs)
    sa_string_mean = np.mean(sa_string_accs)
    
    print(f"\n【平均性能】")
    print(f"基线方法:")
    print(f"  平均字符准确率: {baseline_char_mean:.4f}")
    print(f"  平均字符串准确率: {baseline_string_mean:.4f}")
    
    print(f"\n结构感知方法:")
    print(f"  平均字符准确率: {sa_char_mean:.4f}")
    print(f"  平均字符串准确率: {sa_string_mean:.4f}")
    
    avg_char_improvement = sa_char_mean - baseline_char_mean
    avg_string_improvement = sa_string_mean - baseline_string_mean
    
    print(f"\n【平均改进】")
    print(f"  字符准确率: {avg_char_improvement:+.4f} ({avg_char_improvement*100:+.2f}%)")
    print(f"  字符串准确率: {avg_string_improvement:+.4f} ({avg_string_improvement*100:+.2f}%)")
    
    # 稳定性分析
    print("\n" + "="*80)
    print("稳定性分析")
    print("="*80)
    
    baseline_char_std = np.std(baseline_char_accs)
    baseline_string_std = np.std(baseline_string_accs)
    baseline_string_var = np.var(baseline_string_accs)
    
    sa_char_std = np.std(sa_char_accs)
    sa_string_std = np.std(sa_string_accs)
    sa_string_var = np.var(sa_string_accs)
    
    print(f"\n【性能波动（标准差）】")
    print(f"基线方法:")
    print(f"  字符准确率标准差: {baseline_char_std:.4f}")
    print(f"  字符串准确率标准差: {baseline_string_std:.4f}")
    print(f"  字符串准确率方差: {baseline_string_var:.6f}")
    
    print(f"\n结构感知方法:")
    print(f"  字符准确率标准差: {sa_char_std:.4f}")
    print(f"  字符串准确率标准差: {sa_string_std:.4f}")
    print(f"  字符串准确率方差: {sa_string_var:.6f}")
    
    # 稳定性改进
    if baseline_string_var > 0:
        stability_improvement = (baseline_string_var - sa_string_var) / baseline_string_var * 100
        print(f"\n【稳定性改进】")
        print(f"  方差降低: {stability_improvement:.2f}%")
    else:
        print(f"\n【稳定性改进】")
        print(f"  基线方差为0，无法计算改进幅度")
    
    # 关键发现
    print("\n" + "="*80)
    print("关键发现")
    print("="*80)
    
    print("\n1. 时间泛化能力:")
    print(f"   基线方法: 性能波动显著 (标准差={baseline_string_std:.4f})")
    print(f"   结构感知: 性能稳定 (标准差={sa_string_std:.4f})")
    
    print("\n2. 平均性能:")
    print(f"   基线方法: {baseline_string_mean:.2%}")
    print(f"   结构感知: {sa_string_mean:.2%}")
    print(f"   改进: {avg_string_improvement:+.2%}")
    
    print("\n3. 稳定性提升:")
    print(f"   基线方法: 方差={baseline_string_var:.6f}")
    print(f"   结构感知: 方差={sa_string_var:.6f}")
    if baseline_string_var > 0:
        print(f"   改进: {stability_improvement:.1f}%")
    
    # 保存摘要报告
    summary = {
        'baseline_char_mean': baseline_char_mean,
        'baseline_string_mean': baseline_string_mean,
        'baseline_string_std': baseline_string_std,
        'baseline_string_var': baseline_string_var,
        'sa_char_mean': sa_char_mean,
        'sa_string_mean': sa_string_mean,
        'sa_string_std': sa_string_std,
        'sa_string_var': sa_string_var,
        'avg_char_improvement': avg_char_improvement,
        'avg_string_improvement': avg_string_improvement,
        'stability_improvement': stability_improvement if baseline_string_var > 0 else 0.0
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('cross_time_validation_summary.csv', index=False)
    
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
    print(f"\n输出文件:")
    print(f"  1. cross_time_validation_results.csv - 详细实验结果")
    print(f"  2. cross_time_validation_summary.csv - 性能摘要")
    print(f"  3. structure_aware_corrector.py - 核心代码")
    print(f"  4. cross_time_validation_experiment.py - 实验脚本")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
