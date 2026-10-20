import pandas as pd
import numpy as np
from scipy import stats

# ==================== 读取数据 ====================
dom_text = pd.read_csv('dom_text_450.csv')
dom_audio = pd.read_csv('dom_audio_450.csv')
dom_multi = pd.read_csv('dom_multimodal_450.csv')

# 按种子排序，保证配对
dom_text = dom_text.sort_values('seed').reset_index(drop=True)
dom_audio = dom_audio.sort_values('seed').reset_index(drop=True)
dom_multi = dom_multi.sort_values('seed').reset_index(drop=True)

METRICS = ['emo_macro_f1', 'theme_macro_f1', 'emo_dom_acc', 'theme_dom_acc']
METRIC_NAMES = {
    'emo_macro_f1': '情绪 Macro-F1',
    'theme_macro_f1': '主题 Macro-F1',
    'emo_dom_acc': '主导情绪 Acc',
    'theme_dom_acc': '主导主题 Acc',
}

# ==================== 配对 Wilcoxon 检验 ====================
def paired_wilcoxon(a, b, name_a, name_b, metric):
    """
    a, b: 两个方法的指标数组
    返回差值均值、W统计量、p值、显著性标记
    """
    diff = a - b
    mean_diff = np.mean(diff)
    # 如果所有差值都为0，Wilcoxon会报错
    if np.all(diff == 0):
        return mean_diff, np.nan, 1.0, 'n.s.'
    try:
        stat, p = stats.wilcoxon(a, b, alternative='two-sided')
    except ValueError:
        return mean_diff, np.nan, 1.0, 'n.s.'
    return mean_diff, stat, p, p

# ==================== 对比组 ====================
comparisons = [
    ('多模态 vs 文本', dom_multi, dom_text),
    ('多模态 vs 音频', dom_multi, dom_audio),
    ('文本 vs 音频', dom_text, dom_audio),
]

# ==================== 执行检验 ====================
print("=" * 80)
print("配对 Wilcoxon 符号秩检验（10 种子，双侧）")
print("=" * 80)

all_results = []
for comp_name, df_a, df_b in comparisons:
    print(f"\n【{comp_name}】")
    print(f"{'指标':<20} {'均值A':<10} {'均值B':<10} {'差值':<10} {'p值':<12} {'显著性'}")
    print("-" * 75)
    for metric in METRICS:
        a = df_a[metric].values
        b = df_b[metric].values
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        mean_diff, stat, p, _ = paired_wilcoxon(a, b, comp_name.split(' vs ')[0], comp_name.split(' vs ')[1], metric)
        # 显著性标记
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = 'n.s.'
        print(f"{METRIC_NAMES[metric]:<20} {mean_a:<10.4f} {mean_b:<10.4f} {mean_diff:+.4f}    {p:<12.4f} {sig}")
        all_results.append({
            'comparison': comp_name,
            'metric': metric,
            'mean_A': mean_a,
            'mean_B': mean_b,
            'diff': mean_diff,
            'p_value': p,
            'sig': sig,
        })

# ==================== Bonferroni 校正 ====================
print("\n" + "=" * 80)
print("Bonferroni 校正（每个对比内 4 个指标，α = 0.05/4 = 0.0125）")
print("=" * 80)

n_tests = len(METRICS)
alpha = 0.05
alpha_corrected = alpha / n_tests

print(f"\n校正后显著性水平: α = {alpha_corrected:.4f}")
print(f"\n{'对比':<25} {'指标':<20} {'原始p':<12} {'校正后判断'}")
print("-" * 75)
for r in all_results:
    if r['p_value'] < alpha_corrected:
        verdict = '显著'
    else:
        verdict = '不显著'
    print(f"{r['comparison']:<25} {METRIC_NAMES[r['metric']]:<20} {r['p_value']:<12.4f} {verdict}")

# ==================== 保存 ====================
result_df = pd.DataFrame(all_results)
result_df.to_csv('wilcoxon_results.csv', index=False)
print("\n结果已保存到 wilcoxon_results.csv")

# ==================== 模态缺失实验检验 ====================
print("\n" + "=" * 80)
print("模态缺失实验：Wilcoxon 检验（10 种子）")
print("=" * 80)

miss_df = pd.read_csv('modality_missing_10seeds.csv')

# 按条件提取
full = miss_df[miss_df['condition'] == 'full'].sort_values('seed').reset_index(drop=True)
miss_a = miss_df[miss_df['condition'] == 'missing_audio'].sort_values('seed').reset_index(drop=True)
miss_t = miss_df[miss_df['condition'] == 'missing_text'].sort_values('seed').reset_index(drop=True)

miss_metrics = ['emo_macro_f1', 'theme_macro_f1', 'emo_dom_acc', 'theme_dom_acc']
miss_metric_names = {
    'emo_macro_f1': '情绪 Macro-F1',
    'theme_macro_f1': '主题 Macro-F1',
    'emo_dom_acc': '主导情绪 Acc',
    'theme_dom_acc': '主导主题 Acc',
}

for comp_name, df_a, df_b in [('完整 vs 音频缺失', full, miss_a),
                                ('完整 vs 歌词缺失', full, miss_t)]:
    print(f"\n【{comp_name}】")
    print(f"{'指标':<20} {'完整':<10} {'缺失':<10} {'差值':<10} {'p值':<12} {'显著性'}")
    print("-" * 75)
    for metric in miss_metrics:
        a = df_a[metric].values
        b = df_b[metric].values
        mean_diff, stat, p, _ = paired_wilcoxon(a, b, 'full', 'missing', metric)
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = 'n.s.'
        print(f"{miss_metric_names[metric]:<20} {np.mean(a):<10.4f} {np.mean(b):<10.4f} {mean_diff:+.4f}    {p:<12.4f} {sig}")