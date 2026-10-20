import pandas as pd
import numpy as np
from scipy import stats

# 读取上一步跑出的数据
df = pd.read_csv('ablation_capacity_matched.csv')
pivot = df.pivot(index='seed', columns='mode')

metrics = ['emo_macro_f1', 'theme_macro_f1', 'emo_dom_acc', 'theme_dom_acc']
metric_names = {
    'emo_macro_f1': '情绪 Macro-F1',
    'theme_macro_f1': '主题 Macro-F1',
    'emo_dom_acc': '主导情绪 Acc',
    'theme_dom_acc': '主导主题 Acc',
}

print("=" * 80)
print("参数量匹配消融实验结果（均值 ± 标准差）与 Wilcoxon 检验")
print("=" * 80)
print(f"{'指标':<15} {'解耦 (Full)':<22} {'共享 (Matched)':<22} {'p值':<10} {'显著性'}")
print("-" * 85)

for m in metrics:
    a = pivot[(m, 'full')].values
    b = pivot[(m, 'shared_matched')].values

    mean_a, std_a = np.mean(a), np.std(a, ddof=1)
    mean_b, std_b = np.mean(b), np.std(b, ddof=1)

    stat, p = stats.wilcoxon(a, b, alternative='two-sided')

    if p < 0.001:
        sig = '***'
    elif p < 0.01:
        sig = '**'
    elif p < 0.05:
        sig = '*'
    else:
        sig = 'n.s.'

    print(f"{metric_names[m]:<15} {mean_a:.4f} ± {std_a:.4f}      {mean_b:.4f} ± {std_b:.4f}      {p:.4f}     {sig}")

print("=" * 80)