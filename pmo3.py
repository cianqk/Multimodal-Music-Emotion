import pandas as pd
import numpy as np
from scipy import stats

# 读取 10 种子数据
df = pd.read_csv('pmemo_transfer_10seeds.csv')
pivot = df.pivot(index='seed', columns='method')

metrics = ['valence_acc', 'valence_f1', 'arousal_acc', 'arousal_f1']
metric_names = {
    'valence_acc': 'Valence Acc',
    'valence_f1': 'Valence F1',
    'arousal_acc': 'Arousal Acc',
    'arousal_f1': 'Arousal F1',
}

def paired_wilcoxon(a, b):
    diff = a - b
    if np.all(diff == 0):
        return 1.0, 0.0
    try:
        stat, p = stats.wilcoxon(a, b, alternative='two-sided')
    except ValueError:
        p = 1.0
    return p, np.mean(diff)

comparisons = [
    ('finetune', 'zero_shot', '微调 vs 零样本'),
    ('finetune', 'scratch', '微调 vs 域内'),
    ('zero_shot', 'scratch', '零样本 vs 域内'),
]

print("=" * 80)
print("PMEmo 迁移实验 Wilcoxon 配对检验（10 种子，双侧）")
print("=" * 80)

for m1, m2, name in comparisons:
    print(f"\n【{name}】")
    print(f"{'指标':<15} {'均值(' + m1 + ')':<18} {'均值(' + m2 + ')':<18} {'差值':<10} {'p值':<10} {'显著性'}")
    print("-" * 85)
    for metric in metrics:
        a = pivot[(metric, m1)].values
        b = pivot[(metric, m2)].values
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        p, mean_diff = paired_wilcoxon(a, b)
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = 'n.s.'
        print(f"{metric_names[metric]:<15} {mean_a:<18.4f} {mean_b:<18.4f} {mean_diff:+.4f}    {p:<10.4f} {sig}")