import pandas as pd
import numpy as np
from scipy import stats

# ★ 修改这里：完整模型请读取带主导标签的文件
df_early = pd.read_csv('baseline_early_fusion.csv').sort_values('seed')
df_full = pd.read_csv('dom_multimodal_450.csv').sort_values('seed')  # 或者你对应的主实验文件名

metrics = ['emo_macro_f1', 'theme_macro_f1', 'emo_dom_acc', 'theme_dom_acc']
metric_names = {'emo_macro_f1': '情绪 Macro-F1', 'theme_macro_f1': '主题 Macro-F1',
                'emo_dom_acc': '主导情绪 Acc', 'theme_dom_acc': '主导主题 Acc'}

print("=" * 80)
print("早期融合 (Early Fusion) vs 完整模型 (Ours) 的 Wilcoxon 检验")
print("=" * 80)
print(f"{'指标':<15} {'Early Fusion':<15} {'Ours':<15} {'p值':<10} {'显著性'}")
print("-" * 80)

for m in metrics:
    a = df_full[m].values
    b = df_early[m].values
    # 检验 Ours 是否显著大于 Early Fusion
    stat, p = stats.wilcoxon(a, b, alternative='greater')

    if p < 0.001:
        sig = '***'
    elif p < 0.01:
        sig = '**'
    elif p < 0.05:
        sig = '*'
    else:
        sig = 'n.s.'

    print(
        f"{metric_names[m]:<15} {np.mean(b):.4f} ± {np.std(b, ddof=1):.4f}  {np.mean(a):.4f} ± {np.std(a, ddof=1):.4f}  {p:.4f}     {sig}")