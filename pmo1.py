import os
import re
import pandas as pd
from tqdm import tqdm

# ==================== 配置 ====================
PMEMO_ROOT = r"G:\pyyyyy\pmo\PMEmo2019"
LYRICS_DIR = os.path.join(PMEMO_ROOT, "lyrics")

BASE_OUTPUT = r"G:\pyyyyy\pmo\scc"
LABELS_CSV = os.path.join(BASE_OUTPUT, "pmemo_labels.csv")
OUTPUT_DIR = os.path.join(BASE_OUTPUT, "pmemo_lyrics_text")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"LYRICS_DIR: {LYRICS_DIR}")
print(f"存在: {os.path.exists(LYRICS_DIR)}\n")

# ==================== LRC 解析 ====================
def parse_lrc(lrc_path):
    """去除时间戳和元信息，提取纯歌词"""
    lines = []
    with open(lrc_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            # 跳过元信息行 [by:xxx]、[ti:xxx] 等
            if re.match(r'^\[(by|ti|ar|al|offset|re|ve|length):', line, re.IGNORECASE):
                continue
            # 去除时间戳 [00:12.28] 或 [00:12.280]
            text = re.sub(r'\[\d{1,2}:\d{2}\.\d{2,3}\]', '', line).strip()
            if text:
                lines.append(text)
    return ' '.join(lines)

# ==================== 批量处理 ====================
df = pd.read_csv(LABELS_CSV)
print(f"标签文件共 {len(df)} 条记录")

success, skipped, failed = 0, 0, 0
failed_list = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="LRC"):
    mid = int(row['musicId'])
    lrc_path = os.path.join(LYRICS_DIR, f"{mid}.lrc")
    save_path = os.path.join(OUTPUT_DIR, f"{mid}.txt")

    if os.path.exists(save_path):
        skipped += 1
        continue

    if not os.path.exists(lrc_path):
        failed += 1
        failed_list.append(mid)
        continue

    try:
        text = parse_lrc(lrc_path)
        # 过滤掉空歌词（只有元信息）
        if len(text.strip()) < 5:
            failed += 1
            failed_list.append(mid)
            continue
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(text)
        success += 1
    except Exception as e:
        failed += 1
        failed_list.append(mid)

print(f"\n{'='*50}")
print(f"完成：成功 {success}，跳过 {skipped}，失败 {failed}")
if failed_list:
    print(f"失败/空歌词前 10 个: {failed_list[:10]}")
print(f"{'='*50}")

# ==================== 验证 ====================
existing = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')]
print(f"输出目录共 {len(existing)} 个歌词文件")

if existing:
    # 打印一个中文/英文样本
    for sample in existing[:3]:
        with open(os.path.join(OUTPUT_DIR, sample), 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"\n样本 {sample}（前 150 字符）:")
        print(text[:150])