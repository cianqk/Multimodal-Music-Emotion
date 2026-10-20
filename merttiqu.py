import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch
import librosa
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
METADATA_PATH = r"G:\ppppp\KongZhi\.venv\music_dataset\metadata.csv"
MERT_OUTPUT_DIR = r"music_dataset\mert_embeddings"
os.makedirs(MERT_OUTPUT_DIR, exist_ok=True)

MERT_MODEL_NAME = r"G:\ppppp\KongZhi\music_dataset\mert_model\models\m-a-p--MERT-v1-95M\snapshots\master"

TARGET_SR = 24000
MAX_DURATION = 120
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# ==================== 加载模型 ====================
print(f"\n从本地加载 MERT 模型: {MERT_MODEL_NAME}")
processor = AutoProcessor.from_pretrained(
    MERT_MODEL_NAME, trust_remote_code=True, local_files_only=True
)
model = AutoModel.from_pretrained(
    MERT_MODEL_NAME, trust_remote_code=True, local_files_only=True
).to(DEVICE)
model.eval()
print("模型加载完成。")

# ==================== 读取 metadata ====================
def detect_encoding(path):
    for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb18030']:
        try:
            with open(path, 'r', encoding=enc) as f:
                f.read(1024)
            return enc
        except:
            continue
    return 'utf-8'

df = pd.read_csv(METADATA_PATH, encoding=detect_encoding(METADATA_PATH))
df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
print(f"\n数据行数: {len(df)}")

# ==================== 自动定位音频根目录 ====================
def find_audio_base(csv_path, df):
    rel = str(df['audio_path'].iloc[0]).replace('\\', '/')
    candidates = [
        os.path.dirname(os.path.abspath(csv_path)),
        os.path.dirname(os.path.dirname(os.path.abspath(csv_path))),
        os.getcwd(),
        os.path.dirname(os.getcwd()),
        r"G:\ppppp\KongZhi",
        r"G:\ppppp\KongZhi\music_dataset",
    ]
    for base in candidates:
        full = os.path.join(base, rel)
        if os.path.exists(full):
            print(f"[成功] 找到音频根目录: {base}")
            return base
    return None

AUDIO_BASE = find_audio_base(METADATA_PATH, df)
if AUDIO_BASE is None:
    raise SystemExit("[错误] 无法找到音频文件，请检查 audio_path 路径。")

# ==================== 诊断 processor ====================
# 先用一首歌测试 processor 返回什么
test_audio_path = os.path.join(AUDIO_BASE, str(df['audio_path'].iloc[0]).replace('\\', '/'))
test_audio, _ = librosa.load(test_audio_path, sr=TARGET_SR, mono=True)
test_audio = test_audio[:30 * TARGET_SR]

test_inputs = processor(test_audio, sampling_rate=TARGET_SR, return_tensors="pt")
print(f"\n[诊断] processor 返回的 keys: {list(test_inputs.keys())}")
for k, v in test_inputs.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: shape = {v.shape}, dtype = {v.dtype}")

# ==================== MERT 嵌入提取 ====================
def extract_mert_embedding(audio_path, max_duration=120):
    audio, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    max_samples = max_duration * TARGET_SR
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    # ★ 直接传 numpy 数组给 processor
    inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")

    # 确保 input_values 维度正确
    if 'input_values' in inputs:
        input_values = inputs['input_values']
    elif 'input_features' in inputs:
        input_values = inputs['input_features']
    else:
        # 手动构造
        input_values = torch.from_numpy(audio).float().unsqueeze(0)

    # 如果是 1D，扩展为 2D
    if input_values.dim() == 1:
        input_values = input_values.unsqueeze(0)

    input_values = input_values.to(DEVICE)

    with torch.no_grad():
        outputs = model(input_values=input_values, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    stacked = torch.stack(hidden_states[-4:], dim=0).mean(0)
    embedding = stacked.mean(dim=1).squeeze(0)
    return embedding.cpu().numpy().astype(np.float32)

# ==================== 主循环 ====================
print(f"\n开始提取 MERT 嵌入，共 {len(df)} 首 ...")
success, failed = 0, 0
failed_list = []

for i, row in tqdm(df.iterrows(), total=len(df), desc="MERT 提取"):
    song_id = row['id']
    audio_rel = str(row['audio_path']).replace('\\', '/')
    audio_path = os.path.join(AUDIO_BASE, audio_rel)
    save_path = os.path.join(MERT_OUTPUT_DIR, f"{song_id}_mert.npy")

    if os.path.exists(save_path):
        success += 1
        continue

    if not os.path.exists(audio_path):
        failed += 1
        failed_list.append((song_id, audio_path, "文件不存在"))
        continue

    try:
        emb = extract_mert_embedding(audio_path, max_duration=MAX_DURATION)
        np.save(save_path, emb)
        success += 1
    except Exception as e:
        failed += 1
        failed_list.append((song_id, audio_path, str(e)[:100]))
        # 前 3 个失败打印完整错误便于诊断
        if failed <= 3:
            print(f"\n[详细错误] {song_id}: {type(e).__name__}: {e}")

# ==================== 汇总 ====================
print(f"\n{'='*50}")
print(f"提取完成：成功 {success}，失败 {failed}")
print(f"输出目录: {MERT_OUTPUT_DIR}")
print(f"{'='*50}")

if failed_list:
    print("\n失败列表（前 10 条）:")
    for sid, path, reason in failed_list[:10]:
        print(f"  {sid}: {reason}")
    pd.DataFrame(failed_list, columns=['id', 'path', 'reason']).to_csv('mert_failed_list.csv', index=False)

existing = [f for f in os.listdir(MERT_OUTPUT_DIR) if f.endswith('_mert.npy')]
print(f"\n输出目录中已有 {len(existing)} 个 MERT 嵌入文件。")