import os
os.environ["HF_HOME"] = r"D:\hug\huggingface"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
METADATA_PATH = r"G:\ppppp\KongZhi\.venv\music_dataset\metadata.csv"
MERT_DIR = r"G:\ppppp\KongZhi\music_dataset\mert_embeddings"

BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMOTION_LABELS = ['emotion_intense', 'emotion_soothing', 'emotion_upbeat', 'emotion_sad']
THEME_LABELS = ['theme_love', 'theme_inspirational', 'theme_missing',
                'theme_lonely', 'theme_youth', 'theme_passionate', 'theme_xianxia']
NUM_EMOTION = 4
NUM_THEME = 7

SEEDS = [30, 48, 52, 66, 77, 89, 109, 111, 152, 167]

# ==================== 数据加载 ====================
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
df = df.dropna(subset=['id', 'theme_dominant', 'emotion_dominant']).reset_index(drop=True)
print(f"数据行数: {len(df)}")

def load_npy(path, dim=768):
    try:
        return np.load(path).astype(np.float32)
    except:
        return np.zeros(dim, dtype=np.float32)

audio_feats = []
missing = []
for _, row in df.iterrows():
    mert_path = os.path.join(MERT_DIR, f"{row['id']}_mert.npy")
    if not os.path.exists(mert_path):
        missing.append(row['id'])
    audio_feats.append(load_npy(mert_path, 768))
audio_feats = np.array(audio_feats, dtype=np.float32)

if missing:
    print(f"[警告] {len(missing)} 个 MERT 文件缺失，前 5 个: {missing[:5]}")
print(f"音频特征 shape: {audio_feats.shape}")

multi_labels = df[EMOTION_LABELS + THEME_LABELS].fillna(0).values.astype(np.float32)
theme_dom = df['theme_dominant'].fillna(0).astype(int).values

# ==================== 三划分分层函数（修正版） ====================
def stratified_split_three_way(df, test_size=0.15, val_size=0.15, random_state=42):
    stratify_col = df['theme_dominant'].astype(str)
    counts = stratify_col.value_counts()
    rare_labels = counts[counts < 2].index
    stratify_col = stratify_col.apply(lambda x: 'rare' if x in rare_labels else x)
    if (stratify_col.value_counts() < 2).any():
        idx = np.arange(len(df))
        train_val_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size/(1-test_size), random_state=random_state)
        return train_idx, val_idx, test_idx

    idx = np.arange(len(df))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_val_idx, test_idx in sss1.split(idx, stratify_col):
        pass
    stratify_col_train_val = stratify_col.iloc[train_val_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=random_state)
    for sub_train_idx, sub_val_idx in sss2.split(train_val_idx, stratify_col_train_val):
        pass
    train_idx = train_val_idx[sub_train_idx]
    val_idx = train_val_idx[sub_val_idx]
    return train_idx, val_idx, test_idx

# ==================== Dataset ====================
class AudioDataset(Dataset):
    def __init__(self, audio, emo_labels, theme_labels):
        self.audio = audio
        self.emo_labels = emo_labels
        self.theme_labels = theme_labels

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, item):
        return {
            'audio': torch.tensor(self.audio[item], dtype=torch.float32),
            'emo_labels': torch.tensor(self.emo_labels[item], dtype=torch.float32),
            'theme_labels': torch.tensor(self.theme_labels[item], dtype=torch.float32),
        }

# ==================== 模型 ====================
class AudioMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_emotion=4, num_theme=7, dropout=0.3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.emo_head = nn.Linear(128, num_emotion)
        self.theme_head = nn.Linear(128, num_theme)

    def forward(self, x):
        feat = self.shared(x)
        return self.emo_head(feat), self.theme_head(feat)

# ==================== 评估 ====================
def evaluate(model, loader):
    model.eval()
    emo_pred, emo_true = [], []
    theme_pred, theme_true = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['audio'].to(DEVICE)
            emo_logits, theme_logits = model(x)
            emo_pred.append(torch.sigmoid(emo_logits).cpu().numpy())
            emo_true.append(batch['emo_labels'].numpy())
            theme_pred.append(torch.sigmoid(theme_logits).cpu().numpy())
            theme_true.append(batch['theme_labels'].numpy())
    return (np.vstack(emo_pred), np.vstack(emo_true),
            np.vstack(theme_pred), np.vstack(theme_true))

# ==================== 单次训练 ====================
def train_one(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"\n{'='*60}\n种子 {seed}\n{'='*60}")

    train_idx, val_idx, test_idx = stratified_split_three_way(
        df, test_size=0.15, val_size=0.15, random_state=seed
    )

    scaler = StandardScaler()
    audio_train = scaler.fit_transform(audio_feats[train_idx]).astype(np.float32)
    audio_val = scaler.transform(audio_feats[val_idx]).astype(np.float32)
    audio_test = scaler.transform(audio_feats[test_idx]).astype(np.float32)

    emo_multi = multi_labels[:, :4]
    theme_multi = multi_labels[:, 4:]

    train_ds = AudioDataset(audio_train, emo_multi[train_idx], theme_multi[train_idx])
    val_ds = AudioDataset(audio_val, emo_multi[val_idx], theme_multi[val_idx])
    test_ds = AudioDataset(audio_test, emo_multi[test_idx], theme_multi[test_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = AudioMLP(input_dim=768).to(DEVICE)

    emo_pos = emo_multi[train_idx].sum(axis=0)
    emo_neg = len(train_idx) - emo_pos
    emo_pw = torch.clamp(torch.tensor(emo_neg / (emo_pos + 1e-6), dtype=torch.float32).to(DEVICE), max=10.0)
    criterion_emo = nn.BCEWithLogitsLoss(pos_weight=emo_pw)

    theme_pos = theme_multi[train_idx].sum(axis=0)
    theme_neg = len(train_idx) - theme_pos
    theme_pw = torch.clamp(torch.tensor(theme_neg / (theme_pos + 1e-6), dtype=torch.float32).to(DEVICE), max=10.0)
    criterion_theme = nn.BCEWithLogitsLoss(pos_weight=theme_pw)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val = 0.0
    patience = 5
    trigger = 0
    model_path = f"best_b4_final_seed{seed}.pth"

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x = batch['audio'].to(DEVICE)
            emo_y = batch['emo_labels'].to(DEVICE)
            theme_y = batch['theme_labels'].to(DEVICE)

            optimizer.zero_grad()
            emo_logits, theme_logits = model(x)
            loss = criterion_emo(emo_logits, emo_y) + criterion_theme(theme_logits, theme_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        emo_p, emo_t, theme_p, theme_t = evaluate(model, val_loader)
        emo_bin = (emo_p > 0.3).astype(int)
        theme_bin = (theme_p > 0.3).astype(int)
        emo_macro = f1_score(emo_t, emo_bin, average='macro', zero_division=0)
        theme_macro = f1_score(theme_t, theme_bin, average='macro', zero_division=0)
        val_combined = (emo_macro + theme_macro) / 2

        print(f"Epoch {epoch+1:02d} | Loss {total_loss/len(train_loader):.4f} | "
              f"Val Emo-F1 {emo_macro:.4f} | Val Theme-F1 {theme_macro:.4f} | Combined {val_combined:.4f}")

        if val_combined > best_val:
            best_val = val_combined
            torch.save(model.state_dict(), model_path)
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                print(f"[早停] Epoch {epoch+1}，最佳验证综合 Macro-F1: {best_val:.4f}")
                break

    print("\n加载最佳模型进行最终评估...")
    model.load_state_dict(torch.load(model_path))
    emo_p, emo_t, theme_p, theme_t = evaluate(model, test_loader)
    emo_bin = (emo_p > 0.3).astype(int)
    theme_bin = (theme_p > 0.3).astype(int)

    emo_macro = f1_score(emo_t, emo_bin, average='macro', zero_division=0)
    emo_micro = f1_score(emo_t, emo_bin, average='micro', zero_division=0)
    theme_macro = f1_score(theme_t, theme_bin, average='macro', zero_division=0)
    theme_micro = f1_score(theme_t, theme_bin, average='micro', zero_division=0)

    print(f"测试集 | Emo-Macro-F1 {emo_macro:.4f} | Emo-Micro-F1 {emo_micro:.4f} | "
          f"Theme-Macro-F1 {theme_macro:.4f} | Theme-Micro-F1 {theme_micro:.4f}")

    return {
        'seed': seed,
        'emo_macro_f1': emo_macro,
        'emo_micro_f1': emo_micro,
        'theme_macro_f1': theme_macro,
        'theme_micro_f1': theme_micro,
    }

# ==================== 主循环 ====================
if __name__ == '__main__':
    results = []
    for seed in SEEDS:
        res = train_one(seed)
        results.append(res)

    result_df = pd.DataFrame(results)
    result_df.to_csv('b4_final_450.csv', index=False)

    print("\n" + "="*60)
    print("B4 音频单模态最终结果（450 首，10 种子）")
    print("="*60)
    print(result_df)
    print("\n均值±标准差:")
    print(result_df[['emo_macro_f1','emo_micro_f1','theme_macro_f1','theme_micro_f1']].agg(['mean','std']))
    print("\n结果已保存到 b4_final_450.csv")
    print("\n【对比参照】")
    print("  B3 文本单模态（450 首）：情绪 0.752 / 主题 0.662")