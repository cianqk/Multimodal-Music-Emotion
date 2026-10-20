import os
os.environ["HF_HOME"] = r"D:\hug\huggingface"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from transformers import BertTokenizer, BertModel
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
METADATA_PATH = r"G:\ppppp\KongZhi\.venv\music_dataset\metadata.csv"
MERT_DIR = r"G:\ppppp\KongZhi\music_dataset\mert_embeddings"
MODEL_NAME = 'bert-base-chinese'
MAX_LEN = 256
BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ★ 5 个种子
SEEDS = [30, 48, 52, 66, 77, 89, 109, 111, 152, 167]

EMOTION_LABELS = ['emotion_intense', 'emotion_soothing', 'emotion_upbeat', 'emotion_sad']
THEME_LABELS = ['theme_love', 'theme_inspirational', 'theme_missing',
                'theme_lonely', 'theme_youth', 'theme_passionate', 'theme_xianxia']
NUM_EMOTION = 4
NUM_THEME = 7

# ==================== 自动定位歌词根目录 ====================
METADATA_PATH_ABS = os.path.abspath(METADATA_PATH)

def find_lyrics_base(csv_path):
    sample = pd.read_csv(csv_path, encoding='utf-8-sig', nrows=1)
    rel = str(sample['lyrics_path'].iloc[0]).replace('\\', '/')
    candidates = [
        os.path.dirname(csv_path),
        os.path.dirname(os.path.dirname(csv_path)),
        os.getcwd(),
        os.path.dirname(os.getcwd()),
        os.path.dirname(os.path.dirname(os.getcwd())),
        r"G:\ppppp\KongZhi\.venv",
        r"G:\ppppp\KongZhi",
    ]
    for base in candidates:
        full = os.path.join(base, rel)
        if os.path.exists(full):
            return base
    return None

LYRICS_BASE = find_lyrics_base(METADATA_PATH_ABS)
if LYRICS_BASE is None:
    raise SystemExit("[错误] 无法找到歌词文件。")

# ==================== 数据加载（只加载一次） ====================
def detect_encoding(path):
    for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb18030']:
        try:
            with open(path, 'r', encoding=enc) as f:
                f.read(1024)
            return enc
        except:
            continue
    return 'utf-8'

df = pd.read_csv(METADATA_PATH_ABS, encoding=detect_encoding(METADATA_PATH_ABS))
df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
df = df.dropna(subset=['id', 'theme_dominant', 'emotion_dominant']).reset_index(drop=True)
print(f"数据行数: {len(df)}")

def read_lyrics(path):
    rel = str(path).replace('\\', '/')
    full = os.path.join(LYRICS_BASE, rel)
    try:
        with open(full, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read().strip()
    except:
        return ""

df['lyrics_text'] = df['lyrics_path'].apply(read_lyrics)
df = df[df['lyrics_text'] != ""].reset_index(drop=True)
print(f"加载了 {len(df)} 首歌的歌词。")

def load_npy(path, dim=768):
    try:
        return np.load(path).astype(np.float32)
    except:
        return np.zeros(dim, dtype=np.float32)

audio_feats = []
for _, row in df.iterrows():
    a_path = os.path.join(MERT_DIR, f"{row['id']}_mert.npy")
    audio_feats.append(load_npy(a_path, 768))
audio_feats = np.array(audio_feats, dtype=np.float32)
print(f"音频特征 shape: {audio_feats.shape}")

multi_labels = df[EMOTION_LABELS + THEME_LABELS].fillna(0).values.astype(np.float32)
emotion_dom = df['emotion_dominant'].fillna(0).astype(int).values
theme_dom = df['theme_dominant'].fillna(0).astype(int).values

# ==================== 划分函数 ====================
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
    return train_val_idx[sub_train_idx], train_val_idx[sub_val_idx], test_idx

# ==================== Dataset ====================
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class MultiModalDataset(Dataset):
    def __init__(self, lyrics, audio, emo_labels, theme_labels, emo_dom, theme_dom):
        self.lyrics = lyrics
        self.audio = audio
        self.emo_labels = emo_labels
        self.theme_labels = theme_labels
        self.emo_dom = emo_dom
        self.theme_dom = theme_dom

    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, item):
        enc = tokenizer(
            str(self.lyrics[item]),
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'audio': torch.tensor(self.audio[item], dtype=torch.float32),
            'emo_labels': torch.tensor(self.emo_labels[item], dtype=torch.float32),
            'theme_labels': torch.tensor(self.theme_labels[item], dtype=torch.float32),
            'emo_dom': torch.tensor(self.emo_dom[item], dtype=torch.long),
            'theme_dom': torch.tensor(self.theme_dom[item], dtype=torch.long),
        }

# ==================== 模型定义（与主导标签实验一致，4 头） ====================
class CrossModalTransformer(nn.Module):
    def __init__(self, dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.modal_emb = nn.Parameter(torch.zeros(2, dim))
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads,
                                            dim_feedforward=dim * 2,
                                            dropout=dropout, batch_first=True, activation='relu')
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, audio, text):
        tokens = torch.stack([audio, text], dim=1) + self.modal_emb.unsqueeze(0)
        encoded = self.norm(self.transformer(tokens))
        return encoded[:, 0, :], encoded[:, 1, :]

class E2EMultiModalModel(nn.Module):
    def __init__(self, audio_dim=768, proj_dim=128, hidden_dim=256,
                 num_emotion=4, num_theme=7, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        for p in self.bert.parameters():
            p.requires_grad = False
        for layer in self.bert.encoder.layer[-2:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.bert.pooler.parameters():
            p.requires_grad = True

        self.text_proj = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, proj_dim),
            nn.BatchNorm1d(proj_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
            nn.BatchNorm1d(proj_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.cross_modal = CrossModalTransformer(proj_dim, num_heads=4, dropout=0.1)
        fusion_dim = 2 * proj_dim
        self.fusion_encoder = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.emotion_branch = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout)
        )
        self.theme_branch = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout)
        )
        self.emo_multi_head = nn.Linear(128, num_emotion)
        self.theme_multi_head = nn.Linear(128, num_theme)
        self.emo_dom_head = nn.Linear(128, num_emotion)
        self.theme_dom_head = nn.Linear(128, num_theme)

    def forward(self, input_ids, attention_mask, audio):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = bert_out.last_hidden_state[:, 0, :]
        text_feat = self.text_proj(text_emb)
        audio_feat = self.audio_encoder(audio)
        audio_feat, text_feat = self.cross_modal(audio_feat, text_feat)
        fused = torch.cat([audio_feat, text_feat], dim=1)
        shared = self.fusion_encoder(fused)
        emo_feat = self.emotion_branch(shared)
        theme_feat = self.theme_branch(shared)
        return (self.emo_multi_head(emo_feat), self.theme_multi_head(theme_feat),
                self.emo_dom_head(emo_feat), self.theme_dom_head(theme_feat))

# ==================== 评估函数 ====================
def evaluate_with_missing(model, loader, missing=None):
    emo_mp, emo_mt = [], []
    theme_mp, theme_mt = [], []
    emo_dp, emo_dt = [], []
    theme_dp, theme_dt = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            audio = batch['audio'].to(DEVICE)

            if missing == 'audio':
                audio = torch.zeros_like(audio)
            elif missing == 'text':
                cls_id = tokenizer.cls_token_id
                sep_id = tokenizer.sep_token_id
                input_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
                input_ids[:, 0] = cls_id
                input_ids[:, 1] = sep_id
                attention_mask = torch.zeros_like(attention_mask)
                attention_mask[:, 0] = 1
                attention_mask[:, 1] = 1

            emo_ml, theme_ml, emo_dl, theme_dl = model(input_ids, attention_mask, audio)
            emo_mp.append(torch.sigmoid(emo_ml).cpu().numpy())
            emo_mt.append(batch['emo_labels'].numpy())
            theme_mp.append(torch.sigmoid(theme_ml).cpu().numpy())
            theme_mt.append(batch['theme_labels'].numpy())
            emo_dp.append(torch.argmax(emo_dl, 1).cpu().numpy())
            emo_dt.append(batch['emo_dom'].numpy())
            theme_dp.append(torch.argmax(theme_dl, 1).cpu().numpy())
            theme_dt.append(batch['theme_dom'].numpy())

    emo_mp = np.vstack(emo_mp); emo_mt = np.vstack(emo_mt)
    theme_mp = np.vstack(theme_mp); theme_mt = np.vstack(theme_mt)
    emo_dp = np.concatenate(emo_dp); emo_dt = np.concatenate(emo_dt)
    theme_dp = np.concatenate(theme_dp); theme_dt = np.concatenate(theme_dt)

    emo_bin = (emo_mp > 0.3).astype(int)
    theme_bin = (theme_mp > 0.3).astype(int)

    return {
        'emo_macro_f1': f1_score(emo_mt, emo_bin, average='macro', zero_division=0),
        'emo_micro_f1': f1_score(emo_mt, emo_bin, average='micro', zero_division=0),
        'theme_macro_f1': f1_score(theme_mt, theme_bin, average='macro', zero_division=0),
        'theme_micro_f1': f1_score(theme_mt, theme_bin, average='micro', zero_division=0),
        'emo_dom_acc': accuracy_score(emo_dt, emo_dp),
        'theme_dom_acc': accuracy_score(theme_dt, theme_dp),
    }

# ==================== 主循环 ====================
all_results = []

for SEED in SEEDS:
    print(f"\n{'='*60}")
    print(f"种子 {SEED}")
    print(f"{'='*60}")

    MODEL_PATH = f"best_dom_multimodal_seed{SEED}.pth"
    if not os.path.exists(MODEL_PATH):
        print(f"[跳过] 模型文件不存在: {MODEL_PATH}")
        continue

    # 使用与训练相同的划分
    train_idx, val_idx, test_idx = stratified_split_three_way(df, random_state=SEED)
    print(f"测试集大小: {len(test_idx)}")

    # 音频标准化
    scaler = StandardScaler()
    audio_feats_scaled = scaler.fit_transform(audio_feats).astype(np.float32)

    # 测试集
    test_lyrics = df['lyrics_text'].values[test_idx].tolist()
    test_audio = audio_feats_scaled[test_idx]
    test_emo_multi = multi_labels[test_idx, :4]
    test_theme_multi = multi_labels[test_idx, 4:]
    test_emo_dom = emotion_dom[test_idx]
    test_theme_dom = theme_dom[test_idx]

    test_ds = MultiModalDataset(test_lyrics, test_audio, test_emo_multi, test_theme_multi,
                                 test_emo_dom, test_theme_dom)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 加载模型
    model = E2EMultiModalModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print(f"已加载: {MODEL_PATH}")

    # 三种情况
    full = evaluate_with_missing(model, test_loader, missing=None)
    miss_a = evaluate_with_missing(model, test_loader, missing='audio')
    miss_t = evaluate_with_missing(model, test_loader, missing='text')

    for cond, res in [('full', full), ('missing_audio', miss_a), ('missing_text', miss_t)]:
        row = {'seed': SEED, 'condition': cond}
        row.update(res)
        all_results.append(row)
        print(f"  {cond}: Emo-F1 {res['emo_macro_f1']:.4f} | Theme-F1 {res['theme_macro_f1']:.4f} | "
              f"EmoDom {res['emo_dom_acc']:.4f} | ThemeDom {res['theme_dom_acc']:.4f}")

# ==================== 汇总 ====================
result_df = pd.DataFrame(all_results)
result_df.to_csv('modality_missing_5seeds.csv', index=False)

print("\n" + "="*60)
print("模态缺失实验汇总（5 种子）")
print("="*60)
print(result_df.to_string(index=False))

# 按条件分组统计
print("\n按条件分组均值：")
metrics = ['emo_macro_f1','emo_micro_f1','theme_macro_f1','theme_micro_f1',
           'emo_dom_acc','theme_dom_acc']
print(result_df.groupby('condition')[metrics].agg(['mean','std']))

print("\n结果已保存到 modality_missing_5seeds.csv")