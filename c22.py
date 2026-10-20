import os
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
from transformers import BertTokenizer, BertModel
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
METADATA_PATH = r"G:\ppppp\KongZhi\.venv\music_dataset\metadata.csv"
MERT_DIR = r"music_dataset\mert_embeddings"
MODEL_NAME = 'bert-base-chinese'
MAX_LEN = 256

BATCH_SIZE = 8
EPOCHS = 15
LR_BERT = 1e-5
LR_OTHER = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMOTION_LABELS = ['emotion_intense', 'emotion_soothing', 'emotion_upbeat', 'emotion_sad']
THEME_LABELS = ['theme_love', 'theme_inspirational', 'theme_missing',
                'theme_lonely', 'theme_youth', 'theme_passionate', 'theme_xianxia']
NUM_EMOTION = 4
NUM_THEME = 7

# ★ 方案 A：只跑 cross_attn，全 10 种子
SEEDS = [30, 48, 52, 66, 77, 89, 109, 111, 152, 167]
FUSION_MODES = ['cross_attn']

# ==================== 自动定位歌词根目录 ====================
METADATA_PATH_ABS = os.path.abspath(METADATA_PATH)
print(f"metadata.csv 绝对路径: {METADATA_PATH_ABS}")

def find_lyrics_base(csv_path):
    sample = pd.read_csv(csv_path, encoding='utf-8-sig', nrows=1)
    rel = str(sample['lyrics_path'].iloc[0]).replace('\\', '/')
    candidates = [
        os.path.dirname(csv_path),
        os.path.dirname(os.path.dirname(csv_path)),
        os.getcwd(),
        os.path.dirname(os.getcwd()),
        os.path.dirname(os.path.dirname(os.getcwd())),
        r"G:\ppppp\KongZhi",
        r"G:\ppppp\KongZhi\.venv",
        r"G:\ppppp\KongZhi\music_dataset",
    ]
    for base in candidates:
        full = os.path.join(base, rel)
        if os.path.exists(full):
            print(f"[成功] 歌词根目录: {base}")
            return base
    return None

LYRICS_BASE = find_lyrics_base(METADATA_PATH_ABS)
if LYRICS_BASE is None:
    raise SystemExit("[错误] 无法找到歌词文件。")

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

df = pd.read_csv(METADATA_PATH_ABS, encoding=detect_encoding(METADATA_PATH_ABS))
df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
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
theme_dom = df['theme_dominant'].fillna(0).astype(int).values

# ==================== 三划分分层函数 ====================
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
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class MultiModalDataset(Dataset):
    def __init__(self, lyrics, audio, emo_labels, theme_labels):
        self.lyrics = lyrics
        self.audio = audio
        self.emo_labels = emo_labels
        self.theme_labels = theme_labels

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
        }

# ==================== 跨模态注意力 ====================
class CrossModalTransformer(nn.Module):
    def __init__(self, dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.modal_emb = nn.Parameter(torch.zeros(2, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads,
            dim_feedforward=dim * 2, dropout=dropout,
            batch_first=True, activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, audio, text):
        tokens = torch.stack([audio, text], dim=1)
        tokens = tokens + self.modal_emb.unsqueeze(0)
        encoded = self.transformer(tokens)
        encoded = self.norm(encoded)
        return encoded[:, 0, :], encoded[:, 1, :]

# ==================== 端到端多模态模型 ====================
class E2EMultiModalModel(nn.Module):
    def __init__(self, audio_dim=768, proj_dim=128, hidden_dim=256,
                 num_emotion=4, num_theme=7, dropout=0.3, fusion_mode='cross_attn'):
        super().__init__()
        self.fusion_mode = fusion_mode

        self.bert = BertModel.from_pretrained(MODEL_NAME)
        for param in self.bert.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.bert.pooler.parameters():
            param.requires_grad = True

        self.text_proj = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if fusion_mode == 'cross_attn':
            self.cross_modal = CrossModalTransformer(proj_dim, num_heads=4, dropout=0.1)

        fusion_dim = 2 * proj_dim
        self.fusion_encoder = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.emotion_branch = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.theme_branch = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.emo_head = nn.Linear(128, num_emotion)
        self.theme_head = nn.Linear(128, num_theme)

    def forward(self, input_ids, attention_mask, audio):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = bert_out.last_hidden_state[:, 0, :]
        text_feat = self.text_proj(text_emb)
        audio_feat = self.audio_encoder(audio)

        if self.fusion_mode == 'cross_attn':
            audio_feat, text_feat = self.cross_modal(audio_feat, text_feat)

        fused = torch.cat([audio_feat, text_feat], dim=1)
        shared = self.fusion_encoder(fused)

        emo_feat = self.emotion_branch(shared)
        theme_feat = self.theme_branch(shared)
        return self.emo_head(emo_feat), self.theme_head(theme_feat)

# ==================== 评估 ====================
def evaluate(model, loader):
    model.eval()
    emo_pred, emo_true = [], []
    theme_pred, theme_true = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(DEVICE)
            am = batch['attention_mask'].to(DEVICE)
            au = batch['audio'].to(DEVICE)
            emo_logits, theme_logits = model(ids, am, au)
            emo_pred.append(torch.sigmoid(emo_logits).cpu().numpy())
            emo_true.append(batch['emo_labels'].numpy())
            theme_pred.append(torch.sigmoid(theme_logits).cpu().numpy())
            theme_true.append(batch['theme_labels'].numpy())
    return (np.vstack(emo_pred), np.vstack(emo_true),
            np.vstack(theme_pred), np.vstack(theme_true))

# ==================== 单次训练 ====================
def train_one(seed, fusion_mode):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"\n{'='*60}\n种子 {seed} | 融合 {fusion_mode}\n{'='*60}")

    train_idx, val_idx, test_idx = stratified_split_three_way(
        df, test_size=0.15, val_size=0.15, random_state=seed
    )

    scaler_a = StandardScaler()
    audio_train = scaler_a.fit_transform(audio_feats[train_idx]).astype(np.float32)
    audio_val = scaler_a.transform(audio_feats[val_idx]).astype(np.float32)
    audio_test = scaler_a.transform(audio_feats[test_idx]).astype(np.float32)

    emo_multi = multi_labels[:, :4]
    theme_multi = multi_labels[:, 4:]

    lyrics_train = df['lyrics_text'].values[train_idx].tolist()
    lyrics_val = df['lyrics_text'].values[val_idx].tolist()
    lyrics_test = df['lyrics_text'].values[test_idx].tolist()

    train_ds = MultiModalDataset(lyrics_train, audio_train, emo_multi[train_idx], theme_multi[train_idx])
    val_ds = MultiModalDataset(lyrics_val, audio_val, emo_multi[val_idx], theme_multi[val_idx])
    test_ds = MultiModalDataset(lyrics_test, audio_test, emo_multi[test_idx], theme_multi[test_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = E2EMultiModalModel(fusion_mode=fusion_mode).to(DEVICE)

    emo_pos = emo_multi[train_idx].sum(axis=0)
    emo_neg = len(train_idx) - emo_pos
    emo_pw = torch.clamp(torch.tensor(emo_neg / (emo_pos + 1e-6), dtype=torch.float32).to(DEVICE), max=10.0)
    criterion_emo = nn.BCEWithLogitsLoss(pos_weight=emo_pw)

    theme_pos = theme_multi[train_idx].sum(axis=0)
    theme_neg = len(train_idx) - theme_pos
    theme_pw = torch.clamp(torch.tensor(theme_neg / (theme_pos + 1e-6), dtype=torch.float32).to(DEVICE), max=10.0)
    criterion_theme = nn.BCEWithLogitsLoss(pos_weight=theme_pw)

    bert_params = [p for p in model.bert.parameters() if p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if not n.startswith('bert')]
    optimizer = optim.AdamW([
        {'params': bert_params, 'lr': LR_BERT},
        {'params': other_params, 'lr': LR_OTHER}
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val = 0.0
    patience = 5
    trigger = 0
    model_path = f"best_c3_e2e_{fusion_mode}_seed{seed}.pth"

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            ids = batch['input_ids'].to(DEVICE)
            am = batch['attention_mask'].to(DEVICE)
            au = batch['audio'].to(DEVICE)
            emo_y = batch['emo_labels'].to(DEVICE)
            theme_y = batch['theme_labels'].to(DEVICE)

            optimizer.zero_grad()
            emo_logits, theme_logits = model(ids, am, au)
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
        'fusion_mode': fusion_mode,
        'emo_macro_f1': emo_macro,
        'emo_micro_f1': emo_micro,
        'theme_macro_f1': theme_macro,
        'theme_micro_f1': theme_micro,
    }

# ==================== 主循环 ====================
if __name__ == '__main__':
    results = []
    for fusion_mode in FUSION_MODES:
        for seed in SEEDS:
            res = train_one(seed, fusion_mode)
            results.append(res)

    result_df = pd.DataFrame(results)
    result_df.to_csv('c3_e2e_10seeds.csv', index=False)

    print("\n" + "="*60)
    print("C3 端到端多模态结果（cross_attn，10 种子）")
    print("="*60)
    print(result_df.to_string(index=False))

    print("\n均值±标准差:")
    print(result_df[['emo_macro_f1','emo_micro_f1','theme_macro_f1','theme_micro_f1']].agg(['mean','std']))

    print("\n结果已保存到 c3_e2e_10seeds.csv")
    print("\n【对比参照】")
    print("  B3 端到端文本（10种子）：情绪 0.634 / 主题 0.678")
    print("  C3 端到端多模态（10种子）：见上方均值")
    print("\n【判断标准】")
    print("  若 C3 情绪 > 0.634 且 p<0.05 → 多模态在情绪上有效")
    print("  若 C3 主题 < 0.678      → 多模态在主题上引入噪声")
    print("  两个结论可以同时成立，构成'模态-任务不对称'的发现")