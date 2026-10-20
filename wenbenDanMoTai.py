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
from sklearn.metrics import f1_score
from transformers import BertTokenizer, BertModel
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
METADATA_PATH = r"G:\ppppp\KongZhi\.venv\music_dataset\metadata.csv"
MODEL_NAME = 'bert-base-chinese'
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 20
LR_BERT = 1e-5
LR_HEAD = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMOTION_LABELS = ['emotion_intense', 'emotion_soothing', 'emotion_upbeat', 'emotion_sad']
THEME_LABELS = ['theme_love', 'theme_inspirational', 'theme_missing',
                'theme_lonely', 'theme_youth', 'theme_passionate', 'theme_xianxia']
NUM_EMOTION = 4
NUM_THEME = 7

SEEDS = [30, 48, 52, 66, 77, 89, 109, 111, 152, 167]

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
        r"G:\ppppp\KongZhi\.venv",
        r"G:\ppppp\KongZhi",
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
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class LyricsDataset(Dataset):
    def __init__(self, lyrics, emo_labels, theme_labels):
        self.lyrics = lyrics
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
            'emo_labels': torch.tensor(self.emo_labels[item], dtype=torch.float32),
            'theme_labels': torch.tensor(self.theme_labels[item], dtype=torch.float32),
        }

# ==================== 模型（CLS 池化 + 冻结底层） ====================
class BertMultiLabelModel(nn.Module):
    def __init__(self, num_emotion=4, num_theme=7, dropout=0.5):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        for param in self.bert.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.bert.pooler.parameters():
            param.requires_grad = True

        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.emo_head = nn.Linear(hidden, num_emotion)
        self.theme_head = nn.Linear(hidden, num_theme)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = out.last_hidden_state[:, 0, :]
        pooled = self.dropout(cls_embedding)
        return self.emo_head(pooled), self.theme_head(pooled)

# ==================== 评估 ====================
def evaluate(model, loader):
    model.eval()
    emo_pred, emo_true = [], []
    theme_pred, theme_true = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(DEVICE)
            am = batch['attention_mask'].to(DEVICE)
            emo_logits, theme_logits = model(ids, am)
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

    emo_multi = multi_labels[:, :4]
    theme_multi = multi_labels[:, 4:]

    lyrics_train = df['lyrics_text'].values[train_idx].tolist()
    lyrics_val = df['lyrics_text'].values[val_idx].tolist()
    lyrics_test = df['lyrics_text'].values[test_idx].tolist()

    train_ds = LyricsDataset(lyrics_train, emo_multi[train_idx], theme_multi[train_idx])
    val_ds = LyricsDataset(lyrics_val, emo_multi[val_idx], theme_multi[val_idx])
    test_ds = LyricsDataset(lyrics_test, emo_multi[test_idx], theme_multi[test_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BertMultiLabelModel().to(DEVICE)

    emo_pos = emo_multi[train_idx].sum(axis=0)
    emo_neg = len(train_idx) - emo_pos
    emo_pw = torch.clamp(torch.tensor(emo_neg / (emo_pos + 1e-6), dtype=torch.float32).to(DEVICE), max=10.0)
    criterion_emo = nn.BCEWithLogitsLoss(pos_weight=emo_pw)

    theme_pos = theme_multi[train_idx].sum(axis=0)
    theme_neg = len(train_idx) - theme_pos
    theme_pw = torch.clamp(torch.tensor(theme_neg / (theme_pos + 1e-6), dtype=torch.float32).to(DEVICE), max=10.0)
    criterion_theme = nn.BCEWithLogitsLoss(pos_weight=theme_pw)

    bert_params = [p for p in model.bert.parameters() if p.requires_grad]
    head_params = list(model.emo_head.parameters()) + list(model.theme_head.parameters())
    optimizer = optim.AdamW([
        {'params': bert_params, 'lr': LR_BERT},
        {'params': head_params, 'lr': LR_HEAD}
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val = 0.0
    patience = 7
    trigger = 0
    model_path = f"best_b3_final_seed{seed}.pth"

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            ids = batch['input_ids'].to(DEVICE)
            am = batch['attention_mask'].to(DEVICE)
            emo_y = batch['emo_labels'].to(DEVICE)
            theme_y = batch['theme_labels'].to(DEVICE)

            optimizer.zero_grad()
            emo_logits, theme_logits = model(ids, am)
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
        val_score = 2 * emo_macro * theme_macro / (emo_macro + theme_macro + 1e-6)

        print(f"Epoch {epoch+1:02d} | Loss {total_loss/len(train_loader):.4f} | "
              f"Val Emo-F1 {emo_macro:.4f} | Val Theme-F1 {theme_macro:.4f} | Harmonic {val_score:.4f}")

        if val_score > best_val:
            best_val = val_score
            torch.save(model.state_dict(), model_path)
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                print(f"[早停] Epoch {epoch+1}，最佳验证调和平均: {best_val:.4f}")
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
    result_df.to_csv('b3_final_450.csv', index=False)

    print("\n" + "="*60)
    print("B3 文本单模态最终结果（450 首，10 种子）")
    print("="*60)
    print(result_df)
    print("\n均值±标准差:")
    print(result_df[['emo_macro_f1','emo_micro_f1','theme_macro_f1','theme_micro_f1']].agg(['mean','std']))
    print("\n结果已保存到 b3_final_450.csv")