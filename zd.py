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
from sklearn.metrics import f1_score, accuracy_score
from transformers import BertTokenizer, BertModel
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
# ★★★ 只改这一处：'text' / 'audio' / 'multimodal' ★★★
MODE = 'multimodal'

METADATA_PATH = r"G:\ppppp\KongZhi\.venv\music_dataset\metadata.csv"
MERT_DIR = r"G:\ppppp\KongZhi\music_dataset\mert_embeddings"
MODEL_NAME = 'bert-base-chinese'
MAX_LEN = 256

BATCH_SIZE = 8 if MODE in ['text', 'multimodal'] else 32
EPOCHS = 15 if MODE in ['text', 'multimodal'] else 30
LR_BERT = 1e-5
LR_HEAD = 1e-3
LR_OTHER = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMOTION_LABELS = ['emotion_intense', 'emotion_soothing', 'emotion_upbeat', 'emotion_sad']
THEME_LABELS = ['theme_love', 'theme_inspirational', 'theme_missing',
                'theme_lonely', 'theme_youth', 'theme_passionate', 'theme_xianxia']
NUM_EMOTION = 4
NUM_THEME = 7

ALPHA, BETA, GAMMA, DELTA = 1.0, 1.0, 0.5, 0.5

SEEDS = [30, 48, 52, 66, 77, 89, 109, 111, 152, 167]

OUTPUT_CSV = f'dom_{MODE}_450.csv'

print(f"当前 MODE = {MODE}")
print(f"使用设备: {DEVICE}")

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
if LYRICS_BASE is None and MODE in ['text', 'multimodal']:
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

if MODE in ['text', 'multimodal']:
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

if MODE in ['audio', 'multimodal']:
    audio_feats = []
    for _, row in df.iterrows():
        a_path = os.path.join(MERT_DIR, f"{row['id']}_mert.npy")
        audio_feats.append(load_npy(a_path, 768))
    audio_feats = np.array(audio_feats, dtype=np.float32)
    print(f"音频特征 shape: {audio_feats.shape}")

multi_labels = df[EMOTION_LABELS + THEME_LABELS].fillna(0).values.astype(np.float32)
emotion_dom = df['emotion_dominant'].fillna(0).astype(int).values
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
    return train_val_idx[sub_train_idx], train_val_idx[sub_val_idx], test_idx

# ==================== Dataset ====================
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME) if MODE in ['text', 'multimodal'] else None

class UnifiedDataset(Dataset):
    def __init__(self, lyrics, audio, emo_m, theme_m, emo_d, theme_d, mode):
        self.lyrics = lyrics
        self.audio = audio
        self.emo_m = emo_m
        self.theme_m = theme_m
        self.emo_d = emo_d
        self.theme_d = theme_d
        self.mode = mode

    def __len__(self):
        return len(self.emo_m)

    def __getitem__(self, item):
        out = {
            'emo_labels': torch.tensor(self.emo_m[item], dtype=torch.float32),
            'theme_labels': torch.tensor(self.theme_m[item], dtype=torch.float32),
            'emo_dom': torch.tensor(self.emo_d[item], dtype=torch.long),
            'theme_dom': torch.tensor(self.theme_d[item], dtype=torch.long),
        }
        if self.mode in ['text', 'multimodal']:
            enc = tokenizer(str(self.lyrics[item]), add_special_tokens=True,
                            max_length=MAX_LEN, padding='max_length', truncation=True,
                            return_attention_mask=True, return_tensors='pt')
            out['input_ids'] = enc['input_ids'].flatten()
            out['attention_mask'] = enc['attention_mask'].flatten()
        if self.mode in ['audio', 'multimodal']:
            out['audio'] = torch.tensor(self.audio[item], dtype=torch.float32)
        return out

# ==================== 跨模态注意力 ====================
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

# ==================== 统一模型 ====================
class UnifiedModel(nn.Module):
    def __init__(self, mode='text', audio_dim=768, proj_dim=128, hidden_dim=256,
                 dropout=0.3):
        super().__init__()
        self.mode = mode

        if mode in ['text', 'multimodal']:
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
            text_dim = proj_dim
        else:
            text_dim = 0

        if mode in ['audio', 'multimodal']:
            self.audio_encoder = nn.Sequential(
                nn.Linear(audio_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, proj_dim),
                nn.BatchNorm1d(proj_dim), nn.ReLU(), nn.Dropout(dropout)
            )
            audio_dim_out = proj_dim
        else:
            audio_dim_out = 0

        if mode == 'multimodal':
            self.cross_modal = CrossModalTransformer(proj_dim, num_heads=4, dropout=0.1)
            fusion_dim = 2 * proj_dim
        else:
            fusion_dim = text_dim + audio_dim_out

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
        self.emo_multi_head = nn.Linear(128, NUM_EMOTION)
        self.theme_multi_head = nn.Linear(128, NUM_THEME)
        self.emo_dom_head = nn.Linear(128, NUM_EMOTION)
        self.theme_dom_head = nn.Linear(128, NUM_THEME)

    def forward(self, input_ids=None, attention_mask=None, audio=None):
        if self.mode == 'text':
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            text_feat = self.text_proj(bert_out.last_hidden_state[:, 0, :])
            fused = text_feat
        elif self.mode == 'audio':
            fused = self.audio_encoder(audio)
        else:  # multimodal
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            text_feat = self.text_proj(bert_out.last_hidden_state[:, 0, :])
            audio_feat = self.audio_encoder(audio)
            audio_feat, text_feat = self.cross_modal(audio_feat, text_feat)
            fused = torch.cat([audio_feat, text_feat], dim=1)

        shared = self.fusion_encoder(fused)
        emo_feat = self.emotion_branch(shared)
        theme_feat = self.theme_branch(shared)

        return (self.emo_multi_head(emo_feat), self.theme_multi_head(theme_feat),
                self.emo_dom_head(emo_feat), self.theme_dom_head(theme_feat))

# ==================== 评估 ====================
def evaluate(model, loader):
    model.eval()
    emo_mp, emo_mt = [], []
    theme_mp, theme_mt = [], []
    emo_dp, emo_dt = [], []
    theme_dp, theme_dt = [], []
    with torch.no_grad():
        for batch in loader:
            kwargs = {}
            if MODE in ['text', 'multimodal']:
                kwargs['input_ids'] = batch['input_ids'].to(DEVICE)
                kwargs['attention_mask'] = batch['attention_mask'].to(DEVICE)
            if MODE in ['audio', 'multimodal']:
                kwargs['audio'] = batch['audio'].to(DEVICE)
            emo_ml, theme_ml, emo_dl, theme_dl = model(**kwargs)
            emo_mp.append(torch.sigmoid(emo_ml).cpu().numpy())
            emo_mt.append(batch['emo_labels'].numpy())
            theme_mp.append(torch.sigmoid(theme_ml).cpu().numpy())
            theme_mt.append(batch['theme_labels'].numpy())
            emo_dp.append(torch.argmax(emo_dl, 1).cpu().numpy())
            emo_dt.append(batch['emo_dom'].numpy())
            theme_dp.append(torch.argmax(theme_dl, 1).cpu().numpy())
            theme_dt.append(batch['theme_dom'].numpy())
    return (np.vstack(emo_mp), np.vstack(emo_mt),
            np.vstack(theme_mp), np.vstack(theme_mt),
            np.concatenate(emo_dp), np.concatenate(emo_dt),
            np.concatenate(theme_dp), np.concatenate(theme_dt))

# ==================== 单次训练 ====================
def train_one(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"\n{'='*60}\nMODE={MODE} | 种子 {seed}\n{'='*60}")

    train_idx, val_idx, test_idx = stratified_split_three_way(
        df, test_size=0.15, val_size=0.15, random_state=seed
    )

    emo_multi = multi_labels[:, :4]
    theme_multi = multi_labels[:, 4:]

    if MODE in ['text', 'multimodal']:
        lyrics_train = df['lyrics_text'].values[train_idx].tolist()
        lyrics_val = df['lyrics_text'].values[val_idx].tolist()
        lyrics_test = df['lyrics_text'].values[test_idx].tolist()
    else:
        lyrics_train = lyrics_val = lyrics_test = [''] * len(train_idx)

    if MODE in ['audio', 'multimodal']:
        scaler = StandardScaler()
        audio_train = scaler.fit_transform(audio_feats[train_idx]).astype(np.float32)
        audio_val = scaler.transform(audio_feats[val_idx]).astype(np.float32)
        audio_test = scaler.transform(audio_feats[test_idx]).astype(np.float32)
    else:
        audio_train = audio_val = audio_test = np.zeros((len(train_idx), 768), dtype=np.float32)

    train_ds = UnifiedDataset(lyrics_train, audio_train, emo_multi[train_idx], theme_multi[train_idx],
                              emotion_dom[train_idx], theme_dom[train_idx], MODE)
    val_ds = UnifiedDataset(lyrics_val, audio_val, emo_multi[val_idx], theme_multi[val_idx],
                            emotion_dom[val_idx], theme_dom[val_idx], MODE)
    test_ds = UnifiedDataset(lyrics_test, audio_test, emo_multi[test_idx], theme_multi[test_idx],
                             emotion_dom[test_idx], theme_dom[test_idx], MODE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = UnifiedModel(mode=MODE).to(DEVICE)

    # 多标签损失
    emo_pos = emo_multi[train_idx].sum(0); emo_neg = len(train_idx) - emo_pos
    emo_pw = torch.clamp(torch.tensor(emo_neg/(emo_pos+1e-6), dtype=torch.float32).to(DEVICE), max=10.0)
    criterion_emo_multi = nn.BCEWithLogitsLoss(pos_weight=emo_pw)

    theme_pos = theme_multi[train_idx].sum(0); theme_neg = len(train_idx) - theme_pos
    theme_pw = torch.clamp(torch.tensor(theme_neg/(theme_pos+1e-6), dtype=torch.float32).to(DEVICE), max=10.0)
    criterion_theme_multi = nn.BCEWithLogitsLoss(pos_weight=theme_pw)

    # 主导损失
    emo_c = np.bincount(emotion_dom[train_idx], minlength=NUM_EMOTION); emo_w = 1.0/(emo_c+1e-6)
    emo_w = emo_w / emo_w.sum() * NUM_EMOTION
    criterion_emo_dom = nn.CrossEntropyLoss(
        weight=torch.tensor(emo_w, dtype=torch.float32).to(DEVICE), label_smoothing=0.1)

    theme_c = np.bincount(theme_dom[train_idx], minlength=NUM_THEME); theme_w = 1.0/(theme_c+1e-6)
    theme_w = theme_w / theme_w.sum() * NUM_THEME
    criterion_theme_dom = nn.CrossEntropyLoss(
        weight=torch.tensor(theme_w, dtype=torch.float32).to(DEVICE), label_smoothing=0.1)

    # 优化器：分参数组
    if MODE in ['text', 'multimodal']:
        bert_params = [p for p in model.bert.parameters() if p.requires_grad]
    else:
        bert_params = []
    other_params = [p for n, p in model.named_parameters() if not n.startswith('bert')]
    optimizer = optim.AdamW([
        {'params': bert_params, 'lr': LR_BERT},
        {'params': other_params, 'lr': LR_OTHER}
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val = 0.0; patience = 5; trigger = 0
    model_path = f"best_dom_{MODE}_seed{seed}.pth"

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            kwargs = {}
            if MODE in ['text', 'multimodal']:
                kwargs['input_ids'] = batch['input_ids'].to(DEVICE)
                kwargs['attention_mask'] = batch['attention_mask'].to(DEVICE)
            if MODE in ['audio', 'multimodal']:
                kwargs['audio'] = batch['audio'].to(DEVICE)
            emo_y = batch['emo_labels'].to(DEVICE)
            theme_y = batch['theme_labels'].to(DEVICE)
            emo_dy = batch['emo_dom'].to(DEVICE)
            theme_dy = batch['theme_dom'].to(DEVICE)

            optimizer.zero_grad()
            emo_ml, theme_ml, emo_dl, theme_dl = model(**kwargs)
            loss = (ALPHA * criterion_emo_multi(emo_ml, emo_y)
                    + BETA * criterion_theme_multi(theme_ml, theme_y)
                    + GAMMA * criterion_emo_dom(emo_dl, emo_dy)
                    + DELTA * criterion_theme_dom(theme_dl, theme_dy))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        emo_mp, emo_mt, theme_mp, theme_mt, emo_dp, emo_dt, theme_dp, theme_dt = evaluate(model, val_loader)
        emo_bin = (emo_mp > 0.3).astype(int); theme_bin = (theme_mp > 0.3).astype(int)
        emo_macro = f1_score(emo_mt, emo_bin, average='macro', zero_division=0)
        theme_macro = f1_score(theme_mt, theme_bin, average='macro', zero_division=0)
        emo_d_acc = accuracy_score(emo_dt, emo_dp)
        theme_d_acc = accuracy_score(theme_dt, theme_dp)
        val_combined = (emo_macro + theme_macro) / 2

        print(f"Epoch {epoch+1:02d} | Loss {total_loss/len(train_loader):.4f} | "
              f"Emo-F1 {emo_macro:.4f} | Theme-F1 {theme_macro:.4f} | "
              f"EmoDom {emo_d_acc:.4f} | ThemeDom {theme_d_acc:.4f}")

        if val_combined > best_val:
            best_val = val_combined
            torch.save(model.state_dict(), model_path)
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                print(f"[早停] Epoch {epoch+1}，最佳验证综合: {best_val:.4f}")
                break

    model.load_state_dict(torch.load(model_path))
    emo_mp, emo_mt, theme_mp, theme_mt, emo_dp, emo_dt, theme_dp, theme_dt = evaluate(model, test_loader)
    emo_bin = (emo_mp > 0.3).astype(int); theme_bin = (theme_mp > 0.3).astype(int)

    result = {
        'seed': seed,
        'mode': MODE,
        'emo_macro_f1': f1_score(emo_mt, emo_bin, average='macro', zero_division=0),
        'emo_micro_f1': f1_score(emo_mt, emo_bin, average='micro', zero_division=0),
        'theme_macro_f1': f1_score(theme_mt, theme_bin, average='macro', zero_division=0),
        'theme_micro_f1': f1_score(theme_mt, theme_bin, average='micro', zero_division=0),
        'emo_dom_acc': accuracy_score(emo_dt, emo_dp),
        'theme_dom_acc': accuracy_score(theme_dt, theme_dp),
    }
    print(f"测试集 | Emo-F1 {result['emo_macro_f1']:.4f} | Theme-F1 {result['theme_macro_f1']:.4f} | "
          f"EmoDom {result['emo_dom_acc']:.4f} | ThemeDom {result['theme_dom_acc']:.4f}")
    return result

# ==================== 主循环 ====================
if __name__ == '__main__':
    results = []
    for seed in SEEDS:
        results.append(train_one(seed))

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"最终结果（MODE={MODE}，450 首，10 种子）")
    print(f"{'='*60}")
    print(df_res)
    print("\n均值±标准差:")
    cols = ['emo_macro_f1','emo_micro_f1','theme_macro_f1','theme_micro_f1',
            'emo_dom_acc','theme_dom_acc']
    print(df_res[cols].agg(['mean','std']))
    print(f"\n结果已保存到 {OUTPUT_CSV}")