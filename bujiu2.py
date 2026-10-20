import os
#参数量匹配下的任务解耦分析
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
METADATA_PATH = r"G:\ppppp\KongZhi\.venv\music_dataset\metadata.csv"
MERT_DIR = r"G:\ppppp\KongZhi\music_dataset\mert_embeddings"
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

SEEDS = [30, 48, 52, 66, 77, 89, 109, 111, 152, 167]

# ==================== 自动定位歌词根目录 ====================
METADATA_PATH_ABS = os.path.abspath(METADATA_PATH)


def find_lyrics_base(csv_path):
    sample = pd.read_csv(csv_path, encoding='utf-8-sig', nrows=1)
    rel = str(sample['lyrics_path'].iloc[0]).replace('\\', '/')
    candidates = [os.path.dirname(csv_path), os.path.dirname(os.path.dirname(csv_path)), os.getcwd(),
                  os.path.dirname(os.getcwd())]
    for base in candidates:
        if os.path.exists(os.path.join(base, rel)): return base
    return None


LYRICS_BASE = find_lyrics_base(METADATA_PATH_ABS)


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


def read_lyrics(path):
    try:
        with open(os.path.join(LYRICS_BASE, str(path).replace('\\', '/')), 'r', encoding='utf-8', errors='ignore') as f:
            return f.read().strip()
    except:
        return ""


df['lyrics_text'] = df['lyrics_path'].apply(read_lyrics)
df = df[df['lyrics_text'] != ""].reset_index(drop=True)


def load_npy(path, dim=768):
    try:
        return np.load(path).astype(np.float32)
    except:
        return np.zeros(dim, dtype=np.float32)


audio_feats = np.array([load_npy(os.path.join(MERT_DIR, f"{row['id']}_mert.npy")) for _, row in df.iterrows()],
                       dtype=np.float32)
multi_labels = df[EMOTION_LABELS + THEME_LABELS].fillna(0).values.astype(np.float32)
emotion_dom = df['emotion_dominant'].fillna(0).astype(int).values
theme_dom = df['theme_dominant'].fillna(0).astype(int).values


# ==================== 三划分分层 ====================
def stratified_split_three_way(df, test_size=0.15, val_size=0.15, random_state=42):
    stratify_col = df['theme_dominant'].astype(str)
    counts = stratify_col.value_counts()
    stratify_col = stratify_col.apply(lambda x: 'rare' if x in counts[counts < 2].index else x)
    idx = np.arange(len(df))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_val_idx, test_idx in sss1.split(idx, stratify_col): pass
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=random_state)
    for sub_train_idx, sub_val_idx in sss2.split(train_val_idx, stratify_col.iloc[train_val_idx]): pass
    return train_val_idx[sub_train_idx], train_val_idx[sub_val_idx], test_idx


# ==================== Dataset ====================
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


class MultiModalDataset(Dataset):
    def __init__(self, lyrics, audio, emo_labels, theme_labels, emo_dom, theme_dom):
        self.lyrics, self.audio = lyrics, audio
        self.emo_labels, self.theme_labels = emo_labels, theme_labels
        self.emo_dom, self.theme_dom = emo_dom, theme_dom

    def __len__(self): return len(self.lyrics)

    def __getitem__(self, item):
        enc = tokenizer(str(self.lyrics[item]), max_length=MAX_LEN, padding='max_length', truncation=True,
                        return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].flatten(), 'attention_mask': enc['attention_mask'].flatten(),
            'audio': torch.tensor(self.audio[item], dtype=torch.float32),
            'emo_labels': torch.tensor(self.emo_labels[item], dtype=torch.float32),
            'theme_labels': torch.tensor(self.theme_labels[item], dtype=torch.float32),
            'emo_dom': torch.tensor(self.emo_dom[item], dtype=torch.long),
            'theme_dom': torch.tensor(self.theme_dom[item], dtype=torch.long),
        }


# ==================== 跨模态注意力 ====================
class CrossModalTransformer(nn.Module):
    def __init__(self, dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.modal_emb = nn.Parameter(torch.zeros(2, dim))
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim * 2, dropout=dropout,
                                           batch_first=True, activation='relu')
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, audio, text):
        tokens = torch.stack([audio, text], dim=1) + self.modal_emb.unsqueeze(0)
        encoded = self.norm(self.transformer(tokens))
        return encoded[:, 0, :], encoded[:, 1, :]


# ==================== 模型定义（包含参数匹配的共享架构） ====================
class E2ECapacityMatchedModel(nn.Module):
    def __init__(self, audio_dim=768, proj_dim=128, hidden_dim=256, mode='full'):
        super().__init__()
        self.mode = mode
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        for p in self.bert.parameters(): p.requires_grad = False
        for layer in self.bert.encoder.layer[-2:]:
            for p in layer.parameters(): p.requires_grad = True
        for p in self.bert.pooler.parameters(): p.requires_grad = True

        self.text_proj = nn.Sequential(nn.Linear(self.bert.config.hidden_size, proj_dim), nn.BatchNorm1d(proj_dim),
                                       nn.ReLU(), nn.Dropout(0.3))
        self.audio_encoder = nn.Sequential(nn.Linear(audio_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                           nn.Dropout(0.3),
                                           nn.Linear(hidden_dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.ReLU(),
                                           nn.Dropout(0.3))

        self.cross_modal = CrossModalTransformer(proj_dim, num_heads=4, dropout=0.1)
        self.fusion_encoder = nn.Sequential(nn.Linear(2 * proj_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                            nn.Dropout(0.3))

        if mode == 'full':
            # 解耦架构：两个分支 (256 -> 128)，参数量约为 6.5万
            self.emotion_branch = nn.Sequential(nn.Linear(hidden_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                                nn.Dropout(0.3))
            self.theme_branch = nn.Sequential(nn.Linear(hidden_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                              nn.Dropout(0.3))
            out_dim = 128
        else:  # 'shared_matched'
            # 参数匹配的共享架构：加一个大的共享层 (256 -> 256)，参数量约为 6.5万
            self.shared_capacity_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            out_dim = hidden_dim  # 256

        self.emo_multi_head = nn.Linear(out_dim, NUM_EMOTION)
        self.theme_multi_head = nn.Linear(out_dim, NUM_THEME)
        self.emo_dom_head = nn.Linear(out_dim, NUM_EMOTION)
        self.theme_dom_head = nn.Linear(out_dim, NUM_THEME)

    def forward(self, input_ids, attention_mask, audio):
        text_feat = self.text_proj(
            self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :])
        audio_feat = self.audio_encoder(audio)
        audio_feat, text_feat = self.cross_modal(audio_feat, text_feat)
        fused = torch.cat([audio_feat, text_feat], dim=1)
        shared = self.fusion_encoder(fused)

        if self.mode == 'full':
            emo_feat = self.emotion_branch(shared)
            theme_feat = self.theme_branch(shared)
        else:  # 'shared_matched'
            feat = self.shared_capacity_layer(shared)
            emo_feat = theme_feat = feat

        return self.emo_multi_head(emo_feat), self.theme_multi_head(theme_feat), self.emo_dom_head(
            emo_feat), self.theme_dom_head(theme_feat)


# ==================== 评估与训练 ====================
def evaluate(model, loader):
    model.eval()
    preds = {'em': [], 'em_t': [], 'tm': [], 'tm_t': [], 'ed': [], 'ed_t': [], 'td': [], 'td_t': []}
    with torch.no_grad():
        for batch in loader:
            em, tm, ed, td = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE),
                                   batch['audio'].to(DEVICE))
            preds['em'].append(torch.sigmoid(em).cpu().numpy());
            preds['em_t'].append(batch['emo_labels'].numpy())
            preds['tm'].append(torch.sigmoid(tm).cpu().numpy());
            preds['tm_t'].append(batch['theme_labels'].numpy())
            preds['ed'].append(torch.argmax(ed, 1).cpu().numpy());
            preds['ed_t'].append(batch['emo_dom'].numpy())
            preds['td'].append(torch.argmax(td, 1).cpu().numpy());
            preds['td_t'].append(batch['theme_dom'].numpy())

    em_b = (np.vstack(preds['em']) > 0.3).astype(int);
    tm_b = (np.vstack(preds['tm']) > 0.3).astype(int)
    return {
        'emo_macro_f1': f1_score(np.vstack(preds['em_t']), em_b, average='macro', zero_division=0),
        'theme_macro_f1': f1_score(np.vstack(preds['tm_t']), tm_b, average='macro', zero_division=0),
        'emo_dom_acc': accuracy_score(np.concatenate(preds['ed_t']), np.concatenate(preds['ed'])),
        'theme_dom_acc': accuracy_score(np.concatenate(preds['td_t']), np.concatenate(preds['td']))
    }


def train_one(seed, mode):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    print(f"\n>>> 种子 {seed} | 模式 {mode} <<<")
    train_idx, val_idx, test_idx = stratified_split_three_way(df, random_state=seed)

    scaler = StandardScaler()
    a_tr = scaler.fit_transform(audio_feats[train_idx]).astype(np.float32)
    a_va = scaler.transform(audio_feats[val_idx]).astype(np.float32)
    a_te = scaler.transform(audio_feats[test_idx]).astype(np.float32)

    emo_m = multi_labels[:, :4];
    theme_m = multi_labels[:, 4:]
    tr_ds = MultiModalDataset(df['lyrics_text'].values[train_idx], a_tr, emo_m[train_idx], theme_m[train_idx],
                              emotion_dom[train_idx], theme_dom[train_idx])
    va_ds = MultiModalDataset(df['lyrics_text'].values[val_idx], a_va, emo_m[val_idx], theme_m[val_idx],
                              emotion_dom[val_idx], theme_dom[val_idx])
    te_ds = MultiModalDataset(df['lyrics_text'].values[test_idx], a_te, emo_m[test_idx], theme_m[test_idx],
                              emotion_dom[test_idx], theme_dom[test_idx])

    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = E2ECapacityMatchedModel(mode=mode).to(DEVICE)

    # 打印参数量（供检查用）
    if seed == SEEDS[0]:
        total_params = sum(p.numel() for n, p in model.named_parameters() if not n.startswith('bert'))
        print(f"  [{mode}] 非 BERT 可训练参数量: {total_params:,}")

    # 损失权重计算 (修复 Double 报错)
    emo_pos = emo_m[train_idx].sum(0);
    theme_pos = theme_m[train_idx].sum(0)
    emo_pw = torch.tensor((len(train_idx) - emo_pos) / (emo_pos + 1e-6), dtype=torch.float32).to(DEVICE)
    theme_pw = torch.tensor((len(train_idx) - theme_pos) / (theme_pos + 1e-6), dtype=torch.float32).to(DEVICE)
    c_em = nn.BCEWithLogitsLoss(pos_weight=torch.clamp(emo_pw, max=10.0))
    c_tm = nn.BCEWithLogitsLoss(pos_weight=torch.clamp(theme_pw, max=10.0))

    emo_c = np.bincount(emotion_dom[train_idx], minlength=NUM_EMOTION)
    emo_w = 1.0 / (emo_c + 1e-6);
    emo_w = emo_w / emo_w.sum() * NUM_EMOTION
    c_ed = nn.CrossEntropyLoss(weight=torch.tensor(emo_w, dtype=torch.float32).to(DEVICE), label_smoothing=0.1)

    theme_c = np.bincount(theme_dom[train_idx], minlength=NUM_THEME)
    theme_w = 1.0 / (theme_c + 1e-6);
    theme_w = theme_w / theme_w.sum() * NUM_THEME
    c_td = nn.CrossEntropyLoss(weight=torch.tensor(theme_w, dtype=torch.float32).to(DEVICE), label_smoothing=0.1)

    optimizer = optim.AdamW([
        {'params': [p for p in model.bert.parameters() if p.requires_grad], 'lr': LR_BERT},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('bert')], 'lr': LR_OTHER}
    ], weight_decay=1e-4)

    best_f1, best_state = 0, None
    for epoch in range(EPOCHS):
        model.train()
        for batch in tr_ld:
            optimizer.zero_grad()
            em, tm, ed, td = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE),
                                   batch['audio'].to(DEVICE))
            loss = 1.0 * c_em(em, batch['emo_labels'].to(DEVICE)) + 1.0 * c_tm(tm, batch['theme_labels'].to(DEVICE)) + \
                   0.5 * c_ed(ed, batch['emo_dom'].to(DEVICE)) + 0.5 * c_td(td, batch['theme_dom'].to(DEVICE))
            loss.backward();
            optimizer.step()

        metrics = evaluate(model, va_ld)
        val_score = (metrics['emo_macro_f1'] + metrics['theme_macro_f1']) / 2
        if val_score > best_f1:
            best_f1 = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return {'seed': seed, 'mode': mode, **evaluate(model, te_ld)}


# ==================== 主循环 ====================
if __name__ == '__main__':
    # 跑完整解耦 (full) 和 参数匹配的共享 (shared_matched)
    # 注：如果 'full' 你之前已经跑过 c3_final_450.csv，这里可以直接只跑 shared_matched
    MODES_TO_RUN = ['full', 'shared_matched']

    results = []
    for mode in MODES_TO_RUN:
        for seed in SEEDS:
            results.append(train_one(seed, mode))

    df_res = pd.DataFrame(results)
    df_res.to_csv('ablation_capacity_matched.csv', index=False)

    print("\n" + "=" * 60)
    print("参数匹配消融实验结果（均值）：")
    print("=" * 60)
    print(df_res.groupby('mode')[['emo_macro_f1', 'theme_macro_f1', 'emo_dom_acc', 'theme_dom_acc']].mean())