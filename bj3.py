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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import BertTokenizer, BertModel

# ==================== 配置 ====================
PMEMO_BASE = r"G:\pyyyyy\pmo\scc"
LABELS_CSV = os.path.join(PMEMO_BASE, "pmemo_labels.csv")
MERT_DIR = os.path.join(PMEMO_BASE, "pmemo_mert_embeddings")
LYRICS_DIR = os.path.join(PMEMO_BASE, "pmemo_lyrics_text")

C3_MODEL_DIR = r"G:\ppppp\KongZhi\.venv\shiyan"
C3_MODEL_TEMPLATE = "best_dom_multimodal_seed{seed}.pth"

MODEL_NAME = 'bert-base-chinese'
MAX_LEN = 256
BATCH_SIZE = 32  # Linear Probe 可以大一点
EPOCHS = 30
LR_HEAD = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEEDS = [30, 48, 52, 66, 77]
RATIOS = [0.1, 0.3, 0.5, 0.8]  # 学习曲线使用的数据比例

# ==================== 数据加载 ====================
df_all = pd.read_csv(LABELS_CSV)


def load_npy(path, dim=768):
    try:
        
        return np.load(path).astype(np.float32)
    except:
        return np.zeros(dim, dtype=np.float32)


def load_lyrics(mid):
    try:
        with open(os.path.join(LYRICS_DIR, f"{mid}.txt"), 'r', encoding='utf-8', errors='ignore') as f:
            return f.read().strip()
    except:
        return ""


audio_feats_all, lyrics_text_all, valid_idx = [], [], []
for i, row in df_all.iterrows():
    mid = int(row['musicId'])
    a_path = os.path.join(MERT_DIR, f"{mid}_mert.npy")
    lrc = load_lyrics(mid)
    if os.path.exists(a_path) and len(lrc) > 5:
        audio_feats_all.append(load_npy(a_path))
        lyrics_text_all.append(lrc)
        valid_idx.append(i)

df_all = df_all.iloc[valid_idx].reset_index(drop=True)
audio_feats_all = np.array(audio_feats_all, dtype=np.float32)
valence_all = df_all['valence_bin'].values.astype(np.int64)
arousal_all = df_all['arousal_bin'].values.astype(np.int64)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


class PMEmoDataset(Dataset):
    def __init__(self, lyrics, audio, valence, arousal):
        self.lyrics, self.audio = lyrics, audio
        self.valence, self.arousal = valence, arousal

    def __len__(self): return len(self.lyrics)

    def __getitem__(self, item):
        enc = tokenizer(str(self.lyrics[item]), max_length=MAX_LEN, padding='max_length', truncation=True,
                        return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].flatten(), 'attention_mask': enc['attention_mask'].flatten(),
            'audio': torch.tensor(self.audio[item], dtype=torch.float32),
            'valence': torch.tensor(self.valence[item], dtype=torch.long),
            'arousal': torch.tensor(self.arousal[item], dtype=torch.long),
        }


# ==================== 模型（复用之前的） ====================
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


class PMEmoModel(nn.Module):
    def __init__(self, audio_dim=768, proj_dim=128, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.text_proj = nn.Sequential(nn.Linear(self.bert.config.hidden_size, proj_dim), nn.BatchNorm1d(proj_dim),
                                       nn.ReLU(), nn.Dropout(dropout))
        self.audio_encoder = nn.Sequential(nn.Linear(audio_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                           nn.Dropout(dropout),
                                           nn.Linear(hidden_dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.ReLU(),
                                           nn.Dropout(dropout))
        self.cross_modal = CrossModalTransformer(proj_dim, num_heads=4, dropout=0.1)
        self.fusion_encoder = nn.Sequential(nn.Linear(2 * proj_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                            nn.Dropout(dropout))
        self.valence_branch = nn.Sequential(nn.Linear(hidden_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                            nn.Dropout(dropout))
        self.arousal_branch = nn.Sequential(nn.Linear(hidden_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                            nn.Dropout(dropout))
        self.valence_head = nn.Linear(128, 2)
        self.arousal_head = nn.Linear(128, 2)

    def forward(self, input_ids, attention_mask, audio):
        text_emb = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        text_feat = self.text_proj(text_emb)
        audio_feat = self.audio_encoder(audio)
        audio_feat, text_feat = self.cross_modal(audio_feat, text_feat)
        shared = self.fusion_encoder(torch.cat([audio_feat, text_feat], dim=1))
        return self.valence_head(self.valence_branch(shared)), self.arousal_head(self.arousal_branch(shared))


def load_c3_frontend(model, c3_path):
    if not os.path.exists(c3_path): return
    state_dict = torch.load(c3_path, map_location=DEVICE)
    model_dict = model.state_dict()
    matched = {k: v for k, v in state_dict.items() if
               k in model_dict and model_dict[k].shape == v.shape and not k.startswith(('emo_', 'theme_'))}
    model_dict.update(matched)
    model.load_state_dict(model_dict)


# ==================== 评估与训练 ====================
def evaluate(model, loader):
    model.eval()
    vp, vt, ap, at = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            vl, al = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['audio'].to(DEVICE))
            vp.append(torch.argmax(vl, 1).cpu().numpy());
            vt.append(batch['valence'].numpy())
            ap.append(torch.argmax(al, 1).cpu().numpy());
            at.append(batch['arousal'].numpy())
    vp, vt, ap, at = np.concatenate(vp), np.concatenate(vt), np.concatenate(ap), np.concatenate(at)
    return (f1_score(vt, vp, average='macro', zero_division=0) + f1_score(at, ap, average='macro', zero_division=0)) / 2


def run_linear_probe(seed, ratio, use_pretrain):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)

    # 划分
    train_idx, test_idx = train_test_split(np.arange(len(df_all)), test_size=0.2, random_state=seed,
                                           stratify=valence_all)
    if ratio < 1.0:
        # 从训练集中抽取一部分
        sub_train_idx, _ = train_test_split(train_idx, train_size=ratio, random_state=seed,
                                            stratify=valence_all[train_idx])
    else:
        sub_train_idx = train_idx

    tr_ds = PMEmoDataset([lyrics_text_all[i] for i in sub_train_idx], audio_feats_all[sub_train_idx],
                         valence_all[sub_train_idx], arousal_all[sub_train_idx])
    te_ds = PMEmoDataset([lyrics_text_all[i] for i in test_idx], audio_feats_all[test_idx], valence_all[test_idx],
                         arousal_all[test_idx])
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = PMEmoModel().to(DEVICE)
    if use_pretrain:
        load_c3_frontend(model, os.path.join(C3_MODEL_DIR, C3_MODEL_TEMPLATE.format(seed=seed)))

    # ★ 核心：冻结所有特征提取层
    for p in model.parameters(): p.requires_grad = False
    # 仅解冻最后两个 Head
    for p in model.valence_head.parameters(): p.requires_grad = True
    for p in model.arousal_head.parameters(): p.requires_grad = True

    optimizer = optim.AdamW([{'params': model.valence_head.parameters()}, {'params': model.arousal_head.parameters()}],
                            lr=LR_HEAD, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    for epoch in range(EPOCHS):
        model.train()
        for batch in tr_ld:
            optimizer.zero_grad()
            vl, al = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['audio'].to(DEVICE))
            loss = criterion(vl, batch['valence'].to(DEVICE)) + criterion(al, batch['arousal'].to(DEVICE))
            loss.backward();
            optimizer.step()

        f1 = evaluate(model, te_ld)
        if f1 > best_f1: best_f1 = f1

    return best_f1


# ==================== 主循环 ====================
if __name__ == '__main__':
    results = []
    print("开始 Linear Probing 实验...")

    for ratio in RATIOS:
        for use_pretrain in [True, False]:
            f1s = []
            for seed in SEEDS:
                f1 = run_linear_probe(seed, ratio, use_pretrain)
                f1s.append(f1)
            mean_f1 = np.mean(f1s)
            label = "Pretrained" if use_pretrain else "Random"
            results.append({'ratio': ratio, 'type': label, 'f1': mean_f1})
            print(f"比例: {ratio * 100:>2.0f}% | 类型: {label:<10} | Macro-F1: {mean_f1:.4f}")

    df_res = pd.DataFrame(results)
    df_res.to_csv('pmemo_linear_probe_curve.csv', index=False)