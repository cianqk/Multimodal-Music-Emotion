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
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
PMEMO_BASE = r"G:\pyyyyy\pmo\scc"
LABELS_CSV = os.path.join(PMEMO_BASE, "pmemo_labels.csv")
MERT_DIR = os.path.join(PMEMO_BASE, "pmemo_mert_embeddings")
LYRICS_DIR = os.path.join(PMEMO_BASE, "pmemo_lyrics_text")

C3_MODEL_DIR = r"G:\ppppp\KongZhi\.venv\shiyan"
C3_MODEL_TEMPLATE = "best_dom_multimodal_seed{seed}.pth"

MODEL_NAME = 'bert-base-chinese'
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 15
LR_BERT = 1e-5
LR_OTHER = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ★ 扩展到 10 个种子
SEEDS = [30, 48, 52, 66, 77, 89, 109, 111, 152, 167]

print(f"使用设备: {DEVICE}")

# ==================== 数据加载（只加载一次） ====================
df_all = pd.read_csv(LABELS_CSV)

def load_npy(path, dim=768):
    try:
        return np.load(path).astype(np.float32)
    except:
        return np.zeros(dim, dtype=np.float32)

def load_lyrics(mid):
    p = os.path.join(LYRICS_DIR, f"{mid}.txt")
    try:
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
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
print(f"有效样本: {len(df_all)}")

# ==================== Dataset ====================
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class PMEmoDataset(Dataset):
    def __init__(self, lyrics, audio, valence, arousal):
        self.lyrics = lyrics
        self.audio = audio
        self.valence = valence
        self.arousal = arousal

    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, item):
        enc = tokenizer(
            str(self.lyrics[item]), add_special_tokens=True,
            max_length=MAX_LEN, padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'audio': torch.tensor(self.audio[item], dtype=torch.float32),
            'valence': torch.tensor(self.valence[item], dtype=torch.long),
            'arousal': torch.tensor(self.arousal[item], dtype=torch.long),
        }

# ==================== 模型定义 ====================
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

class PMEmoModel(nn.Module):
    def __init__(self, audio_dim=768, proj_dim=128, hidden_dim=256, dropout=0.3):
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
        self.valence_branch = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout)
        )
        self.arousal_branch = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout)
        )
        self.valence_head = nn.Linear(128, 2)
        self.arousal_head = nn.Linear(128, 2)

    def forward(self, input_ids, attention_mask, audio):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = bert_out.last_hidden_state[:, 0, :]
        text_feat = self.text_proj(text_emb)
        audio_feat = self.audio_encoder(audio)
        audio_feat, text_feat = self.cross_modal(audio_feat, text_feat)
        fused = torch.cat([audio_feat, text_feat], dim=1)
        shared = self.fusion_encoder(fused)
        v_feat = self.valence_branch(shared)
        a_feat = self.arousal_branch(shared)
        return self.valence_head(v_feat), self.arousal_head(a_feat)

# ==================== 加载 C3 前端权重 ====================
def load_c3_frontend(model, c3_path):
    if not os.path.exists(c3_path):
        print(f"[警告] C3 权重不存在: {c3_path}，跳过加载")
        return 0
    state_dict = torch.load(c3_path, map_location=DEVICE)
    model_dict = model.state_dict()
    matched = {}
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            if k.startswith(('emo_multi_head', 'theme_multi_head', 'emo_dom_head', 'theme_dom_head',
                              'emo_head', 'theme_head')):
                continue
            matched[k] = v
    model_dict.update(matched)
    model.load_state_dict(model_dict)
    print(f"已加载 {len(matched)} 个参数")
    return len(matched)

# ==================== 评估函数 ====================
def evaluate(model, loader):
    model.eval()
    v_pred, v_true, a_pred, a_true = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(DEVICE)
            am = batch['attention_mask'].to(DEVICE)
            au = batch['audio'].to(DEVICE)
            v_logits, a_logits = model(ids, am, au)
            v_pred.append(torch.argmax(v_logits, 1).cpu().numpy())
            v_true.append(batch['valence'].numpy())
            a_pred.append(torch.argmax(a_logits, 1).cpu().numpy())
            a_true.append(batch['arousal'].numpy())
    v_pred = np.concatenate(v_pred); v_true = np.concatenate(v_true)
    a_pred = np.concatenate(a_pred); a_true = np.concatenate(a_true)
    return {
        'valence_acc': accuracy_score(v_true, v_pred),
        'valence_f1': f1_score(v_true, v_pred, average='macro', zero_division=0),
        'arousal_acc': accuracy_score(a_true, a_pred),
        'arousal_f1': f1_score(a_true, a_pred, average='macro', zero_division=0),
    }

# ==================== 单次实验（一个种子） ====================
def run_one_seed(SEED):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"\n{'='*60}")
    print(f"种子 {SEED}")
    print(f"{'='*60}")

    # 划分
    train_idx, test_idx = train_test_split(
        np.arange(len(df_all)), test_size=0.2, random_state=SEED,
        stratify=valence_all
    )
    print(f"训练集: {len(train_idx)}, 测试集: {len(test_idx)}")

    train_ds = PMEmoDataset(
        [lyrics_text_all[i] for i in train_idx],
        audio_feats_all[train_idx],
        valence_all[train_idx],
        arousal_all[train_idx]
    )
    test_ds = PMEmoDataset(
        [lyrics_text_all[i] for i in test_idx],
        audio_feats_all[test_idx],
        valence_all[test_idx],
        arousal_all[test_idx]
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    c3_path = os.path.join(C3_MODEL_DIR, C3_MODEL_TEMPLATE.format(seed=SEED))
    criterion = nn.CrossEntropyLoss()

    # ========== 实验 1：零样本迁移 ==========
    model = PMEmoModel().to(DEVICE)
    load_c3_frontend(model, c3_path)
    zero_shot = evaluate(model, test_loader)
    print(f"零样本: V-Acc {zero_shot['valence_acc']:.4f} V-F1 {zero_shot['valence_f1']:.4f} | "
          f"A-Acc {zero_shot['arousal_acc']:.4f} A-F1 {zero_shot['arousal_f1']:.4f}")

    # ========== 实验 2：微调迁移 ==========
    model_ft = PMEmoModel().to(DEVICE)
    load_c3_frontend(model_ft, c3_path)

    bert_params = [p for p in model_ft.bert.parameters() if p.requires_grad]
    other_params = [p for n, p in model_ft.named_parameters() if not n.startswith('bert')]
    optimizer = optim.AdamW([
        {'params': bert_params, 'lr': LR_BERT},
        {'params': other_params, 'lr': LR_OTHER}
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1 = 0
    best_state = None
    for epoch in range(EPOCHS):
        model_ft.train()
        for batch in train_loader:
            ids = batch['input_ids'].to(DEVICE)
            am = batch['attention_mask'].to(DEVICE)
            au = batch['audio'].to(DEVICE)
            v_y = batch['valence'].to(DEVICE)
            a_y = batch['arousal'].to(DEVICE)

            optimizer.zero_grad()
            v_logits, a_logits = model_ft(ids, am, au)
            loss = criterion(v_logits, v_y) + criterion(a_logits, a_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_ft.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        metrics = evaluate(model_ft, test_loader)
        avg_f1 = (metrics['valence_f1'] + metrics['arousal_f1']) / 2
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_state = {k: v.cpu().clone() for k, v in model_ft.state_dict().items()}

    model_ft.load_state_dict(best_state)
    finetune = evaluate(model_ft, test_loader)
    print(f"微调:   V-Acc {finetune['valence_acc']:.4f} V-F1 {finetune['valence_f1']:.4f} | "
          f"A-Acc {finetune['arousal_acc']:.4f} A-F1 {finetune['arousal_f1']:.4f}")

    # ========== 实验 3：域内基线 ==========
    model_scratch = PMEmoModel().to(DEVICE)  # 不加载 C3
    bert_params = [p for p in model_scratch.bert.parameters() if p.requires_grad]
    other_params = [p for n, p in model_scratch.named_parameters() if not n.startswith('bert')]
    optimizer = optim.AdamW([
        {'params': bert_params, 'lr': LR_BERT},
        {'params': other_params, 'lr': LR_OTHER}
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1 = 0
    best_state = None
    for epoch in range(EPOCHS):
        model_scratch.train()
        for batch in train_loader:
            ids = batch['input_ids'].to(DEVICE)
            am = batch['attention_mask'].to(DEVICE)
            au = batch['audio'].to(DEVICE)
            v_y = batch['valence'].to(DEVICE)
            a_y = batch['arousal'].to(DEVICE)

            optimizer.zero_grad()
            v_logits, a_logits = model_scratch(ids, am, au)
            loss = criterion(v_logits, v_y) + criterion(a_logits, a_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_scratch.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        metrics = evaluate(model_scratch, test_loader)
        avg_f1 = (metrics['valence_f1'] + metrics['arousal_f1']) / 2
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_state = {k: v.cpu().clone() for k, v in model_scratch.state_dict().items()}

    model_scratch.load_state_dict(best_state)
    scratch = evaluate(model_scratch, test_loader)
    print(f"域内:   V-Acc {scratch['valence_acc']:.4f} V-F1 {scratch['valence_f1']:.4f} | "
          f"A-Acc {scratch['arousal_acc']:.4f} A-F1 {scratch['arousal_f1']:.4f}")

    # 清理显存
    del model, model_ft, model_scratch
    torch.cuda.empty_cache()

    return {
        'seed': SEED,
        'zero_shot': zero_shot,
        'finetune': finetune,
        'scratch': scratch,
    }

# ==================== 主循环 ====================
if __name__ == '__main__':
    all_results = []
    for seed in SEEDS:
        r = run_one_seed(seed)
        all_results.append(r)

    # ========== 汇总 ==========
    print("\n" + "="*70)
    print("PMEmo 迁移实验汇总（10 种子）")
    print("="*70)

    rows = []
    for r in all_results:
        for method in ['zero_shot', 'finetune', 'scratch']:
            m = r[method]
            rows.append({
                'seed': r['seed'],
                'method': method,
                **m
            })
    result_df = pd.DataFrame(rows)
    result_df.to_csv('pmemo_transfer_10seeds.csv', index=False)

    # 分组统计
    print("\n按方法分组的均值±标准差:")
    metrics = ['valence_acc', 'valence_f1', 'arousal_acc', 'arousal_f1']
    summary = result_df.groupby('method')[metrics].agg(['mean', 'std'])
    print(summary)

    # 打印易读的表格
    print("\n" + "="*70)
    print(f"{'方法':<15} {'V-Acc':<15} {'V-F1':<15} {'A-Acc':<15} {'A-F1':<15}")
    print("-" * 70)
    for method in ['zero_shot', 'finetune', 'scratch']:
        sub = result_df[result_df['method'] == method]
        print(f"{method:<15} "
              f"{sub['valence_acc'].mean():.4f}±{sub['valence_acc'].std():.3f}  "
              f"{sub['valence_f1'].mean():.4f}±{sub['valence_f1'].std():.3f}  "
              f"{sub['arousal_acc'].mean():.4f}±{sub['arousal_acc'].std():.3f}  "
              f"{sub['arousal_f1'].mean():.4f}±{sub['arousal_f1'].std():.3f}")

    print(f"\n结果已保存到 pmemo_transfer_10seeds.csv")