
import pkgutil, importlib.machinery

if not hasattr(pkgutil, 'ImpImporter'):
    class _Stub: pass
    pkgutil.ImpImporter = _Stub

if not hasattr(importlib.machinery.FileFinder, 'find_module'):
    importlib.machinery.FileFinder.find_module = lambda self, name, path=None: None


import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
import numpy as np
from argparse import Namespace
import torch.serialization

sys.path.insert(0, "/kaggle/working/disque/disque")

from disque_module import DisQUEModule
from models.hybrid_student import HybridStudent
from datasets.dataset import (
    KonIQDataset,
    LIVEChallengeDataset,
    TID2013Dataset,
    KADIDDataset,
    get_train_transform,
    get_val_transform,
)


device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size   = 6
epochs       = 35
lr           = 5e-5            
weight_decay = 1e-5
SEED         = 42
GRAD_CLIP    = 0.5             
WARMUP_EPOCHS = 3

KONIQ_SUBSET = 2000
LIVEC_SUBSET = 1162
KADID_SUBSET = 2000

torch.manual_seed(SEED)
np.random.seed(SEED)


KONIQ_IMG_DIR   = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/koniq10k_512x384/512x384"
KONIQ_ANNO_PATH = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/koniq10k_scores_and_distributions.csv"

LIVE_C_IMG_DIR  = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/ChallengeDB_release/ChallengeDB_release/Images"
LIVE_C_MOS_PATH = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/ChallengeDB_release/ChallengeDB_release/Data/AllMOS_release.mat"
LIVE_C_IMG_PATH = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/ChallengeDB_release/ChallengeDB_release/Data/AllImages_release.mat"

TID2013_ROOT    = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/tid2013"

KADID_IMAGE_DIRS = [
    "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/kadid10k_part1/images",
    "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/kadid10k_part2/images",
    "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/kadid10k_part3/images",
]
KADID_CSV_PATH  = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/kadid10k_part1/dmos.csv"

TEACHER_CKPT       = "/kaggle/working/disque/DisQUE_Checkpoints/DisQUE_SDR.ckpt"
REIQA_QUALITY_CKPT = "/kaggle/input/datasets/chunnuchirkut/reiqa-checkpoints/quality_aware_r50.pth"
REIQA_CONTENT_CKPT = "/kaggle/input/datasets/chunnuchirkut/reiqa-checkpoints/content_aware_r50.pth"

SAVE_DIR      = "/kaggle/working/checkpoints"
BEST_PATH     = os.path.join(SAVE_DIR, "best_hybrid_student.pth")
UPLOADED_CKPT = ""
os.makedirs(SAVE_DIR, exist_ok=True)


def safe_save(obj, path):
    tmp = path + ".tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
        return True
    except RuntimeError as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"Save failed: {e}")
        return False


print(f"🔹 Loading teacher from {TEACHER_CKPT}")
torch.serialization.add_safe_globals([Namespace])
teacher = DisQUEModule.load_from_checkpoint(
    TEACHER_CKPT, map_location=device
).to(device).eval()

for p in teacher.parameters():
    p.requires_grad = False

print("Teacher loaded and frozen")


student = HybridStudent(
    reiqa_quality_ckpt=REIQA_QUALITY_CKPT,
    reiqa_content_ckpt=REIQA_CONTENT_CKPT,
).to(device)


print("\nLoading datasets...")
rng = np.random.default_rng(SEED)

def subsample(n_total, subset_size):
    idx = np.arange(n_total)
    rng.shuffle(idx)
    return idx[:min(subset_size, n_total)].tolist()

# KonIQ
koniq_full = KonIQDataset(KONIQ_IMG_DIR, KONIQ_ANNO_PATH, transform=get_train_transform())
sub = subsample(len(koniq_full), KONIQ_SUBSET)
sp  = int(len(sub) * 0.8)
koniq_train_ds = Subset(koniq_full, sub[:sp])
koniq_val_ds   = Subset(KonIQDataset(KONIQ_IMG_DIR, KONIQ_ANNO_PATH,
                                    transform=get_val_transform()), sub[sp:])

# LIVE-C
livec_full = LIVEChallengeDataset(LIVE_C_IMG_DIR, LIVE_C_MOS_PATH,
                                 LIVE_C_IMG_PATH, transform=get_train_transform())
sub_lc = subsample(len(livec_full), LIVEC_SUBSET)
sp_lc  = int(len(sub_lc) * 0.8)
livec_train_ds = Subset(livec_full, sub_lc[:sp_lc])
livec_val_ds   = Subset(LIVEChallengeDataset(LIVE_C_IMG_DIR, LIVE_C_MOS_PATH,
                                            LIVE_C_IMG_PATH,
                                            transform=get_val_transform()), sub_lc[sp_lc:])

# TID
tid_full = TID2013Dataset(TID2013_ROOT, transform=get_train_transform())
all_t = list(range(len(tid_full)))
rng.shuffle(all_t)
sp_t = int(len(all_t) * 0.8)
tid_train_ds = Subset(tid_full, all_t[:sp_t])
tid_val_ds   = Subset(TID2013Dataset(TID2013_ROOT,
                                    transform=get_val_transform()), all_t[sp_t:])

# KADID
kadid_full = KADIDDataset(KADID_IMAGE_DIRS, KADID_CSV_PATH, transform=get_train_transform())
sub_k = subsample(len(kadid_full), KADID_SUBSET)
sp_k  = int(len(sub_k) * 0.8)
kadid_train_ds = Subset(kadid_full, sub_k[:sp_k])
kadid_val_ds   = Subset(KADIDDataset(KADID_IMAGE_DIRS, KADID_CSV_PATH,
                                    transform=get_val_transform()), sub_k[sp_k:])

train_ds = ConcatDataset([koniq_train_ds, livec_train_ds, tid_train_ds, kadid_train_ds])

train_loader = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True, num_workers=2, pin_memory=True)

val_loaders = {
    "KonIQ": DataLoader(koniq_val_ds, batch_size=batch_size),
    "LIVE-C": DataLoader(livec_val_ds, batch_size=batch_size),
    "TID2013": DataLoader(tid_val_ds, batch_size=batch_size),
    "KADID": DataLoader(kadid_val_ds, batch_size=batch_size),
}

criterion = nn.MSELoss()
optimiser = optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)

warmup_scheduler = optim.lr_scheduler.LambdaLR(
    optimiser, lr_lambda=lambda e: (e+1)/WARMUP_EPOCHS if e < WARMUP_EPOCHS else 1
)

cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimiser, T_max=(epochs - WARMUP_EPOCHS), eta_min=lr/100
)

scaler = GradScaler(enabled=torch.cuda.is_available())

best_srocc = -1.0

def evaluate_loader(loader):
    student.eval()
    preds, labels = [], []

    with torch.no_grad():
        for dist, mos, ref in loader:
            dist, mos, ref = dist.to(device), mos.to(device), ref.to(device)

            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                _, score = student(dist, ref=ref)

            preds.extend(score.view(-1).cpu().numpy())
            labels.extend(mos.view(-1).cpu().numpy())

    srocc, _ = spearmanr(preds, labels)
    plcc,  _ = pearsonr(preds, labels)

    return float(srocc), float(plcc)


print(f"\nTraining on {device}")

for epoch in range(epochs):
    student.train()
    total_loss = 0

    for dist, mos, ref in tqdm(train_loader):
        dist, mos, ref = dist.to(device), mos.to(device), ref.to(device)

        optimiser.zero_grad()

        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            student_feat, student_score = student(dist, ref=ref)

            student_score = student_score.view(-1)
            mos = mos.view(-1)

            with torch.no_grad():
                app = teacher.appearance_enc(dist)
                teacher_feat = torch.cat(app, dim=1)

            loss_mos = criterion(student_score, mos)

            if teacher_feat.shape == student_feat.shape:
                loss_feat = criterion(
                    nn.functional.normalize(student_feat, dim=1),
                    nn.functional.normalize(teacher_feat, dim=1)
                )
                loss = loss_mos + 0.5 * loss_feat
            else:
                loss = loss_mos

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)

        torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)

        scaler.step(optimiser)
        scaler.update()

        total_loss += loss.item()

    if epoch < WARMUP_EPOCHS:
        warmup_scheduler.step()
    else:
        cosine_scheduler.step()

    print(f"\nEpoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    sroccs = []
    for name, loader in val_loaders.items():
        srocc, plcc = evaluate_loader(loader)
        print(f"{name}: SROCC={srocc:.4f}, PLCC={plcc:.4f}")
        sroccs.append(srocc)

    mean_srocc = np.mean(sroccs)

    if mean_srocc > best_srocc:
        best_srocc = mean_srocc
        safe_save({
            "model_state_dict": student.state_dict(),
            "mean_srocc": mean_srocc
        }, BEST_PATH)

        print(f"Saved best model: {best_srocc:.4f}")

print(f"\nDone. Best SROCC: {best_srocc:.4f}")
