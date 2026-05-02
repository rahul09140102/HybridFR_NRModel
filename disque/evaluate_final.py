
import pkgutil, importlib.machinery
if not hasattr(pkgutil, 'ImpImporter'):
    class _Stub: pass
    pkgutil.ImpImporter = _Stub
if not hasattr(importlib.machinery.FileFinder, 'find_module'):
    importlib.machinery.FileFinder.find_module = lambda self, name, path=None: None

import sys, os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

sys.path.insert(0, "/kaggle/working/disque/disque")

from datasets.dataset import (
    KonIQDataset,
    LIVEChallengeDataset,
    TID2013Dataset,
    KADIDDataset,
    get_val_transform,
)
from models.hybrid_student import HybridStudent


device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
SEED       = 42

KONIQ_SUBSET = 2000
LIVEC_SUBSET = 1162
KADID_SUBSET = 2000

rng = np.random.default_rng(SEED)

def subsample(n_total, subset_size):
    idx = np.arange(n_total)
    rng.shuffle(idx)
    return idx[:min(subset_size, n_total)].tolist()


ckpt_path          = "/kaggle/input/datasets/hello123567890/checkpoint12345/best_hybrid_student.pth"
reiqa_quality_ckpt = "/kaggle/input/datasets/chunnuchirkut/reiqa-checkpoints/quality_aware_r50.pth"
reiqa_content_ckpt = "/kaggle/input/datasets/chunnuchirkut/reiqa-checkpoints/content_aware_r50.pth"

KONIQ_IMG_DIR   = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/koniq10k_512x384/512x384"
KONIQ_ANNO_PATH = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/koniq10k_scores_and_distributions.csv"

LIVE_C_IMG_DIR  = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/ChallengeDB_release/ChallengeDB_release/Images"
LIVE_C_MOS_PATH = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/ChallengeDB_release/ChallengeDB_release/Data/AllMOS_release.mat"
LIVE_C_IMG_PATH = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/ChallengeDB_release/ChallengeDB_release/Data/AllImages_release.mat"

TID2013_ROOT = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/tid2013"

KADID_IMAGE_DIRS = [
    "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/kadid10k_part1/images",
    "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/kadid10k_part2/images",
    "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/kadid10k_part3/images",
]
KADID_CSV_PATH = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/kadid10k_part1/dmos.csv"


print(f"🔹 Loading checkpoint: {ckpt_path}")
student = HybridStudent(
    reiqa_quality_ckpt=reiqa_quality_ckpt,
    reiqa_content_ckpt=reiqa_content_ckpt,
).to(device)

ckpt = torch.load(ckpt_path, map_location=device)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    student.load_state_dict(ckpt["model_state_dict"], strict=False)
else:
    student.load_state_dict(ckpt, strict=False)
student.eval()
print("Model loaded\n")


print("Reproducing train_distill.py splits...")

koniq_full = KonIQDataset(KONIQ_IMG_DIR, KONIQ_ANNO_PATH, transform=get_val_transform())
sub = subsample(len(koniq_full), KONIQ_SUBSET)          
sp  = int(len(sub) * 0.8)                               
koniq_val_idx    = sub[sp:]                             
koniq_unseen_idx = [i for i in range(len(koniq_full)) if i not in set(sub)]  
print(f"  KonIQ  — val(train split): {len(koniq_val_idx)} | unseen: {len(koniq_unseen_idx)}")

livec_full = LIVEChallengeDataset(
    LIVE_C_IMG_DIR, LIVE_C_MOS_PATH, LIVE_C_IMG_PATH, transform=get_val_transform()
)
sub_lc = subsample(len(livec_full), LIVEC_SUBSET)       
sp_lc  = int(len(sub_lc) * 0.8)
livec_val_idx = sub_lc[sp_lc:]                       
print(f"  LIVE-C — val(train split): {len(livec_val_idx)} | unseen: 0 (all sampled)")

tid_full = TID2013Dataset(TID2013_ROOT, transform=get_val_transform())
all_t = list(range(len(tid_full)))
rng.shuffle(all_t)                                      # RNG call 3
sp_t = int(len(all_t) * 0.8)
tid_val_idx = all_t[sp_t:]                              # 600 — same as training val
print(f"  TID    — val(train split): {len(tid_val_idx)} | unseen: 0 (all 3000 used)")

kadid_full = KADIDDataset(KADID_IMAGE_DIRS, KADID_CSV_PATH, transform=get_val_transform())
sub_k = subsample(len(kadid_full), KADID_SUBSET)        # RNG call 4
sp_k  = int(len(sub_k) * 0.8)
kadid_val_idx    = sub_k[sp_k:]                         # 400 — same as training val
kadid_unseen_idx = [i for i in range(len(kadid_full)) if i not in set(sub_k)]  # ~8125
print(f"  KADID  — val(train split): {len(kadid_val_idx)} | unseen: {len(kadid_unseen_idx)}")


loaders = {
  
    "KonIQ": DataLoader(
        Subset(koniq_full, koniq_unseen_idx), batch_size=batch_size, shuffle=False
    ),
 
    "LIVE-C": DataLoader(
        Subset(livec_full, livec_val_idx), batch_size=batch_size, shuffle=False
    ),
 
    "TID": DataLoader(
        Subset(tid_full, tid_val_idx), batch_size=batch_size, shuffle=False
    ),

    "KADID": DataLoader(
        Subset(kadid_full, kadid_unseen_idx), batch_size=batch_size, shuffle=False
    ),
}


def evaluate(loader, name):
    preds, labels = [], []
    with torch.no_grad():
        for dist, mos, ref in tqdm(loader, desc=name, leave=False):
            dist = dist.to(device)
            ref  = ref.to(device)
            _, score = student(dist, ref=ref)
            preds.extend(score.view(-1).cpu().numpy())
            labels.extend(mos.view(-1).numpy())

    if len(preds) < 2:
        print(f"  {name}: skipped (too few samples)")
        return None, None

    srocc, _ = spearmanr(preds, labels)
    plcc,  _ = pearsonr(preds, labels)
    return float(srocc), float(plcc)

print("\n" + "=" * 65)
print("  Results")
print("=" * 65)
print(f"  {'Dataset':<30} {'SROCC':>8} {'PLCC':>8}  Notes")
print("  " + "-" * 61)

results = {}
for name, loader in loaders.items():
    s, p = evaluate(loader, name)
    results[name] = (s, p)
    
    if s is not None:
        print(f"  {name:<30} {s:>8.4f} {p:>8.4f}  ")

print("=" * 65)


