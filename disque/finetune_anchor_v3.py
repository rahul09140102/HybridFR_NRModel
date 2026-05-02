
# ------------------------------------------------------------
# Hybrid FR-NR IQA — UNSEEN / VAL SET Evaluation
# Matches the EXACT RNG call order and split logic from
# fine_tune_honest_8k.py
#
# KonIQ : 10073 total → shuffle → 8000 train / rest val
# LIVE-C: 1162 total  → 80% Fisher / 20% val (monitor only)
# TID   : 3000 total  → 80/20 split → val
# KADID : 10125 total → shuffle → 8000 train / rest val
# ------------------------------------------------------------

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
    get_val_transform
)
from models.hybrid_student import HybridStudent

# ----------------------------
# CONFIG — MUST MATCH fine_tune_honest_8k.py EXACTLY
# ----------------------------
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
SEED       = 42

STAGE1_KONIQ_SUBSET  = 2000
STAGE1_LIVEC_SUBSET  = 1162
STAGE1_KADID_SUBSET  = 2000
FINETUNE_KONIQ_TOTAL = 8000
FINETUNE_KADID_TOTAL = 8000

rng = np.random.default_rng(SEED)

# ----------------------------
# PATHS
# ----------------------------
ckpt_path          = "/kaggle/input/datasets/hello123567890/checkpointfinetune/best_anchor_v3.pth"
reiqa_quality_ckpt = "/kaggle/input/datasets/chunnuchirkut/reiqa-checkpoints/quality_aware_r50.pth"
reiqa_content_ckpt = "/kaggle/input/datasets/chunnuchirkut/reiqa-checkpoints/content_aware_r50.pth"

KONIQ_IMG_DIR    = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/koniq10k_512x384/512x384"
KONIQ_ANNO_PATH  = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/koniq10k_scores_and_distributions.csv"
LIVE_C_IMG_DIR   = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/ChallengeDB_release/ChallengeDB_release/Images"
LIVE_C_MOS_PATH  = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/ChallengeDB_release/ChallengeDB_release/Data/AllMOS_release.mat"
LIVE_C_IMG_PATH  = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/ChallengeDB_release/ChallengeDB_release/Data/AllImages_release.mat"
TID2013_ROOT     = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/tid2013"
KADID_IMAGE_DIRS = [
    "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/kadid10k_part1/images",
    "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/kadid10k_part2/images",
    "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/kadid10k_part3/images",
]
KADID_CSV_PATH   = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/kadid10k_part1/dmos.csv"

# ----------------------------
# LOAD MODEL
# ----------------------------
print(f"🔹 Loading checkpoint from {ckpt_path}")

student = HybridStudent(
    reiqa_quality_ckpt=reiqa_quality_ckpt,
    reiqa_content_ckpt=reiqa_content_ckpt,
).to(device)

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
student.load_state_dict(state, strict=False)
student.eval()
print("✅ Model loaded\n")

# ----------------------------
# BUILD VAL SETS
# Replicates RNG call order from fine_tune_honest_8k.py:
#   RNG Call 1: KonIQ  — shuffle all, take 8000 train, rest = val
#   RNG Call 2: LIVE-C — shuffle 1162, 20% = val
#   RNG Call 3: TID    — shuffle all, 20% = val
#   RNG Call 4: KADID  — shuffle all, take 8000 train, rest = val
# ----------------------------
print("📦 Building val sets (replicating fine_tune_honest_8k.py split)...")

loaders = {}

# ── RNG Call 1: KonIQ ──────────────────────────────────────
koniq_full = KonIQDataset(KONIQ_IMG_DIR, KONIQ_ANNO_PATH, transform=get_val_transform())
koniq_n    = len(koniq_full)

stage1_sub_k_all = np.arange(koniq_n)
rng.shuffle(stage1_sub_k_all)                           # ← RNG Call 1

stage1_koniq    = stage1_sub_k_all[:STAGE1_KONIQ_SUBSET].tolist()
extra_koniq_all = stage1_sub_k_all[STAGE1_KONIQ_SUBSET:].tolist()
extra_needed_k  = FINETUNE_KONIQ_TOTAL - STAGE1_KONIQ_SUBSET   # 6000

koniq_train_idx = stage1_koniq + extra_koniq_all[:extra_needed_k]
koniq_val_idx   = extra_koniq_all[extra_needed_k:]      # indices NOT used in training

print(f"  KonIQ  val : {len(koniq_val_idx)} images  (train was {len(koniq_train_idx)})")
loaders["KonIQ "] = DataLoader(
    Subset(koniq_full, koniq_val_idx),
    batch_size=batch_size, shuffle=False, num_workers=2
)

# ── RNG Call 2: LIVE-C ─────────────────────────────────────
livec_full = LIVEChallengeDataset(
    LIVE_C_IMG_DIR, LIVE_C_MOS_PATH, LIVE_C_IMG_PATH, transform=get_val_transform()
)
livec_n           = len(livec_full)
stage1_sub_lc_all = np.arange(livec_n)
rng.shuffle(stage1_sub_lc_all)                          # ← RNG Call 2

stage1_livec     = stage1_sub_lc_all[:STAGE1_LIVEC_SUBSET].tolist()
sp_lc            = int(len(stage1_livec) * 0.8)
livec_val_idx    = stage1_livec[sp_lc:]                 # 20% of the 1162 used in Stage 1

print(f"  LIVE-C val : {len(livec_val_idx)} images  (monitor, seen in Stage 1 training)")
loaders["LIVE-C"] = DataLoader(
    Subset(livec_full, livec_val_idx),
    batch_size=batch_size, shuffle=False, num_workers=2
)

# ── RNG Call 3: TID ────────────────────────────────────────
tid_full = TID2013Dataset(TID2013_ROOT, transform=get_val_transform())
all_t    = list(range(len(tid_full)))
rng.shuffle(all_t)                                      # ← RNG Call 3
sp_t     = int(len(all_t) * 0.8)
tid_val_idx = all_t[sp_t:]

print(f"  TID    val : {len(tid_val_idx)} images  (unseen from Stage 1 training)")
loaders["TID2013"] = DataLoader(
    Subset(tid_full, tid_val_idx),
    batch_size=batch_size, shuffle=False, num_workers=2
)

# ── RNG Call 4: KADID ──────────────────────────────────────
kadid_full = KADIDDataset(KADID_IMAGE_DIRS, KADID_CSV_PATH, transform=get_val_transform())
kadid_n    = len(kadid_full)

stage1_sub_kd_all = np.arange(kadid_n)
rng.shuffle(stage1_sub_kd_all)                          # ← RNG Call 4

stage1_kadid    = stage1_sub_kd_all[:STAGE1_KADID_SUBSET].tolist()
extra_kadid_all = stage1_sub_kd_all[STAGE1_KADID_SUBSET:].tolist()
extra_needed_kd = FINETUNE_KADID_TOTAL - STAGE1_KADID_SUBSET   # 6000

kadid_train_idx = stage1_kadid + extra_kadid_all[:extra_needed_kd]
kadid_val_idx   = extra_kadid_all[extra_needed_kd:]     # indices NOT used in training

print(f"  KADID  val : {len(kadid_val_idx)} images  (train was {len(kadid_train_idx)})")
loaders["KADID "] = DataLoader(
    Subset(kadid_full, kadid_val_idx),
    batch_size=batch_size, shuffle=False, num_workers=2
)

# ----------------------------
# ALSO LOAD INDICES FROM CHECKPOINT (sanity check)
# ----------------------------
if "koniq_val_idx" in ckpt:
    saved_k = set(ckpt["koniq_val_idx"])
    regen_k = set(koniq_val_idx)
    match   = "✅" if saved_k == regen_k else f"⚠️  MISMATCH ({len(saved_k - regen_k)} diff)"
    print(f"\n  KonIQ  idx sanity check : {match}")

if "kadid_val_idx" in ckpt:
    saved_kd = set(ckpt["kadid_val_idx"])
    regen_kd = set(kadid_val_idx)
    match    = "✅" if saved_kd == regen_kd else f"⚠️  MISMATCH ({len(saved_kd - regen_kd)} diff)"
    print(f"  KADID  idx sanity check : {match}")

if "tid_val_idx" in ckpt:
    saved_t = set(ckpt["tid_val_idx"])
    regen_t = set(tid_val_idx)
    match   = "✅" if saved_t == regen_t else f"⚠️  MISMATCH ({len(saved_t - regen_t)} diff)"
    print(f"  TID    idx sanity check : {match}")

if "livec_val_idx" in ckpt:
    saved_lc = set(ckpt["livec_val_idx"])
    regen_lc = set(livec_val_idx)
    match    = "✅" if saved_lc == regen_lc else f"⚠️  MISMATCH ({len(saved_lc - regen_lc)} diff)"
    print(f"  LIVE-C idx sanity check : {match}")

# ----------------------------
# EVALUATION
# ----------------------------
def evaluate(loader, name):
    preds, labels = [], []
    with torch.no_grad():
        for dist, mos, ref in tqdm(loader, desc=name):
            dist = dist.to(device)
            ref  = ref.to(device)
            _, score = student(dist, ref=ref)
            preds.extend(score.view(-1).cpu().numpy())
            labels.extend(mos.view(-1).numpy())

    if len(preds) < 2:
        print(f"  {name:30s}  ← skipped (insufficient samples)")
        return None, None

    srocc, _ = spearmanr(preds, labels)
    plcc,  _ = pearsonr(preds, labels)
    return float(srocc), float(plcc)

# ----------------------------
# RUN
# ----------------------------


print("\n" + "=" * 65)
print("  Results — VAL splits (held-out from fine-tuning)")
print("=" * 65)
print(f"  {'Dataset':<14} {'SROCC':>7}  {'PLCC':>7} ")
print("-" * 65)

results = {}
for name, loader in loaders.items():
    srocc, plcc = evaluate(loader, name)
    results[name] = (srocc, plcc)
    if srocc is not None:
        print(f"  {name:<14} {srocc:>7.4f}  {plcc:>7.4f}  ")

print("=" * 65)

# ----------------------------
# MEAN OF PRIMARY METRICS
# ----------------------------
k_s  = results.get("KonIQ ", (None,))[0]
kd_s = results.get("KADID ", (None,))[0]
if k_s and kd_s:
    print(f"\n  Mean KonIQ + KADID SROCC : {np.mean([k_s, kd_s]):.4f}")


# ----------------------------
# CHECKPOINT METADATA
# ----------------------------
if isinstance(ckpt, dict):
    print(f"\n  📋 Checkpoint metadata:")
    for key in ["epoch", "koniq_srocc", "kadid_srocc", "mean_ft_srocc", "ewc_lambda", "seed"]:
        if key in ckpt:
            print(f"     {key:<20}: {ckpt[key]}")
