#  MUST BE FIRST — before any other import 
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
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
import numpy as np
from argparse import Namespace
import torch.serialization

sys.path.insert(0, "/kaggle/working/disque/disque")

from disque_module import DisQUEModule
from models.hybrid_student import HybridStudent
from datasets.dataset import (
    NRDataset,
    LIVEChallengeDataset,
    LIVEIQADataset,
    TID2013Dataset,
    get_train_transform,
    get_val_transform,
)

device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size   = 16
epochs       = 35
lr           = 1e-4
weight_decay = 1e-5
SEED         = 42


torch.manual_seed(SEED)
np.random.seed(SEED)


koniq_img_dir   = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/koniq10k_512x384/512x384"
koniq_anno_path = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/koniq10k_scores_and_distributions.csv"
live_c_img_dir  = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/ChallengeDB_release/ChallengeDB_release/Images"
live_c_mos_path = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/ChallengeDB_release/ChallengeDB_release/Data/AllMOS_release.mat"
live_iqa_root   = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/LIVE_IAQ/LIVE_IAQ/databaserelease2"
tid2013_root    = "/kaggle/input/datasets/chunnuchirkut/myprojectdataset/tid2013"

teacher_ckpt       = "/kaggle/working/disque/DisQUE_Checkpoints/DisQUE_SDR.ckpt"
reiqa_quality_ckpt = "/kaggle/input/datasets/chunnuchirkut/reiqa-checkpoints/quality_aware_r50.pth"
reiqa_content_ckpt = "/kaggle/input/datasets/chunnuchirkut/reiqa-checkpoints/content_aware_r50.pth"

save_dir = "/kaggle/working/checkpoints"
os.makedirs(save_dir, exist_ok=True)


uploaded_ckpt = "/kaggle/input/datasets/hello123567890/hybridiqacheckpoints35epochs/last_checkpoint.pth"


print(f"Loading teacher from {teacher_ckpt}")
torch.serialization.add_safe_globals([Namespace])
teacher = DisQUEModule.load_from_checkpoint(
    teacher_ckpt, map_location=device
).to(device).eval()
for p in teacher.parameters():
    p.requires_grad = False
print("Teacher loaded and frozen")


student = HybridStudent(
    reiqa_quality_ckpt=reiqa_quality_ckpt,
    reiqa_content_ckpt=reiqa_content_ckpt,
).to(device)


print("Loading datasets...")
koniq_train   = NRDataset(koniq_img_dir, koniq_anno_path, "koniq",
                           transform=get_train_transform())
livec_train   = LIVEChallengeDataset(live_c_img_dir, live_c_mos_path,
                           transform=get_train_transform())
liveiqa_train = LIVEIQADataset(live_iqa_root,
                           transform=get_train_transform())
tid_train     = TID2013Dataset(tid2013_root,
                           transform=get_train_transform())

koniq_val     = NRDataset(koniq_img_dir, koniq_anno_path, "koniq",
                           transform=get_val_transform())
livec_val     = LIVEChallengeDataset(live_c_img_dir, live_c_mos_path,
                           transform=get_val_transform())
liveiqa_val   = LIVEIQADataset(live_iqa_root,
                           transform=get_val_transform())
tid_val       = TID2013Dataset(tid2013_root,
                           transform=get_val_transform())


combined_train = ConcatDataset([koniq_train, livec_train,
                                 liveiqa_train, tid_train])
combined_val   = ConcatDataset([koniq_val, livec_val,
                                 liveiqa_val, tid_val])

total_size = len(combined_train)
train_size = int(0.8 * total_size)
val_size   = total_size - train_size


generator = torch.Generator().manual_seed(SEED)
train_indices, val_indices = torch.utils.data.random_split(
    range(total_size), [train_size, val_size],
    generator=generator
)

from torch.utils.data import Subset
train_dataset = Subset(combined_train, train_indices.indices)
val_dataset   = Subset(combined_val,   val_indices.indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)

print(f"Train={len(train_dataset)}  Val={len(val_dataset)}")
print(f" KonIQ={len(koniq_train)} LIVE-C={len(livec_train)} "
      f"LIVE-IQA={len(liveiqa_train)} TID2013={len(tid_train)}")


mse_loss  = nn.MSELoss()
optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)
scaler    = GradScaler()


start_epoch = 1
best_srocc  = -1.0

local_ckpt = os.path.join(save_dir, "last_checkpoint.pth")

if os.path.exists(local_ckpt):
    load_path = local_ckpt
    print(f"Found local checkpoint → {local_ckpt}")
elif os.path.exists(uploaded_ckpt):
    load_path = uploaded_ckpt
    print(f" Found uploaded checkpoint → {uploaded_ckpt}")
else:
    load_path = ""
    print("No checkpoint found — starting fresh from epoch 1")

if load_path:
    ckpt = torch.load(load_path, map_location=device, weights_only=False)

    missing, unexpected = student.load_state_dict(
        ckpt["model_state_dict"], strict=False
    )
    if missing:
        print(f" New keys (random init): {missing}")
    if unexpected:
        print(f"  Old keys skipped: {unexpected}")

    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print("  Optimizer state restored")
    except ValueError:
        print("   Optimizer state skipped (architecture changed)")

    try:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    except Exception:
        pass

    try:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print("  Scheduler state restored")
    except Exception:
        pass

    start_epoch = ckpt["epoch"] + 1
    best_srocc  = ckpt["best_srocc"]
    print(f"Resuming from epoch {start_epoch}  (best SROCC so far: {best_srocc:.4f})")

if start_epoch > epochs:
    print(f" Checkpoint epoch ({start_epoch - 1}) >= target epochs ({epochs})")
    print(f"   Resetting to epoch 1 — training from scratch")
    start_epoch = 1
    best_srocc  = -1.0


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


for epoch in range(start_epoch, epochs + 1):
    student.train()
    total_loss = 0.0
    fr_count   = 0
    nr_count   = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

    for distorted, mos, ref in loop:
        distorted = distorted.to(device)
        mos       = mos.to(device).unsqueeze(1)
        ref       = ref.to(device)

        optimizer.zero_grad()

        is_fr    = ref.abs().sum(dim=[1, 2, 3]) > 0
        fr_count += is_fr.sum().item()
        nr_count += (~is_fr).sum().item()

        with torch.no_grad():
            _, _, _, teacher_embeds = teacher._get_predictions(
                (distorted, ref, distorted, ref)
            )
            teacher_feat = torch.mean(
                torch.stack(teacher_embeds, dim=0), dim=0
            ).detach()

        with autocast():
            student_feat, student_score = student(distorted, ref=ref)
            loss_feat = mse_loss(student_feat, teacher_feat)
            loss_mos  = mse_loss(student_score, mos)
            loss      = loss_feat + 0.5 * loss_mos

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}", fr=fr_count, nr=nr_count)

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch [{epoch}/{epochs}] Loss={avg_loss:.4f} "
          f"FR_samples={fr_count} NR_samples={nr_count}")


    student.eval()
    val_loss   = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for distorted, mos, ref in val_loader:
            distorted = distorted.to(device)
            mos       = mos.to(device).unsqueeze(1)
            ref       = ref.to(device)

            _, pred   = student(distorted, ref=ref)
            val_loss += mse_loss(pred, mos).item()
            all_preds.extend(pred.reshape(-1).cpu().numpy())
            all_labels.extend(mos.reshape(-1).cpu().numpy())

    avg_val  = val_loss / len(val_loader)
    srocc, _ = spearmanr(all_preds, all_labels)
    plcc,  _ = pearsonr(all_preds, all_labels)
    rmse     = np.sqrt(mean_squared_error(all_labels, all_preds))
    print(f"Val Loss={avg_val:.4f}  SROCC={srocc:.4f}  PLCC={plcc:.4f}  RMSE={rmse:.4f}")

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"LR updated to {current_lr:.6f}")

    if srocc > best_srocc:
        best_srocc = srocc
        best_path  = os.path.join(save_dir, "best_hybrid_student.pth")
        atomic_save(student.state_dict(), best_path)
        print(f"Best model saved → {best_path}  (SROCC={srocc:.4f})")

    last_path = os.path.join(save_dir, "last_checkpoint.pth")
    atomic_save({
        "epoch":                epoch,
        "model_state_dict":     student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict":    scaler.state_dict(),
        "best_srocc":           best_srocc,
    }, last_path)
    print(f"Checkpoint saved → {last_path}")

with open(os.path.join(save_dir, "final_metrics.txt"), "w") as f:
    f.write(f"Best SROCC: {best_srocc:.4f}\nFinal Epoch: {epoch}\n")
print(f"Metrics saved → {os.path.join(save_dir, 'final_metrics.txt')}")
print("\nTraining complete!")
