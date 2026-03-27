
import os
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
from scipy.io import loadmat


class SDRSemiGridDataset:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Not used in student distillation")

class HDRSemiGridDataset:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Not used in student distillation")



def get_train_transform():
    return transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop((352, 480)),
        transforms.ToTensor(),
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.ToTensor(),
    ])

def get_transform():
    return get_val_transform()


class NRDataset(data.Dataset):
    def __init__(self, img_dir, anno_path, dataset_name='koniq', transform=None):
        self.img_dir   = img_dir
        self.transform = transform or get_val_transform()

        if dataset_name.lower() == 'spaq':
            df = pd.read_excel(anno_path)
            self.image_names = df["Image name"].tolist()
            self.mos_values  = df["MOS"].tolist()
        elif dataset_name.lower() == 'koniq':
            df      = pd.read_csv(anno_path)
            mos_col = next((c for c in ["MOS", "mos", "mean_score"] if c in df.columns), None)
            if mos_col is None:
                raise KeyError(f"No MOS column in {anno_path}")
            name_col         = "image_name" if "image_name" in df.columns else "img_name"
            self.image_names = df[name_col].tolist()
            self.mos_values  = df[mos_col].tolist()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        distorted = Image.open(
            os.path.join(self.img_dir, self.image_names[idx])
        ).convert("RGB")
        distorted = self.transform(distorted)
        ref       = torch.zeros_like(distorted)        # NR sentinel
        mos       = torch.tensor(self.mos_values[idx] / 5.0, dtype=torch.float32)
        return distorted, mos, ref



class LIVEChallengeDataset(data.Dataset):
    def __init__(self, img_root, mos_mat_path, transform=None):
        self.transform = transform or get_val_transform()
        self.mos       = loadmat(mos_mat_path)["AllMOS_release"].squeeze()

        regular  = []
        training = []

        # Only look at files directly in img_root (not subfolders)
        for f in os.listdir(img_root):
            if f.lower().endswith(('.bmp', '.jpg', '.png')):
                fname = os.path.splitext(f)[0]
                try:
                    regular.append((int(fname), os.path.join(img_root, f)))
                except ValueError:
                    pass

        # Training images are in the trainingImages subfolder
        training_dir = os.path.join(img_root, 'trainingImages')
        if os.path.exists(training_dir):
            for f in os.listdir(training_dir):
                if f.lower().endswith(('.bmp', '.jpg', '.png')):
                    fname = os.path.splitext(f)[0]
                    tnum = int(fname[1:]) if fname[1:].isdigit() else 999
                    training.append((tnum, os.path.join(training_dir, f)))

        regular.sort(key=lambda x: x[0])
        training.sort(key=lambda x: x[0])

        self.image_paths = [p for _, p in regular] + [p for _, p in training]

        assert len(self.image_paths) == len(self.mos), \
            f"LIVE-C mismatch: images={len(self.image_paths)} MOS={len(self.mos)}"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        ref   = torch.zeros_like(image)
        mos   = torch.tensor(float(self.mos[idx]) / 100.0, dtype=torch.float32)
        return image, mos, ref


class LIVEIQADataset(data.Dataset):
    DIST_FOLDERS = ['fastfading', 'gblur', 'jpeg', 'jp2k', 'wn']

    def __init__(self, root, transform=None):
        self.transform = transform or get_val_transform()

        refnames_mat = loadmat(os.path.join(root, 'refnames_all.mat'))
        refnames_all = refnames_mat['refnames_all'].squeeze()

        dmos_mat = loadmat(os.path.join(root, 'dmos.mat'))
        dmos_all = dmos_mat['dmos'].squeeze()
        orgs     = dmos_mat['orgs'].squeeze()

        self.distorted_paths = []
        self.ref_paths       = []
        self.mos_values      = []

        global_idx = 0
        for folder in self.DIST_FOLDERS:
            dist_dir = os.path.join(root, folder)
            imgs     = sorted([f for f in os.listdir(dist_dir)
                                if f.lower().endswith('.bmp')])
            for img_name in imgs:
                if orgs[global_idx] == 1:
                    global_idx += 1
                    continue

                ref_cell = refnames_all[global_idx]
                ref_name = str(ref_cell.item() if hasattr(ref_cell, 'item')
                               else ref_cell).strip()

                dist_path = os.path.join(dist_dir, img_name)
                ref_path  = os.path.join(root, 'refimgs', ref_name)

                if os.path.exists(dist_path) and os.path.exists(ref_path):
                    self.distorted_paths.append(dist_path)
                    self.ref_paths.append(ref_path)
                    self.mos_values.append(float(dmos_all[global_idx]))

                global_idx += 1

        print(f"[LIVE-IQA] Loaded {len(self.distorted_paths)} FR pairs.")

    def __len__(self):
        return len(self.distorted_paths)

    def __getitem__(self, idx):
        distorted = self.transform(
            Image.open(self.distorted_paths[idx]).convert("RGB")
        )
        ref = self.transform(
            Image.open(self.ref_paths[idx]).convert("RGB")
        )
        # Invert DMOS — higher = better quality
        mos = torch.tensor(1.0 - self.mos_values[idx] / 100.0, dtype=torch.float32)
        return distorted, mos, ref



class TID2013Dataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform or get_val_transform()

        dist_dir = os.path.join(root, 'distorted_images')
        ref_dir  = os.path.join(root, 'reference_images')
        mos_file = os.path.join(root, 'mos_with_names.txt')

        self.distorted_paths = []
        self.ref_paths       = []
        self.mos_values      = []

        with open(mos_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts    = line.split()
                mos_val  = float(parts[0])
                img_name = parts[1]

                dist_path = os.path.join(dist_dir, img_name)

                ref_base = img_name.split('_')[0]
                ref_path = os.path.join(ref_dir, ref_base.upper() + '.BMP')
                if not os.path.exists(ref_path):
                    ref_path = os.path.join(ref_dir, ref_base + '.bmp')

                if os.path.exists(dist_path) and os.path.exists(ref_path):
                    self.distorted_paths.append(dist_path)
                    self.ref_paths.append(ref_path)
                    self.mos_values.append(mos_val)

        print(f"[TID2013] Loaded {len(self.distorted_paths)} FR pairs.")

    def __len__(self):
        return len(self.distorted_paths)

    def __getitem__(self, idx):
        distorted = self.transform(
            Image.open(self.distorted_paths[idx]).convert("RGB")
        )
        ref = self.transform(
            Image.open(self.ref_paths[idx]).convert("RGB")
        )
        mos = torch.tensor(self.mos_values[idx] / 9.0, dtype=torch.float32)
        return distorted, mos, ref
