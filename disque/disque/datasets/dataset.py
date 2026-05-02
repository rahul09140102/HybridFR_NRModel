
import os
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image


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
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def get_transform():
    return get_val_transform()


class KonIQDataset(data.Dataset):
    MOS_MAX = 5.0

    def __init__(self, img_dir, anno_path, transform=None):
        self.img_dir   = img_dir
        self.transform = transform or get_val_transform()

        df = pd.read_csv(anno_path)
        mos_col  = next((c for c in ["MOS", "mos", "mean_score"] if c in df.columns), None)
        name_col = next((c for c in ["image_name", "img_name", "filename"] if c in df.columns), None)
        if mos_col is None:
            raise KeyError(f"[KonIQ] No MOS column. Available: {df.columns.tolist()}")
        if name_col is None:
            raise KeyError(f"[KonIQ] No image-name column. Available: {df.columns.tolist()}")

        self.image_names = df[name_col].tolist()
        self.mos_values  = [float(v) / self.MOS_MAX for v in df[mos_col].tolist()]

        arr = np.array(self.mos_values)
        print(f"[KonIQ-10k] Loaded {len(self.image_names)} images.")
        print(f"  MOS raw=[{df[mos_col].min():.2f},{df[mos_col].max():.2f}]  "
              f"norm=[{arr.min():.3f},{arr.max():.3f}]  "
              f"mean={arr.mean():.3f}  std={arr.std():.3f}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        path      = os.path.join(self.img_dir, self.image_names[idx])
        distorted = self.transform(Image.open(path).convert("RGB"))
        ref       = torch.zeros_like(distorted)
        mos       = torch.tensor(self.mos_values[idx], dtype=torch.float32)
        return distorted, mos, ref

NRDataset = KonIQDataset

class LIVEChallengeDataset(data.Dataset):
    MOS_MAX = 100.0

    def __init__(self, img_dir, mos_mat_path, images_mat_path=None, transform=None):
        import scipy.io as sio

        self.img_dir   = img_dir
        self.transform = transform or get_val_transform()

        if os.path.isdir(mos_mat_path):
            data_dir    = mos_mat_path
            mos_file    = os.path.join(data_dir, "AllMOS_release.mat")
            images_file = os.path.join(data_dir, "AllImages_release.mat")
        elif os.path.isfile(mos_mat_path):
            mos_file    = mos_mat_path
            if images_mat_path and os.path.isfile(images_mat_path):
                images_file = images_mat_path
            else:
                images_file = os.path.join(os.path.dirname(mos_mat_path),
                                            "AllImages_release.mat")
        else:
            raise FileNotFoundError(f"[LIVE-C] mos_mat_path not found: {mos_mat_path}")

        if not os.path.isfile(mos_file):
            raise FileNotFoundError(f"[LIVE-C] AllMOS_release.mat not found: {mos_file}")
        if not os.path.isfile(images_file):
            raise FileNotFoundError(f"[LIVE-C] AllImages_release.mat not found: {images_file}")

        mos_mat    = sio.loadmat(mos_file)
        images_mat = sio.loadmat(images_file)

        mos_raw     = mos_mat["AllMOS_release"].flatten().astype(float)
        images_cell = images_mat["AllImages_release"]

        filenames = self._extract_filenames(images_cell)

        if len(filenames) != len(mos_raw):
            raise ValueError(
                f"[LIVE-C] Length mismatch after parsing: {len(filenames)} "
                f"images vs {len(mos_raw)} MOS values. "
                f"images_cell.shape={images_cell.shape}")

        pairs = []
        skipped_missing = []
        for fn, m in zip(filenames, mos_raw):
            full_path = os.path.join(img_dir, fn)
            if os.path.isfile(full_path):
                pairs.append((fn, m))
            else:
                skipped_missing.append(fn)

        if skipped_missing:
            print(f"[LIVE-Challenge] Skipped {len(skipped_missing)} entries not "
                  f"found in img_dir (training images in trainingImages/ subfolder): "
                  f"{skipped_missing[:7]}")

        if not pairs:
            raise ValueError(
                f"[LIVE-C] Zero images found in img_dir='{img_dir}'. "
                f"Check that LIVE_C_IMG_DIR points to the Images/ folder.")

        self.image_names = [p[0] for p in pairs]
        self.mos_values  = [p[1] / self.MOS_MAX for p in pairs]

        arr       = np.array(self.mos_values)
        n_dropped = len(filenames) - len(pairs)
        print(f"[LIVE-Challenge] Loaded {len(self.image_names)} images "
              f"(dropped {n_dropped} zero-MOS training images).")
        print(f"  MOS raw=[{min(p[1] for p in pairs):.2f},"
              f"{max(p[1] for p in pairs):.2f}]  "
              f"norm=[{arr.min():.3f},{arr.max():.3f}]  "
              f"mean={arr.mean():.3f}  std={arr.std():.3f}")

    @staticmethod
    def _extract_filenames(images_cell):
        """
        Robustly extract a flat list of filename strings from scipy's
        representation of a MATLAB cell array.

        Handles all observed layouts:
          Layout A — flat row:    shape (1, N)  — each [0, i]   is a filename
          Layout B — flat col:    shape (N, 1)  — each [i, 0]   is a filename
          Layout C — nested:      shape (1, 1)  — [0, 0] holds N filenames
        """
        def _to_str(elem):
            """Unwrap a scipy mat cell element to a plain Python string."""
            for _ in range(5):
                if hasattr(elem, 'dtype') and elem.dtype == object:
                    if elem.size == 1:
                        elem = elem.item()
                    else:
                        break
                else:
                    break
            if hasattr(elem, 'tolist'):
                elem = elem.tolist()
            if isinstance(elem, list):
                elem = elem[0] if elem else ""
            return str(elem).strip()

        rows, cols = images_cell.shape


        if rows == 1 and cols == 1:
            inner = images_cell[0, 0]
          
            if hasattr(inner, 'flatten'):
                inner = inner.flatten()
            return [_to_str(x) for x in inner]

        if rows == 1:
            return [_to_str(images_cell[0, i]) for i in range(cols)]

        if cols == 1:
            return [_to_str(images_cell[i, 0]) for i in range(rows)]

        return [_to_str(images_cell[i, j])
                for i in range(rows) for j in range(cols)]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        path      = os.path.join(self.img_dir, self.image_names[idx])
        distorted = self.transform(Image.open(path).convert("RGB"))
        ref       = torch.zeros_like(distorted)
        mos       = torch.tensor(self.mos_values[idx], dtype=torch.float32)
        return distorted, mos, ref


class TID2013Dataset(data.Dataset):
    MOS_MAX = 9.0

    def __init__(self, root, transform=None):
        self.dist_dir  = os.path.join(root, "distorted_images")
        self.ref_dir   = os.path.join(root, "reference_images")
        self.transform = transform or get_val_transform()

        mos_file = os.path.join(root, "mos_with_names.txt")
        if not os.path.isfile(mos_file):
            raise FileNotFoundError(f"[TID2013] mos_with_names.txt not found: {mos_file}")

        dist_cache = self._build_cache(self.dist_dir)
        ref_cache  = self._build_cache(self.ref_dir)

        if not dist_cache:
            raise RuntimeError(
                f"[TID2013] No images in distorted_images/: {self.dist_dir}")
        if not ref_cache:
            raise RuntimeError(
                f"[TID2013] No images in reference_images/: {self.ref_dir}")

        self.pairs = []
        skipped    = 0

        with open(mos_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue

                mos_raw   = float(parts[0])
                dist_name = parts[1]       

                ref_id   = dist_name.split("_")[0].lower()
                ref_name = ref_id + ".bmp"

                dist_path = dist_cache.get(dist_name.lower())
                ref_path  = ref_cache.get(ref_name.lower())

                if dist_path is None or ref_path is None:
                    skipped += 1
                    continue

                self.pairs.append((dist_path, ref_path, mos_raw / self.MOS_MAX))

        if skipped:
            print(f"[TID2013] Warning: skipped {skipped} entries (files not found).")

        if not self.pairs:
            sample_dist = list(dist_cache.keys())[:5]
            sample_ref  = list(ref_cache.keys())[:5]
            raise RuntimeError(
                f"[TID2013] Zero valid pairs loaded.\n"
                f"  dist_cache sample: {sample_dist}\n"
                f"  ref_cache  sample: {sample_ref}\n"
                f"  Check that distorted_images/ and reference_images/ "
                f"are the correct subdirectories under: {root}")

        arr = np.array([p[2] for p in self.pairs])
        print(f"[TID2013] Loaded {len(self.pairs)} FR pairs.")
        print(f"  MOS norm=[{arr.min():.3f},{arr.max():.3f}]  "
              f"mean={arr.mean():.3f}  std={arr.std():.3f}")

    @staticmethod
    def _build_cache(directory):
        """Return {lowercase_filename: full_path} for all image files in dir."""
        cache = {}
        if not os.path.isdir(directory):
            return cache
        for fname in os.listdir(directory):
            if fname.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
                cache[fname.lower()] = os.path.join(directory, fname)
        return cache

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        dist_path, ref_path, mos_norm = self.pairs[idx]
        distorted = self.transform(Image.open(dist_path).convert("RGB"))
        reference = self.transform(Image.open(ref_path).convert("RGB"))
        mos       = torch.tensor(mos_norm, dtype=torch.float32)
        return distorted, mos, reference



class KADIDDataset(data.Dataset):
    DMOS_MAX = 5.0

    def __init__(self, image_dirs, csv_path, transform=None):
        self.transform = transform or get_val_transform()

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"[KADID] dmos.csv not found: {csv_path}")

        df = pd.read_csv(csv_path)
        for col in ["dist_img", "ref_img", "dmos"]:
            if col not in df.columns:
                raise KeyError(f"[KADID] Missing column '{col}'. "
                               f"Available: {df.columns.tolist()}")

        self._path_cache = {}
        for d in image_dirs:
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self._path_cache[fname] = os.path.join(d, fname)

        if not self._path_cache:
            raise RuntimeError(f"[KADID] No images found in: {image_dirs}")

        self.pairs = []
        skipped    = 0
        for _, row in df.iterrows():
            dist_name = str(row["dist_img"]).strip()
            ref_name  = str(row["ref_img"]).strip()
            dmos_raw  = float(row["dmos"])
            dist_path = self._path_cache.get(dist_name)
            ref_path  = self._path_cache.get(ref_name)
            if dist_path is None or ref_path is None:
                skipped += 1
                continue
            self.pairs.append((dist_path, ref_path, dmos_raw / self.DMOS_MAX))

        if skipped:
            print(f"[KADID] Warning: skipped {skipped}/{len(df)} rows.")
        if not self.pairs:
            raise RuntimeError(
                f"[KADID] Zero valid pairs. "
                f"cache={len(self._path_cache)} images. Check image_dirs.")

        arr = np.array([p[2] for p in self.pairs])
        print(f"[KADID-10k] Loaded {len(self.pairs)} FR pairs "
              f"({skipped} skipped, {len(self._path_cache)} images indexed).")
        print(f"  DMOS norm=[{arr.min():.3f},{arr.max():.3f}]  "
              f"mean={arr.mean():.3f}  std={arr.std():.3f}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        dist_path, ref_path, dmos_norm = self.pairs[idx]
        distorted = self.transform(Image.open(dist_path).convert("RGB"))
        reference = self.transform(Image.open(ref_path).convert("RGB"))
        mos       = torch.tensor(dmos_norm, dtype=torch.float32)
        return distorted, mos, reference
