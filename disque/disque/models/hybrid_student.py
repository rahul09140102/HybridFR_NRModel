
import torch
import torch.nn as nn
from torchvision import models
import os
 
 
class HybridStudent(nn.Module):
    def __init__(self,
                 reiqa_quality_ckpt="/kaggle/input/datasets/chunnuchirkut/reiqa-checkpoints/quality_aware_r50.pth",
                 reiqa_content_ckpt="//kaggle/input/datasets/chunnuchirkut/reiqa-checkpoints/content_aware_r50.pth"):
        super().__init__()
 

        contrique_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.contrique = nn.Sequential(*list(contrique_backbone.children())[:-1])
        self.contrique_feat_dim = 2048
        print("[INFO] Using ResNet50 backbone for CONTRIQUE branch.")
 
     
        self.reiqa_quality = self._build_reiqa_branch(reiqa_quality_ckpt, "quality")
        self.reiqa_content  = self._build_reiqa_branch(reiqa_content_ckpt, "content")
        self.reiqa_feat_dim = 2048
 
    
        self.nr_fusion = nn.Sequential(
            nn.Linear(6144, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
        )

        self.fr_fusion = nn.Sequential(
            nn.Linear(10240, 8192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(8192, 2048),
            nn.ReLU(inplace=True),
        )
 
        
        self.regressor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
 
    def _build_reiqa_branch(self, ckpt_path, name):
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model = nn.Sequential(*list(model.children())[:-1])
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                state_dict = {k.replace("module.", ""): v for k, v in ckpt.items()}
                model.load_state_dict(state_dict, strict=False)
                print(f"[REIQA-{name}] Loaded successfully.")
            except Exception as e:
                print(f"[WARN] Could not load {name} REIQA weights: {e}")
        else:
            print(f"[WARN] No {name} REIQA checkpoint. Using ImageNet weights.")
        return model
 
    def forward(self, x, ref=None):
        """
        Handles mixed FR+NR batches correctly.
        Each sample is routed independently based on whether its ref is real.
 
        Args:
            x   : (B, 3, H, W) — distorted image
            ref : (B, 3, H, W) — reference image or zeros sentinel
 
        Returns:
            fused_feat : (B, 2048)
            score      : (B, 1)
        """
        B = x.size(0)
 
       
        dist_feat = self.contrique(x).view(B, -1)        # (B, 2048)
        reiqa_q   = self.reiqa_quality(x).view(B, -1)    # (B, 2048)
        reiqa_c   = self.reiqa_content(x).view(B, -1)    # (B, 2048)
 
        
        if ref is None:
            fr_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
        else:
            
            fr_mask = ref.abs().sum(dim=[1, 2, 3]) > 0
 
        nr_mask = ~fr_mask
 
       
        fused_feat = torch.zeros(B, 2048, device=x.device, dtype=dist_feat.dtype)
 
     
        if nr_mask.any():
            nr_combined = torch.cat(
                [dist_feat[nr_mask], reiqa_q[nr_mask], reiqa_c[nr_mask]], dim=1
            )  # (n_nr, 6144)
            fused_feat[nr_mask] = self.nr_fusion(nr_combined)
 
       
        if fr_mask.any():
            ref_feat  = self.contrique(ref[fr_mask]).view(fr_mask.sum(), -1)
            diff_feat = dist_feat[fr_mask] - ref_feat
            fr_combined = torch.cat(
                [dist_feat[fr_mask], ref_feat, diff_feat,
                 reiqa_q[fr_mask], reiqa_c[fr_mask]], dim=1
            )  # (n_fr, 10240)
            fused_feat[fr_mask] = self.fr_fusion(fr_combined)
 
        score = self.regressor(fused_feat)   # (B, 1)
        return fused_feat, score
