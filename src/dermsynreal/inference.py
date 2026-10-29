"""Standalone, frozen-checkpoint inference for the four uploaded timm backbones.

Imports torch/timm only when called, to allow offline unit tests without a GPU.
DO NOT set pretrained=True: evaluation must load the user's actual trained weights.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from .evaluation import CLASSES, _label

MODEL_ALLOWLIST = {
    "resnet50.a1_in1k":224,
    "tf_efficientnet_b3.ns_jft_in1k":300,
    "convnext_tiny.fb_in22k_ft_in1k":224,
    "vit_small_patch14_dinov2.lvd142m":518,
}
TRAIN_ORDER = ("acne","fungal_infection","vitiligo","hyperpigmentation","skin_cancer","normal")


def _read_manifest(path, *, domain, other_ids=None, train_synthetic_manifest=None):
    """Input: image_id,image_path,label,split,group_id. Extra dataset_source and
    prompt_family_id for synthetic diagnostics. Synthetic train-family manifest
    must be supplied for any synthetic test.
    """
    domain = domain.lower()
    if domain not in {"real","synthetic"}:
        raise ValueError("domain must be real or synthetic")
    with Path(path).open(newline="",encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id","image_path","label","split","group_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing test manifest columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty {domain} test manifest")
    if domain == "synthetic" and not train_synthetic_manifest:
        raise ValueError("Synthetic-test inference needs a trained synthetic-family manifest for leakage checks")
    used_ids, seen = set(other_ids or []), set()
    for r in rows:
        iid = (r.get("image_id") or "").strip()
        if not iid or iid in seen or iid in used_ids:
            raise ValueError(f"Missing/colliding test image_id {iid!r}")
        seen.add(iid)
        if r["split"].strip().lower() != "test":
            raise ValueError(f"{domain} evaluation only accepts split=test")
        if not r["group_id"].strip():
            raise ValueError(f"Missing group_id for {iid}")
        _label(r["label"])
        if domain == "synthetic" and not (r.get("prompt_family_id") or "").strip():
            raise ValueError(f"Synthetic test item requires prompt_family_id: {iid}")
    if domain == "synthetic":
        with Path(train_synthetic_manifest).open(newline="",encoding="utf-8-sig") as f:
            tr = list(csv.DictReader(f))
        if not tr:
            raise ValueError("Empty training synthetic family manifest")
        used = set()
        trained_ids = set()
        for row in tr:
            is_syn = str(row.get("is_synthetic","true")).lower() in {"true","1","yes"}
            if not is_syn:
                continue
            family = (row.get("prompt_family_id") or "").strip()
            group = (row.get("group_id") or "").strip()
            if not family and group.startswith("synthetic_"):
                family = group[len("synthetic_"):]
            if not family:
                raise ValueError("Synthetic training manifest lacks prompt_family_id/group_id family")
            used.add(family)
            if row.get("image_id"):
                trained_ids.add(row["image_id"])
        if not used:
            raise ValueError("No synthetic training family records for independence test")
        for row in rows:
            if row["prompt_family_id"] in used or row["image_id"] in trained_ids:
                raise ValueError(f"Training/test synthetic prompt family or image ID overlaps: {row['image_id']}")
    return rows


def _image_id_or_hash(r):
    return r["image_id"]


def run_timm_inference(*, model_id, run_config, weights_file, real_manifest,
                       real_image_root, out_csv, synthetic_manifest=None,
                       synthetic_image_root=None, train_synthetic_manifest=None,
                       batch_size=8, num_workers=2, allow_test=False):
    if not allow_test:
        raise ValueError("Final test locked; call with allow_test=True only after model/experiment choices are frozen")
    config = json.loads(Path(run_config).read_text(encoding="utf-8"))
    expected = MODEL_ALLOWLIST.get(model_id)
    if expected is None:
        raise ValueError(f"Unsupported existing notebook model_id {model_id}")
    if config.get("model_id") != model_id:
        raise ValueError("Trained run_config model_id does not match chosen inference backbone")
    if int(config.get("image_size",-1)) != expected:
        raise ValueError("Trained run_config has a different input image size")
    if [str(x).lower() for x in config.get("labels",[])] != list(TRAIN_ORDER):
        raise ValueError("Class order differs from the training checkpoint")
    import numpy as np
    import torch
    import timm
    from PIL import Image, ImageOps
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms as T
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real = _read_manifest(real_manifest,domain="real")
    parts = [(r, "real", Path(real_image_root)) for r in real]
    if synthetic_manifest:
        if not synthetic_image_root:
            raise ValueError("Synthetic image root is required with synthetic manifest")
        syn = _read_manifest(synthetic_manifest,domain="synthetic",
                             other_ids={r["image_id"] for r in real},
                             train_synthetic_manifest=train_synthetic_manifest)
        parts += [(r, "synthetic", Path(synthetic_image_root)) for r in syn]
    model = timm.create_model(model_id,pretrained=False,num_classes=6)
    state = torch.load(str(weights_file),map_location="cpu",weights_only=True)
    if not isinstance(state,dict):
        raise TypeError("Checkpoint must be a state dictionary of model tensors")
    model.load_state_dict(state,strict=True)
    model.to(device).eval()
    model_cfg = timm.data.resolve_model_data_config(model)
    mean,std = model_cfg["mean"],model_cfg["std"]
    class PadSquare:
        def __call__(self, im):
            im=ImageOps.exif_transpose(im).convert("RGB")
            w,h=im.size
            side=max(w,h)
            canvas=Image.new("RGB",(side,side),(128,128,128))
            canvas.paste(im,((side-w)//2,(side-h)//2))
            return canvas
    transform=T.Compose([PadSquare(),T.Resize((expected,expected)),
                         T.ToTensor(),T.Normalize(mean,std)])
    class TestImages(Dataset):
        def __len__(self): return len(parts)
        def __getitem__(self,i):
            record,domain,root=parts[i]
            rel=Path(record["image_path"])
            path=rel if rel.is_absolute() else root/rel
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                image.load()
                if min(image.size)<64:
                    raise ValueError(f"Image too small: {path}")
                rgb=ImageOps.exif_transpose(image).convert("RGB")
                pixels=np.asarray(rgb,dtype=np.uint8)
                if np.std(pixels)<2 or float(np.mean(pixels))<3:
                    raise ValueError(f"Near-blank/black image rejected: {path}")
                tensor=transform(rgb)
            return tensor,i
    loader=DataLoader(TestImages(),batch_size=batch_size,num_workers=num_workers,
                      shuffle=False,pin_memory=device.type=="cuda")
    probabilities=[None]*len(parts)
    with torch.inference_mode():
        for batch,positions in loader:
            scores=torch.softmax(model(batch.to(device)),dim=1).float().cpu().numpy()
            for idx,p in zip(positions.tolist(),scores):
                if not np.all(np.isfinite(p)) or abs(float(np.sum(p))-1)>0.002:
                    raise ValueError("Invalid probability distribution")
                probabilities[idx]=p
    columns=["image_id","true_label","pred_label","source","split","group_id",
             "dataset_source","prompt_family_id"]+["p_"+c for c in TRAIN_ORDER]
    rows=[]
    for i,(record,domain,root) in enumerate(parts):
        prob=probabilities[i]
        if prob is None: raise RuntimeError("Missing prediction")
        output={
            "image_id":record["image_id"],
            "true_label":_label(record["label"]),
            "pred_label":_label(TRAIN_ORDER[int(np.argmax(prob))]),
            "source":domain,"split":"test","group_id":record["group_id"],
            "dataset_source":record.get("dataset_source",""),
            "prompt_family_id":record.get("prompt_family_id","")
        }
        output.update({"p_"+c:float(prob[j]) for j,c in enumerate(TRAIN_ORDER)})
        rows.append(output)
    target=Path(out_csv);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("w",newline="",encoding="utf-8") as stream:
        w=csv.DictWriter(stream,fieldnames=columns)
        w.writeheader();w.writerows(rows)
    evidence={
        "model_id":model_id,"training_config":str(run_config),
        "training_config_sha256":hashlib.sha256(Path(run_config).read_bytes()).hexdigest(),
        "checkpoint_sha256":hashlib.sha256(Path(weights_file).read_bytes()).hexdigest(),
        "real_manifest_sha256":hashlib.sha256(Path(real_manifest).read_bytes()).hexdigest(),
        "synthetic_manifest_sha256":hashlib.sha256(Path(synthetic_manifest).read_bytes()).hexdigest() if synthetic_manifest else None,
        "inference_preprocessing":"RGB EXIF PadSquare(128) resize nearest training size, timm data config normalize",
        "device":str(device),"timm_version":getattr(timm,"__version__",None),
        "torch_version":torch.__version__,"prediction_rows":len(rows),
        "class_order":list(TRAIN_ORDER),"locked_final_test":True
    }
    target.with_suffix(".provenance.json").write_text(json.dumps(evidence,indent=2)+"\n")
    return evidence
