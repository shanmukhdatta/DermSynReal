"""Leakage-conscious, source-preserving real-image deduplication and group split.
Kaggle notebook embeds this code; no pip installation is necessary in standard Kaggle.
"""
from __future__ import annotations

import collections
import hashlib
import html
import json
import math
import os
import random
import re
import shutil
import warnings
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy.fft import dctn

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
CLASS_NAMES = ('Acne', 'Fungal_Infection', 'Vitiligo', 'Hyperpigmentation',
               'Skin_Cancer', 'Normal')
SPLITS = ('train', 'val', 'test')

@dataclass
class Settings:
    class_dirs: dict[str, str]
    output_dir: str = '/kaggle/working/facial_real_split'
    resnet_weights_path: str = ''
    allow_random_weights_for_smoke_test: bool = False
    seed: int = 42
    splits: tuple[float, float, float] = (0.70, 0.15, 0.15)
    phash_block_distance: int = 5
    dhash_h_search_distance: int = 4
    dhash_v_block_distance: int = 4
    dhash_embed_confirm_cosine: float = 0.95
    embedding_search_cosine: float = 0.90
    embedding_protect_cosine: float = 0.985
    embedding_top_k: int = 15
    embedding_batch_size: int = 32
    similarity_chunk_size: int = 128
    orb_verify: bool = True
    orb_candidates_limit: int = 1000
    orb_min_good_matches: int = 14
    orb_min_inliers: int = 10
    orb_min_inlier_ratio: float = 0.55
    max_protected_component_fraction: float = 0.10
    max_quarantine_fraction: float = 0.20
    split_trials: int = 24
    gallery_max_pairs: int = 250
    zip_output: bool = True

    def validate(self):
        if set(self.class_dirs) != set(CLASS_NAMES):
            raise ValueError(f'class_dirs must have exactly these six keys: {CLASS_NAMES}')
        for label, raw in self.class_dirs.items():
            if not raw or 'EDIT_' in raw:
                raise ValueError(f'Edit the Kaggle input directory for {label}: {raw!r}')
            path = Path(raw)
            if not path.is_dir():
                raise FileNotFoundError(f'Cannot find {label} directory: {path}. Attach dataset with Add Input and use /kaggle/input/... path.')
        if len(set(str(Path(v).resolve()) for v in self.class_dirs.values())) != 6:
            raise ValueError('Each class must have a separate directory; shared directory would duplicate/mislabel images.')
        for a, p in self.class_dirs.items():
            for b, q in self.class_dirs.items():
                if a != b and (Path(p).resolve() in Path(q).resolve().parents):
                    raise ValueError(f'Class directories must not contain one another: {a} and {b}')
        if not np.isclose(sum(self.splits), 1.0) or any(x <= 0 for x in self.splits):
            raise ValueError('splits must be three positive fractions summing to 1')
        if self.embedding_top_k < 1 or self.split_trials < 1:
            raise ValueError('embedding_top_k and split_trials must be >= 1')
        out = Path(self.output_dir).resolve()
        for p in self.class_dirs.values():
            if out == Path(p).resolve() or out in Path(p).resolve().parents:
                raise ValueError('output_dir must not contain an input directory')
            if Path(p).resolve() in out.parents:
                raise ValueError('output_dir must not be nested under a class input directory')


def to_json_ready(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, tuple): return list(obj)
    raise TypeError(f'JSON serialization: unsupported {type(obj)}')


def save_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2, default=to_json_ready), encoding='utf-8')


def normalized_image(im):
    # Read EXIF orientation before comparing pixel content or creating features.
    return ImageOps.exif_transpose(im).convert('RGB')


def image_hashes(im):
    # Four orientations; 180 rotation equals horizontal+vertical flipping.
    oriented = (im, im.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
                im.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
                im.transpose(Image.Transpose.ROTATE_180))
    ph, dh_h, dh_v = [], [], []
    for view in oriented:
        grey = view.convert('L')
        arr = np.asarray(grey.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
        freq = dctn(arr, type=2, norm='ortho')[:8, :8].copy()
        # Median ignores DC component, as in standard pHash; output remains 64 bits.
        comparison = freq > np.median(freq.flatten()[1:])
        bits = ''.join('1' if x else '0' for x in comparison.flatten())
        ph.append(int(bits, 2))
        # dHash horizontal compares neighboring columns; vertical compares rows.
        ah = np.asarray(grey.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
        av = np.asarray(grey.resize((8, 9), Image.Resampling.LANCZOS), dtype=np.int16)
        bh = ah[:, 1:] > ah[:, :-1]
        bv = av[1:, :] > av[:-1, :]
        dh_h.append(int(''.join('1' if x else '0' for x in bh.flatten()), 2))
        dh_v.append(int(''.join('1' if x else '0' for x in bv.flatten()), 2))
    return tuple(ph), tuple(dh_h), tuple(dh_v)


def scan_images(settings):
    items, invalid = [], []
    for label in CLASS_NAMES:
        folder = Path(settings.class_dirs[label]).resolve()
        paths = sorted((p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS), key=str)
        print(f'[{label}] found {len(paths)} image files')
        for path in paths:
            rel = path.relative_to(folder)
            try:
                sha = hashlib.sha256()
                with path.open('rb') as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b''):
                        sha.update(chunk)
                with Image.open(path) as raw:
                    im = normalized_image(raw)
                    im.load()
                    w, h = im.size
                    if w < 16 or h < 16:
                        raise ValueError(f'image too small: {w}x{h}')
                    pixel_digest = hashlib.sha256(f'{w}:{h}:RGB:'.encode() + im.tobytes()).hexdigest()
                    p, dh, dv = image_hashes(im)
                items.append({
                    'image_id': len(items), 'path': str(path), 'label': label,
                    'rel_path': rel.as_posix(), 'source_dataset': rel.parts[0] if len(rel.parts) > 1 else folder.name,
                    'sha256': sha.hexdigest(), 'pixel_hash': pixel_digest,
                    'width': w, 'height': h, 'bytes': path.stat().st_size,
                    'phash': p, 'dhash_h': dh, 'dhash_v': dv,
                })
            except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as ex:
                invalid.append({'path': str(path), 'label': label, 'reason': f'{type(ex).__name__}: {str(ex)[:300]}'})
    if not items:
        raise ValueError('No valid image files found in class directories.')
    print(f'Valid: {len(items)}; invalid: {len(invalid)}')
    return items, invalid


class BKTree:
    """Exact Hamming-radius candidate retrieval on 64-bit hashes."""
    def __init__(self): self.root = None

    def add(self, value, image_id):
        if self.root is None:
            self.root = [value, [image_id], {}]
            return
        node = self.root
        while True:
            dist = (value ^ node[0]).bit_count()
            if dist == 0:
                node[1].append(image_id)
                return
            child = node[2].get(dist)
            if child is None:
                node[2][dist] = [value, [image_id], {}]
                return
            node = child

    def search(self, value, radius):
        if self.root is None: return []
        hits = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            dist = (value ^ node[0]).bit_count()
            if dist <= radius: hits.extend(node[1])
            lo, hi = dist - radius, dist + radius
            stack.extend(child for d, child in node[2].items() if lo <= d <= hi)
        return hits


def pair_key(a, b): return (min(a, b), max(a, b))


def min_hash_distance(aa, bb):
    return min((a ^ b).bit_count() for a in aa for b in bb)


def min_dhash_distances(a, b):
    # Score h/v in the same orientation pair, not independent combinations.
    possible = []
    for i in range(4):
        for j in range(4):
            h = (a['dhash_h'][i] ^ b['dhash_h'][j]).bit_count()
            v = (a['dhash_v'][i] ^ b['dhash_v'][j]).bit_count()
            possible.append((h + v, h, v))
    _, h, v = min(possible)
    return h, v


def hash_candidates(items, settings):
    pairs = set()
    print('Retrieving pHash candidates with a Hamming BK-tree...')
    tree = BKTree()
    for image in items:
        ix = image['image_id']
        for h in image['phash']:
            for other in tree.search(h, settings.phash_block_distance):
                if ix != other: pairs.add(pair_key(ix, other))
        for h in image['phash']: tree.add(h, ix)
    print(f'pHash generated {len(pairs)} candidate pairs')
    # Only both-direction dHash evidence will produce a blocking link.
    print('Retrieving additional horizontal + vertical dHash candidates...')
    tree = BKTree()
    for image in items:
        ix = image['image_id']
        for h in image['dhash_h']:
            for other in tree.search(h, settings.dhash_h_search_distance):
                if ix == other: continue
                if min_dhash_distances(image, items[other])[1] <= settings.dhash_v_block_distance:
                    pairs.add(pair_key(ix, other))
        for h in image['dhash_h']: tree.add(h, ix)
    print(f'Hash candidate union: {len(pairs)} pairs')
    return pairs


def load_resnet(settings):
    import torch
    from torchvision.models import ResNet50_Weights, resnet50
    weights = ResNet50_Weights.IMAGENET1K_V2
    if settings.resnet_weights_path:
        path = Path(settings.resnet_weights_path)
        if not path.exists(): raise FileNotFoundError(f'Local pretrained ResNet-50 weights not found: {path}')
        # Load our own state_dict without downloading.
        model = resnet50(weights=None)
        state = torch.load(str(path), map_location='cpu', weights_only=True)
        if 'state_dict' in state: state = state['state_dict']
        model.load_state_dict(state, strict=True)
        source = f'local:{path}'
    else:
        try:
            model = resnet50(weights=weights)
            source = 'torchvision:ResNet50_Weights.IMAGENET1K_V2'
        except (OSError, RuntimeError, ConnectionError) as ex:
            if not settings.allow_random_weights_for_smoke_test:
                raise RuntimeError(
                    'Pretrained ResNet-50 weights unavailable. In Kaggle, enable Internet under '
                    'Notebook Settings, OR attach an ImageNet ResNet50 .pth weights file as Kaggle '
                    'Input and set RESNET_WEIGHTS_PATH. Do NOT use random weights for real splits.'
                ) from ex
            warnings.warn('SMOKE TEST ONLY: untrained ResNet-50 embeddings are NOT scientifically valid.', stacklevel=2)
            model = resnet50(weights=None)
            source = 'UNTRAINED_SMOKE_TEST_ONLY'
    model.fc = torch.nn.Identity()
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, weights.transforms(), device, source


def extract_embeddings(items, settings):
    import torch
    from torch.utils.data import Dataset, DataLoader
    model, preprocess, device, source = load_resnet(settings)

    class DatasetImages(Dataset):
        def __len__(self): return len(items)
        def __getitem__(self, ix):
            with Image.open(items[ix]['path']) as raw:
                img = normalized_image(raw)
                return preprocess(img), ix

    # Workers=0 avoids multiprocessing import/serialization issues in notebooks.
    loader = DataLoader(DatasetImages(), batch_size=settings.embedding_batch_size,
                        shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))
    features = np.zeros((len(items), 2048), dtype=np.float32)
    print(f'Extracting 2048-D ResNet-50 features on {device} ({source})...')
    with torch.inference_mode():
        for batch_idx, (imgs, ids) in enumerate(loader):
            predicted = model(imgs.to(device, non_blocking=True))
            array = predicted.float().cpu().numpy()
            array /= np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)
            features[ids.numpy()] = array
            if (batch_idx + 1) % 50 == 0: print(f'  embedded {min((batch_idx + 1) * settings.embedding_batch_size, len(items))}/{len(items)}')
    del model
    if device.type == 'cuda': torch.cuda.empty_cache()
    return features, source


def semantic_candidates(features, settings):
    # Exact chunked top-k cosine search; no FAISS, no extra pip install.
    n = len(features)
    k = min(n - 1, settings.embedding_top_k)
    pairs = set()
    if k <= 0: return pairs
    print(f'Retrieving ResNet top-{k} candidates, threshold={settings.embedding_search_cosine}...')
    for start in range(0, n, settings.similarity_chunk_size):
        end = min(n, start + settings.similarity_chunk_size)
        similarities = features[start:end] @ features.T
        similarities[np.arange(end - start), np.arange(start, end)] = -2.0
        for local, scores in enumerate(similarities):
            threshold_idxs = np.flatnonzero(scores >= settings.embedding_search_cosine)
            if len(threshold_idxs) > k:
                # Get top-k of matched entries, without a global sort.
                keep = np.argpartition(scores[threshold_idxs], -k)[-k:]
                threshold_idxs = threshold_idxs[keep]
            for j in threshold_idxs:
                pairs.add(pair_key(start + local, int(j)))
    print(f'ResNet generated {len(pairs)} candidate pairs')
    return pairs


def orb_geometric_match(path_a, path_b, settings):
    try:
        import cv2
        with Image.open(path_a) as raw: img_a = normalized_image(raw)
        with Image.open(path_b) as raw: img_b = normalized_image(raw)
        # Coarse correspondence is only a supporting duplicate signal.
        def gray_small(img):
            img.thumbnail((800, 800), Image.Resampling.LANCZOS)
            return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
        aa, bb = gray_small(img_a), gray_small(img_b)
        detector = cv2.ORB_create(nfeatures=1800)
        kp1, des1 = detector.detectAndCompute(aa, None)
        kp2, des2 = detector.detectAndCompute(bb, None)
        if des1 is None or des2 is None: return False, 0, 0
        match = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)
        good = [m for m, n in match if m.distance < 0.72 * n.distance]
        if len(good) < settings.orb_min_good_matches: return False, len(good), 0
        p1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        p2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
        inliers = int(mask.sum()) if matrix is not None and mask is not None else 0
        confirmed = (inliers >= settings.orb_min_inliers and
                     inliers / max(len(good), 1) >= settings.orb_min_inlier_ratio)
        return confirmed, len(good), inliers
    except Exception as ex:
        warnings.warn(f'ORB verification failed for {Path(path_a).name}: {ex}', stacklevel=2)
        return False, 0, 0


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n)); self.rank = [0] * n
    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b: return
        if self.rank[a] < self.rank[b]: a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]: self.rank[a] += 1


def verify_and_group(items, features, hash_pairs, embedding_pairs, settings):
    uf = UnionFind(len(items))
    edge_reasons = collections.Counter()
    exact_groups = collections.defaultdict(list)
    pixel_groups = collections.defaultdict(list)
    for img in items:
        exact_groups[img['sha256']].append(img['image_id'])
        pixel_groups[img['pixel_hash']].append(img['image_id'])
    for reason, groups in [('sha256', exact_groups), ('pixel_hash', pixel_groups)]:
        for ids in groups.values():
            if len(ids) > 1:
                for other in ids[1:]: uf.union(ids[0], other); edge_reasons[reason] += 1
    candidates = hash_pairs | embedding_pairs
    details = []
    print(f'Evaluating {len(candidates)} distinct hash/embedding candidate pairs...')
    # ORB is only attempted on the most relevant semantic candidates that did not already match pHash.
    orb_order = []
    for a, b in sorted(candidates):
        ia, ib = items[a], items[b]
        ph = min_hash_distance(ia['phash'], ib['phash'])
        dh, dv = min_dhash_distances(ia, ib)
        sim = float(np.dot(features[a], features[b]))
        same_exact = (ia['sha256'] == ib['sha256'] or ia['pixel_hash'] == ib['pixel_hash'])
        # 'protected' explicitly means same partition, NOT proven same patient/photo.
        reason = ''
        if same_exact: reason = 'exact_duplicate'
        elif ph <= settings.phash_block_distance: reason = 'phash_protection'
        elif dh <= settings.dhash_h_search_distance and dv <= settings.dhash_v_block_distance and sim >= settings.dhash_embed_confirm_cosine:
            reason = 'dhash_resnet_protection'
        elif sim >= settings.embedding_protect_cosine:
            reason = 'high_resnet_protection'
        if reason:
            uf.union(a, b)
            edge_reasons[reason] += 1
        item = dict(id_a=a, id_b=b, path_a=ia['path'], path_b=ib['path'],
                    class_a=ia['label'], class_b=ib['label'], phash_distance=ph,
                    dhash_horizontal_distance=dh, dhash_vertical_distance=dv,
                    cosine_similarity=round(sim, 6),
                    same_sha256=ia['sha256'] == ib['sha256'], same_pixel_hash=ia['pixel_hash'] == ib['pixel_hash'],
                    orb_good_matches=0, orb_inliers=0, orb_verified=False,
                    blocking_reason=reason, needs_human_review=not same_exact)
        details.append(item)
        if settings.orb_verify and not reason and sim >= 0.96:
            orb_order.append((sim, len(details) - 1))
    if settings.orb_verify:
        print(f'Running ORB geometric verification on up to {settings.orb_candidates_limit} uncertain candidates...')
        for _, index in sorted(orb_order, reverse=True)[:settings.orb_candidates_limit]:
            p = details[index]
            ok, matches, inliers = orb_geometric_match(p['path_a'], p['path_b'], settings)
            p.update(orb_good_matches=matches, orb_inliers=inliers, orb_verified=ok)
            if ok:
                p['blocking_reason'] = 'orb_geometric_protection'
                uf.union(p['id_a'], p['id_b'])
                edge_reasons['orb_geometric_protection'] += 1
    members = collections.defaultdict(list)
    for img in items: members[uf.find(img['image_id'])].append(img['image_id'])
    # IDs deterministic for fixed input list and matching configuration.
    groups = {f'G{ind:06d}': ids for ind, (_, ids) in enumerate(
        sorted(members.items(), key=lambda kv: min(kv[1])), 1)}
    group_of = {i: group for group, ids in groups.items() for i in ids}
    for p in details:
        p['group_a'] = group_of[p['id_a']]
        p['group_b'] = group_of[p['id_b']]
    return groups, group_of, details, dict(edge_reasons)


def select_representatives(items, groups):
    # Collisions across labels are quarantined. Keep only one source copy per exact pixel group.
    representatives = {}
    quarantined = []
    removals = []
    for group_id, ids in groups.items():
        grouped = collections.defaultdict(list)
        for i in ids:
            grouped[items[i]['pixel_hash']].append(i)
        all_labels = {items[i]['label'] for i in ids}
        if len(all_labels) > 1:
            # Multiple classes inside protected component need manual label adjudication.
            for i in ids:
                quarantined.append({'group_id': group_id, 'image_id': i, 'path': items[i]['path'],
                                    'label': items[i]['label'], 'reason': 'cross_class_similarity_group'})
            continue
        for _, matches in sorted(grouped.items()):
            # Higher resolution representative; deterministic tie-break on path.
            chosen = sorted(matches, key=lambda i: (-items[i]['width'] * items[i]['height'],
                                                   -items[i]['bytes'], items[i]['path']))[0]
            representatives[chosen] = group_id
            for other in matches:
                if other != chosen:
                    removals.append({'removed_image_id': other, 'removed_path': items[other]['path'],
                                     'retained_image_id': chosen, 'retained_path': items[chosen]['path'],
                                     'group_id': group_id, 'reason': 'same_normalized_pixels'})
    return representatives, quarantined, removals


def group_stratified_split(items, reps, settings):
    group_label = collections.defaultdict(list)
    for idx, group in reps.items(): group_label[group].append(idx)
    by_class = collections.defaultdict(list)
    for group, members in group_label.items():
        assert len({items[i]['label'] for i in members}) == 1
        by_class[items[members[0]]['label']].append((group, len(members)))
    allocations = {}
    report = []
    for class_num, label in enumerate(CLASS_NAMES):
        entries = sorted(by_class[label])
        if len(entries) < 3:
            raise ValueError(f'{label} has only {len(entries)} independent groups after quarantine; '
                             'need at least 3 for train/val/test. Inspect reports / adjust review decisions.')
        total = sum(size for _, size in entries)
        target = np.array(settings.splits) * total
        best = None
        for trial in range(settings.split_trials):
            rng = random.Random(settings.seed + class_num * 100000 + trial)
            shuffled = entries[:]
            rng.shuffle(shuffled)
            # Largest groups first; randomized order for tie-breaking/trial diversity.
            shuffled.sort(key=lambda x: -x[1])
            counts = np.zeros(3, dtype=np.float64)
            group_counts = np.zeros(3, dtype=int)
            decision = {}
            for ix, (group, size) in enumerate(shuffled):
                remaining = len(shuffled) - ix
                missing = np.flatnonzero(group_counts == 0)
                allowed = missing if len(missing) == remaining else np.arange(3)
                best_s = min(allowed, key=lambda s: (
                    # Prefer target fill, but assign at least one group per split.
                    ((counts + np.eye(3)[s] * size - target) ** 2 / np.maximum(target, 1)).sum(),
                    counts[s] / max(target[s], 1), s))
                counts[best_s] += size
                group_counts[best_s] += 1
                decision[group] = SPLITS[int(best_s)]
            if any(group_counts == 0): continue
            score = float((((counts - target) ** 2) / np.maximum(target, 1)).sum())
            if best is None or score < best[0]: best = (score, decision, counts.copy(), group_counts.copy())
        if best is None: raise ValueError(f'Split assignment failed for {label}')
        _, decisions, counts, group_counts = best
        allocations.update(decisions)
        for split_idx, split in enumerate(SPLITS):
            report.append({'class': label, 'split': split, 'images': int(counts[split_idx]),
                           'target_images': round(float(target[split_idx]), 1),
                           'groups': int(group_counts[split_idx]),
                           'fraction_within_class': round(float(counts[split_idx] / total), 5)})
    return allocations, pd.DataFrame(report)


def safe_name(img):
    stem = re.sub(r'[^a-zA-Z0-9_-]+', '_', Path(img['path']).stem).strip('_')[:36] or 'image'
    return f"{img['sha256'][:18]}_{img['image_id']:06d}_{stem}{Path(img['path']).suffix.lower()}"


def create_review_html(details, items, output_dir, limit):
    """Self-contained HTML preview embeds resized thumbnails, not source pixel changes."""
    import base64
    import io
    relevant = [p for p in details if p['needs_human_review']]
    relevant.sort(key=lambda p: (p['class_a'] != p['class_b'],
                                  bool(p['blocking_reason']), p['cosine_similarity']), reverse=True)
    chosen = relevant[:limit]
    def thumbnail(path):
        try:
            with Image.open(path) as raw:
                thumb = normalized_image(raw)
                thumb.thumbnail((240, 240), Image.Resampling.LANCZOS)
                dest = io.BytesIO()
                thumb.save(dest, format='JPEG', quality=78)
            return 'data:image/jpeg;base64,' + base64.b64encode(dest.getvalue()).decode('ascii')
        except OSError: return ''
    rows = ['<!doctype html><html><head><meta charset="utf-8"><title>Skin dataset match review</title>',
            '<style>body{font:14px Arial;margin:24px;background:#f6f7f9;color:#151515}table{width:100%;border-collapse:collapse}',
            'td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left;vertical-align:top}img{max-width:220px;max-height:220px}',
            'code{overflow-wrap:anywhere}small{color:#555}</style></head><body><h2>Candidate pair review gallery</h2>',
            '<p>Display only; original images are not resized in exported splits. Pairs are not necessarily duplicates.</p>',
            '<table><tr><th>A</th><th>B</th><th>Comparison</th></tr>']
    for p in chosen:
        x, y = thumbnail(p['path_a']), thumbnail(p['path_b'])
        rows.append('<tr><td><img src="' + x + '"><br><code>' + html.escape(p['path_a']) +
                    '</code></td><td><img src="' + y + '"><br><code>' + html.escape(p['path_b']) +
                    '</code></td><td>' + html.escape(p['class_a'] + ' ↔ ' + p['class_b']) +
                    f'<br>pHash={p["phash_distance"]}, dHashH/V={p["dhash_horizontal_distance"]}/{p["dhash_vertical_distance"]}' +
                    f'<br>ResNet cosine={p["cosine_similarity"]:.4f}' +
                    '<br>link=' + html.escape(p['blocking_reason'] or 'REVIEW ONLY') + '</td></tr>')
    rows.append('</table></body></html>')
    (output_dir / 'reports' / 'review_gallery.html').write_text('\n'.join(rows), encoding='utf-8')


def generate_leakage_audit(items, group_of, details, reps, allocations, settings):
    group_assignments = {group: allocations[group] for group in set(reps.values())}
    cross_hard = []
    cross_semantic = []
    for p in details:
        ga, gb = p['group_a'], p['group_b']
        if ga not in group_assignments or gb not in group_assignments: continue
        if group_assignments[ga] == group_assignments[gb]: continue
        if p['blocking_reason']:
            cross_hard.append(p)
        elif p['cosine_similarity'] >= settings.embedding_search_cosine:
            cross_semantic.append(p)
    return {
        'status': 'PASS_DETECTED_BLOCKING_EDGES' if not cross_hard else 'FAIL',
        'strong_cross_split_links': len(cross_hard),
        'unresolved_cross_split_semantic_candidates': len(cross_semantic),
        'limitations': [
            'No patient IDs: cannot establish patient independence.',
            'ResNet search is top-k and similarity thresholds are heuristic; unseen near-duplicates may remain.',
            'Semantic-only candidates are not confirmed duplicates; inspect review_candidates_cross_split.csv.',
            'This audit does not establish clinical accuracy or eliminate every possible leakage pathway.'
        ],
    }, cross_hard, cross_semantic


def zip_dataset(folder):
    result = folder.parent / (folder.name + '.zip')
    # Construct archive in temp name, then replace (prevents exposing half-built ZIP).
    temp = result.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp, mode='w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=4, allowZip64=True) as archive:
        for path in sorted(folder.rglob('*')):
            if path.is_file(): archive.write(path, path.relative_to(folder))
    temp.replace(result)
    return result


def run_pipeline(settings):
    settings.validate()
    out = Path(settings.output_dir).resolve()
    # Never silently mix files with a previous run's split.
    if out.exists():
        raise FileExistsError(f'{out} already exists; remove/rename previous output intentionally before rerunning.')
    (out / 'reports').mkdir(parents=True)
    save_json(out / 'reports' / 'config.json', asdict(settings))
    items, invalid = scan_images(settings)
    pd.DataFrame(invalid, columns=['path', 'label', 'reason']).to_csv(out / 'reports' / 'invalid_images.csv', index=False)
    overview = pd.DataFrame([{k:v for k,v in img.items() if k not in ('phash','dhash_h','dhash_v')} for img in items])
    overview.to_csv(out / 'reports' / 'inventory.csv', index=False)
    # For real datasets fail rather than run random-weight embedding without user's knowledge.
    features, weights_source = extract_embeddings(items, settings)
    hash_pairs = hash_candidates(items, settings)
    emb_pairs = semantic_candidates(features, settings)
    groups, group_of, details, edge_reasons = verify_and_group(items, features, hash_pairs, emb_pairs, settings)
    group_sizes = sorted((len(ids) for ids in groups.values()), reverse=True)
    max_allowed = max(30, math.ceil(len(items) * settings.max_protected_component_fraction))
    pd.DataFrame(details, columns=[
        'id_a','id_b','path_a','path_b','class_a','class_b','phash_distance',
        'dhash_horizontal_distance','dhash_vertical_distance','cosine_similarity',
        'same_sha256','same_pixel_hash','orb_good_matches','orb_inliers','orb_verified',
        'blocking_reason','needs_human_review','group_a','group_b']).to_csv(
        out / 'reports' / 'candidate_pairs.csv', index=False)
    create_review_html(details, items, out, settings.gallery_max_pairs)
    pd.DataFrame([{'image_id': img['image_id'], 'path': img['path'], 'label': img['label'],
                   'group_id': group_of[img['image_id']]} for img in items]).to_csv(
        out / 'reports' / 'all_group_assignments.csv', index=False)
    if group_sizes and group_sizes[0] > max_allowed:
        save_json(out / 'reports' / 'STOP_REASON.json', {'reason': 'suspected_giant_similarity_component',
                  'largest_group_size': group_sizes[0], 'allowed_maximum': max_allowed,
                  'explanation': 'Inspect candidate_pairs.csv and review_gallery.html; overly permissive hashes may merge unrelated photos.'})
        raise ValueError(f'Largest protected similarity group has {group_sizes[0]} of {len(items)} images '
                         f'(limit {max_allowed}). Outputs contain reports but no accepted split ZIP.')
    reps, quarantined, removals = select_representatives(items, groups)
    pd.DataFrame(quarantined, columns=['group_id','image_id','path','label','reason']).to_csv(out / 'reports' / 'quarantined_conflicts.csv', index=False)
    pd.DataFrame(removals, columns=['removed_image_id','removed_path','retained_image_id','retained_path','group_id','reason']).to_csv(
        out / 'reports' / 'removed_duplicates.csv', index=False)
    if len(quarantined) > len(items) * settings.max_quarantine_fraction:
        save_json(out / 'reports' / 'STOP_REASON.json', {'reason': 'excess_cross_label_quarantine',
                 'quarantined_count': len(quarantined), 'total': len(items),
                 'explanation': 'Too many conflicting candidate groups; manually review before finalizing.'})
        raise ValueError('Too many cross-label ambiguous images quarantined; inspect reports before running training.')
    try:
        allocations, counts = group_stratified_split(items, reps, settings)
    except Exception as ex:
        save_json(out / 'reports' / 'STOP_REASON.json', {'reason': 'split_failed', 'error': str(ex)})
        raise
    counts.to_csv(out / 'reports' / 'split_statistics.csv', index=False)
    audit, cross_hard, cross_semantic = generate_leakage_audit(items, group_of, details, reps, allocations, settings)
    pd.DataFrame(cross_semantic, columns=details[0].keys() if details else None).to_csv(
        out / 'reports' / 'review_candidates_cross_split.csv', index=False)
    save_json(out / 'reports' / 'leakage_audit.json', audit)
    if audit['status'] == 'FAIL':
        save_json(out / 'reports' / 'STOP_REASON.json', {'reason': 'blocking_pair_crosses_split', 'num': len(cross_hard)})
        raise AssertionError('Blocking similarity pair crosses train/val/test — refusing final export')
    records = []
    for idx, group in sorted(reps.items()):
        img = items[idx]
        split = allocations[group]
        name = safe_name(img)
        relative = Path(split) / img['label'] / name
        records.append({
            'image_id': idx, 'label': img['label'], 'split': split, 'group_id': group,
            'source_path': img['path'], 'relative_path': relative.as_posix(),
            'source_dataset': img['source_dataset'], 'sha256': img['sha256'],
            'pixel_hash': img['pixel_hash'], 'width': img['width'], 'height': img['height'],
        })
    manifest = pd.DataFrame(records)
    if not all(manifest['split'].eq(s).any() for s in SPLITS):
        raise AssertionError('Empty split created unexpectedly')
    for s in SPLITS:
        for name in CLASS_NAMES: (out / s / name).mkdir(parents=True, exist_ok=True)
    for r in records:
        dest = out / r['relative_path']
        if dest.exists(): raise FileExistsError(f'Filename collision: {dest}')
        shutil.copy2(r['source_path'], dest)
    manifest.to_csv(out / 'reports' / 'split_manifest.csv', index=False)
    for split in SPLITS:
        manifest[manifest['split'] == split].to_csv(out / 'reports' / f'{split}_manifest.csv', index=False)
    source_stats = manifest.groupby(['source_dataset','label','split']).size().rename('images').reset_index()
    source_stats.to_csv(out / 'reports' / 'source_distribution.csv', index=False)
    # Final independent checks on copied files, not just source manifests.
    if manifest['relative_path'].duplicated().any(): raise AssertionError('Duplicated export names')
    if (manifest.groupby('group_id')['split'].nunique() > 1).any(): raise AssertionError('Group leakage')
    if (manifest.groupby('pixel_hash')['split'].nunique() > 1).any(): raise AssertionError('Exact pixel hash leakage')
    if (manifest.groupby('sha256')['split'].nunique() > 1).any(): raise AssertionError('Exact SHA leakage')
    for split in SPLITS:
        if set(manifest.loc[manifest['split'] == split, 'label']) != set(CLASS_NAMES):
            raise AssertionError(f'One or more classes absent from {split}')
    sha_check_failures = []
    for r in records:
        dest = out / r['relative_path']
        if not dest.is_file() or hashlib.sha256(dest.read_bytes()).hexdigest() != r['sha256']:
            sha_check_failures.append(str(dest))
    if sha_check_failures: raise AssertionError(f'Export copy checksum failure: {sha_check_failures[:3]}')
    audit.update({
        'image_counts': manifest.groupby('split').size().to_dict(),
        'original_images': len(items), 'exported_images': len(manifest),
        'exact_pixel_copies_removed': len(removals), 'cross_class_images_quarantined': len(quarantined),
        'invalid_images': len(invalid), 'groups_detected': len(groups),
        'max_component_size': max(group_sizes), 'link_types': edge_reasons,
        'resnet_weights': weights_source,
        'copied_images_sha256_verified': True,
        'preprocessing': 'Unmodified copies of original files; only EXIF-normalized decoding was used for matching.',
        'warning': 'Never treat PASS_DETECTED_BLOCKING_EDGES as a guarantee of patient-level independence or zero residual leakage.',
    })
    save_json(out / 'reports' / 'leakage_audit.json', audit)
    print('\nFinal counts:')
    print(counts.to_string(index=False))
    print(f'\nAudit status: {audit["status"]}; cross-split semantic-only candidates: {len(cross_semantic)}')
    print(f'Exported: {len(manifest)} images, with {len(removals)} pixel copies removed; {len(quarantined)} ambiguous conflicts quarantined.')
    if settings.zip_output:
        zip_path = zip_dataset(out)
        print(f'Kaggle ZIP: {zip_path} ({zip_path.stat().st_size / 1024**2:.1f} MiB)')
    print(f'Split folders and reports: {out}')
    return manifest, audit
