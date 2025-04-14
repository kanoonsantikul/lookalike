import os
import shutil
import argparse
from collections import defaultdict
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import torch
import torchvision.transforms as T

# -------------------------------
# Config & Device Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# -------------------------------
# Torch Hub Cache Setup
# -------------------------------
def set_torch_cache_dir(path):
    if path:
        torch.hub.set_dir(path)
        print(f"[INFO] Torch hub cache directory set to: {path}")

# -------------------------------
# Load DINOv2 Model
# -------------------------------
def load_dinov2():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    model.eval()
    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return model, transform

# -------------------------------
# Extract Visual Features
# -------------------------------
def extract_features(image_paths, model, transform):
    features = []
    with torch.no_grad():
        for path in tqdm(image_paths, desc="Extracting features"):
            try:
                image = Image.open(path).convert("RGB")
                tensor = transform(image).unsqueeze(0).to(device)
                output = model(tensor)
                features.append(output.squeeze().cpu().numpy())
            except Exception as e:
                print(f"[WARN] Failed to process {path}: {e}")
                features.append(np.zeros(384))  # fallback for ViT-S/14
    return np.array(features)

# -------------------------------
# Dimensionality Reduction
# -------------------------------
def reduce_features(features, n_components=100):
    print(f"[INFO] Reducing features to {n_components} dimensions using PCA...")
    pca = PCA(n_components=n_components)
    return pca.fit_transform(features)

# -------------------------------
# Clustering & Copying
# -------------------------------
def cluster_images(image_paths, features, output_dir, eps, min_samples, prefix):
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(features)
    labels = clustering.labels_

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[INFO] Found {num_clusters} clusters.")

    grouped = defaultdict(list)
    for path, label, feature in zip(image_paths, labels, features):
        grouped[label].append((path, feature))

    # Sort clusters and sort images inside each cluster by similarity to centroid
    ordered_paths = []
    for label in sorted(grouped.keys()):
        group = grouped[label]
        if len(group) == 1:
            ordered_paths.append(group[0][0])
            continue

        paths, feats = zip(*group)
        feats = np.stack(feats)
        centroid = feats.mean(axis=0)
        distances = np.linalg.norm(feats - centroid, axis=1)
        sorted_indices = np.argsort(distances)
        ordered_paths.extend([paths[i] for i in sorted_indices])

    # Rename and copy
    total = len(ordered_paths)
    padding = len(str(total))
    os.makedirs(output_dir, exist_ok=True)

    for idx, path in tqdm(enumerate(ordered_paths, start=1), total=total, desc="Renaming grouped images"):
        ext = os.path.splitext(path)[1].lower()
        new_filename = f"{prefix}{str(idx).zfill(padding)}{ext}"
        target_path = os.path.join(output_dir, new_filename)
        shutil.copy(path, target_path)

# -------------------------------
# Main Pipeline
# -------------------------------
def lookalike(input_dir, output_dir, eps, min_samples, cache_dir, use_pca, prefix):
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), ".lookalike_cache")
        print(f"[INFO] No --cache specified. Using default: {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)
    set_torch_cache_dir(cache_dir)

    image_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_paths:
        print("[ERROR] No images found in input folder.")
        return

    model, transform = load_dinov2()
    features = extract_features(image_paths, model, transform)

    if use_pca:
        print("[INFO] Reduce features using PCA")
        features = reduce_features(features, n_components=100)

    cluster_images(image_paths, features, output_dir, eps, min_samples, prefix)
    print(f"[DONE] Organized {len(image_paths)} images into: {output_dir}")

# -------------------------------
# CLI Entry Point
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Lookalike: Group similar photos using DINOv2 visual embeddings.")
    parser.add_argument("--input", required=True, help="Input image folder")
    parser.add_argument("--output", default="output", help="Output folder")
    parser.add_argument("--eps", type=float, default=0.3, help="DBSCAN eps (cosine distance threshold)")
    parser.add_argument("--min_samples", type=int, default=2, help="Minimum samples per cluster")
    parser.add_argument("--cache", type=str, help="Path to torch hub cache directory")
    parser.add_argument("--use_pca", action="store_true", help="Apply PCA before clustering")
    parser.add_argument("--prefix", type=str, default="cluster", help="Prefix for renamed output files")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    lookalike(args.input, args.output, args.eps, args.min_samples, args.cache, args.use_pca, args.prefix)
