import os
import shutil
import argparse
from collections import defaultdict
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN

import torch
from torchvision import models
from torchvision.models import ResNet50_Weights

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
        print(f"[INFO] Torch cache directory set to: {path}")

# -------------------------------
# Feature Extractor Setup
# -------------------------------
def load_model():
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove final FC
    model.to(device)
    model.eval()
    return model, weights.transforms()

# -------------------------------
# Feature Extraction
# -------------------------------
def extract_features(image_paths, model, transform):
    features = []
    with torch.no_grad():
        for path in tqdm(image_paths, desc="Extracting features"):
            try:
                image = Image.open(path).convert('RGB')
                tensor = transform(image).unsqueeze(0).to(device)
                output = model(tensor).squeeze().cpu().numpy()
                features.append(output)
            except Exception as e:
                print(f"[WARN] Failed to process {path}: {e}")
                features.append(np.zeros(2048))  # fallback to zeros if broken
    return np.array(features)

# -------------------------------
# Clustering & Copying Files
# -------------------------------
def cluster_images(image_paths, features, output_dir, eps, min_samples):
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(features)
    labels = clustering.labels_

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[INFO] Found {num_clusters} clusters.")

    grouped = defaultdict(list)
    for path, label in zip(image_paths, labels):
        grouped[label].append(path)

    for label, paths in tqdm(grouped.items(), desc="Copying grouped images"):
        subfolder = "unclustered" if label == -1 else f"cluster_{label}"
        target_dir = os.path.join(output_dir, subfolder)
        os.makedirs(target_dir, exist_ok=True)
        for path in paths:
            shutil.copy(path, target_dir)

# -------------------------------
# Main Pipeline
# -------------------------------
def lookalike(input_dir, output_dir, eps, min_samples, cache_dir):
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), ".lookalike_cache")
        print(f"[INFO] No --cache specified. Using default: {cache_dir}")
    else:
        print(f"[INFO] Using specified cache directory: {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)
    set_torch_cache_dir(cache_dir)

    image_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not image_paths:
        print("[ERROR] No images found.")
        return

    model, transform = load_model()
    features = extract_features(image_paths, model, transform)
    cluster_images(image_paths, features, output_dir, eps, min_samples)
    print(f"[DONE] Organized {len(image_paths)} images into: {output_dir}")

# -------------------------------
# CLI Entry Point
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Lookalike: Organize similar photos using AI.")
    parser.add_argument('--input', required=True, help='Path to input folder')
    parser.add_argument('--output', default='output', help='Path to save clustered images')
    parser.add_argument('--eps', type=float, default=5.0, help='DBSCAN eps (distance threshold)')
    parser.add_argument('--min_samples', type=int, default=2, help='Minimum samples per cluster')
    parser.add_argument('--cache', type=str, help='Path to torch hub cache directory')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    lookalike(args.input, args.output, args.eps, args.min_samples, args.cache)
