import os
import shutil
import argparse
from PIL import Image
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN

import torch
import torchvision.transforms as transforms
import torchvision.models as models

# -------------------------------
# Set custom torch.hub cache dir
# -------------------------------
def set_torch_cache_dir(path):
    if path:
        torch.hub.set_dir(path)
        print(f"[INFO] Torch cache directory set to: {path}")

# -------------------------------
# Feature extractor using ResNet
# -------------------------------
def get_feature_extractor():
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # remove final layer
    model.eval()
    return model, weights.transforms()

# -------------------------------
# Preprocess image using model's transform
# -------------------------------
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# -------------------------------
# Load images and compute features
# -------------------------------
def extract_features(image_paths, model, transform):
    features = []
    with torch.no_grad():
        for path in tqdm(image_paths, desc="Extracting features"):
            img_tensor = preprocess_image(path, transform)
            feature = model(img_tensor).squeeze().numpy()
            features.append(feature)
    return np.array(features)

# -------------------------------
# Cluster and organize images
# -------------------------------
def cluster_and_copy(image_paths, features, output_dir, eps=5, min_samples=2):
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(features)
    labels = clustering.labels_

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {num_clusters} clusters.")

    label_to_paths = defaultdict(list)
    for path, label in zip(image_paths, labels):
        label_to_paths[label].append(path)

    for label, paths in tqdm(label_to_paths.items(), desc="Copying grouped images"):
        label_dir = "unclustered" if label == -1 else f"cluster_{label}"
        save_dir = os.path.join(output_dir, label_dir)
        os.makedirs(save_dir, exist_ok=True)

        for path in paths:
            shutil.copy(path, save_dir)

# -------------------------------
# Main function
# -------------------------------
def main(input_dir, output_dir, eps=5.0, min_samples=2, cache_dir=None):
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), ".lookalike_cache")
        print(f"[INFO] No --cache specified. Using default cache dir: {cache_dir}")
    else:
        print(f"[INFO] Using specified cache dir: {cache_dir}")

    os.makedirs(cache_dir, exist_ok=True)
    set_torch_cache_dir(cache_dir)

    image_paths = [os.path.join(input_dir, fname) for fname in os.listdir(input_dir)
                   if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_paths:
        print("No images found in input directory.")
        return

    model, transform = get_feature_extractor()
    features = extract_features(image_paths, model, transform)
    cluster_and_copy(image_paths, features, output_dir, eps, min_samples)

    print(f"[DONE] Organized {len(image_paths)} images into '{output_dir}'.")

# -------------------------------
# CLI entry point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lookalike: Organize similar photos using AI.")
    parser.add_argument('--input', type=str, required=True, help='Path to input image folder')
    parser.add_argument('--output', type=str, default='output', help='Path to output folder')
    parser.add_argument('--eps', type=float, default=5.0, help='DBSCAN eps value (distance threshold)')
    parser.add_argument('--min_samples', type=int, default=2, help='DBSCAN min_samples value')
    parser.add_argument('--cache', type=str, default=None, help='Path to torch hub cache directory')

    args = parser.parse_args()
    main(args.input, args.output, args.eps, args.min_samples, args.cache)
