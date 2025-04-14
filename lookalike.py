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
import torchvision.transforms as transforms

# -------------------------------
# Device & Torch Cache Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

def set_torch_hub_cache_dir(cache_path):
    if cache_path:
        torch.hub.set_dir(cache_path)
        print(f"[INFO] Torch hub cache directory set to: {cache_path}")

# -------------------------------
# Load DINOv2 Model & Transform
# -------------------------------
def load_dinov2_model():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return model, transform

# -------------------------------
# Extract Image Features
# -------------------------------
def extract_image_features(image_paths, model, transform):
    feature_vectors = []
    with torch.no_grad():
        for path in tqdm(image_paths, desc="Extracting image features"):
            try:
                image = Image.open(path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)
                output = model(input_tensor)
                feature_vectors.append(output.squeeze().cpu().numpy())
            except Exception as error:
                print(f"[WARN] Failed to process {path}: {error}")
                feature_vectors.append(np.zeros(384))  # fallback for DINOv2 ViT-S/14
    return np.array(feature_vectors)

# -------------------------------
# Dimensionality Reduction
# -------------------------------
def apply_pca(feature_vectors, num_components=100):
    print(f"[INFO] Reducing features to {num_components} dimensions using PCA...")
    pca = PCA(n_components=num_components)
    return pca.fit_transform(feature_vectors)

# -------------------------------
# Intra-cluster Similarity Sort
# -------------------------------
def sort_cluster_by_similarity(image_paths, feature_vectors):
    if not image_paths:
        return []

    remaining = list(zip(image_paths, feature_vectors))
    sorted_paths = []

    # Start from image with oldest modified time
    first_path = min(remaining, key=lambda x: os.path.getmtime(x[0]))
    sorted_paths.append(first_path[0])
    remaining.remove(first_path)

    last_vector = first_path[1]
    while remaining:
        next_path, next_vector = min(
            remaining, key=lambda x: np.dot(last_vector, x[1]) / (np.linalg.norm(last_vector) * np.linalg.norm(x[1]) + 1e-8)
        )
        sorted_paths.append(next_path)
        last_vector = next_vector
        remaining.remove((next_path, next_vector))

    return sorted_paths

# -------------------------------
# Cluster Images & Rename
# -------------------------------
def cluster_and_rename_images(image_paths, feature_vectors, output_dir, eps, min_samples, file_prefix):
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(feature_vectors)
    labels = clustering.labels_

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[INFO] Found {num_clusters} clusters.")

    clusters = defaultdict(list)
    for path, label, vector in zip(image_paths, labels, feature_vectors):
        clusters[label].append((path, vector))

    ordered_image_paths = []
    for label in sorted(clusters.keys()):
        cluster_items = clusters[label]
        sorted_cluster_paths = sort_cluster_by_similarity(
            [p for p, _ in cluster_items], [v for _, v in cluster_items]
        )
        ordered_image_paths.extend(sorted_cluster_paths)

    os.makedirs(output_dir, exist_ok=True)
    total_images = len(ordered_image_paths)
    padding = len(str(total_images))

    for index, original_path in tqdm(enumerate(ordered_image_paths, start=1), total=total_images, desc="Renaming images"):
        extension = os.path.splitext(original_path)[1].lower()
        new_name = f"{file_prefix}{str(index).zfill(padding)}{extension}"
        target_path = os.path.join(output_dir, new_name)
        shutil.copy(original_path, target_path)

# -------------------------------
# Copy & Rename Videos
# -------------------------------
def copy_and_rename_videos(video_paths, output_dir, file_prefix):
    total_videos = len(video_paths)
    padding = len(str(total_videos))
    os.makedirs(output_dir, exist_ok=True)

    for index, video_path in tqdm(enumerate(video_paths, start=1), total=total_videos, desc="Copying videos"):
        extension = os.path.splitext(video_path)[1].lower()
        new_name = f"{file_prefix}{str(index).zfill(padding)}_vid{extension}"
        target_path = os.path.join(output_dir, new_name)
        shutil.copy(video_path, target_path)

# -------------------------------
# Main Execution Pipeline
# -------------------------------
def run_lookalike(input_dir, output_dir, eps, min_samples, cache_dir, use_pca, file_prefix):
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), ".lookalike_cache")
    os.makedirs(cache_dir, exist_ok=True)
    set_torch_hub_cache_dir(cache_dir)

    all_file_paths = [
        os.path.join(input_dir, file_name)
        for file_name in os.listdir(input_dir)
    ]
    image_paths = [f for f in all_file_paths if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    video_paths = [f for f in all_file_paths if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]

    if not image_paths and not video_paths:
        print("[ERROR] No images or videos found in input directory.")
        return

    if image_paths:
        model, transform = load_dinov2_model()
        feature_vectors = extract_image_features(image_paths, model, transform)

        if use_pca:
            feature_vectors = apply_pca(feature_vectors)

        cluster_and_rename_images(image_paths, feature_vectors, output_dir, eps, min_samples, file_prefix)

    if video_paths:
        copy_and_rename_videos(video_paths, output_dir, file_prefix)

    print(f"[DONE] Processed {len(image_paths)} images and {len(video_paths)} videos into: {output_dir}")

# -------------------------------
# CLI Entry Point
# -------------------------------
def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description="Lookalike: Cluster and sort images/videos by visual similarity.")
    parser.add_argument("--input", required=True, help="Input directory with images and videos")
    parser.add_argument("--output", default="output", help="Output directory for renamed files")
    parser.add_argument("--eps", type=float, default=0.3, help="DBSCAN eps (cosine distance threshold)")
    parser.add_argument("--min_samples", type=int, default=2, help="Minimum samples per DBSCAN cluster")
    parser.add_argument("--cache", type=str, help="Path to torch hub cache directory")
    parser.add_argument("--use_pca", action="store_true", help="Enable PCA dimensionality reduction")
    parser.add_argument("--prefix", type=str, default="cluster", help="Prefix for renamed files")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_command_line_arguments()
    run_lookalike(args.input, args.output, args.eps, args.min_samples, args.cache, args.use_pca, args.prefix)
