import os
import cv2
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pickle

PATCH_DIR = "data/processed/patches_32x32"
CSV_PATH = "data/processed/patch_metadata.csv"
OUT_DATASET = "data/processed/qgan_dataset.npz"
OUT_ENCODERS = "data/processed/qgan_encoders.pkl"

def load_patch(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0  # Normalize to [0,1]

def main():
    df = pd.read_csv(CSV_PATH)

    # Filter missing or broken images
    filtered = []
    image_tensors = []
    for i, row in df.iterrows():
        split = row['split']
        filename = row['filename']
        label = row['label']
        patch_path = os.path.join(PATCH_DIR, split, label, filename)
        if os.path.exists(patch_path):
            img = load_patch(patch_path)
            if img.shape == (32, 32):  # Valid shape
                image_tensors.append(img)
                filtered.append(row)

    df = pd.DataFrame(filtered)

    # Prepare conditioning vector
    encoder = OneHotEncoder(sparse=False)
    scaler = MinMaxScaler()

    label_onehot = encoder.fit_transform(df[['label']])
    physics_features = df[['aspect_ratio', 'area']]
    physics_scaled = scaler.fit_transform(physics_features)

    condition_vectors = np.hstack([label_onehot, physics_scaled])
    image_array = np.stack(image_tensors)

    # Save as .npz (for PyTorch or TensorFlow)
    np.savez_compressed(OUT_DATASET, X=image_array, Y=condition_vectors)

    # Save encoders (use in inference)
    with open(OUT_ENCODERS, "wb") as f:
        pickle.dump({"encoder": encoder, "scaler": scaler}, f)

    print(f"✅ Saved QGAN-ready dataset: {OUT_DATASET}")
    print(f"📦 Total patches: {len(image_array)} | Condition vector size: {condition_vectors.shape[1]}")

if __name__ == "__main__":
    main()
