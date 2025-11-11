import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


def generate_dataset(dataset_root: str, csv_path: str, n_images: int = 24):
    os.makedirs(dataset_root, exist_ok=True)
    records = []
    labels = ["Low Load", "Medium Load", "High Load"]
    for i in range(n_images):
        label = labels[i % 3]
        score = {"Low Load": 0.2, "Medium Load": 0.6, "High Load": 0.9}[label]
        H, W = 120, 160
        # create a temperature-like pattern with a hot band varying in intensity
        base = np.linspace(0, 1, H).reshape(H, 1) * np.ones((1, W))
        hot_band = np.exp(-0.5 * ((np.arange(H) - H/2) / (H/(10 + (i % 5))))**2).reshape(H, 1)
        synthetic_temp = 30 + 60 * (0.3 * base + 0.7 * hot_band)
        norm = (synthetic_temp - synthetic_temp.min()) / (synthetic_temp.max() - synthetic_temp.min())
        rgb = (plt.cm.inferno(norm)[..., :3]*255).astype(np.uint8)
        fname = f"synthetic_{i:03d}.png"
        cv2.imwrite(os.path.join(dataset_root, fname), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        records.append({"filepath": fname, "label": label, "criticality": score})
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"Synthetic dataset written: {csv_path} with {n_images} images")


if __name__ == "__main__":
    project_dir = os.getcwd()
    dataset_root = os.path.join(project_dir, "dataset")
    csv_path = os.path.join(dataset_root, "labels.csv")
    generate_dataset(dataset_root, csv_path)


