import os
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import cv2

from busbar.features import preprocess_image_to_features


def iter_images(folder: str):
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    for p in sorted(Path(folder).glob("*")):
        if p.is_file() and p.suffix in exts:
            yield p


def compute_criticality_from_image(img_path: str, min_c: float, max_c: float) -> float:
    """
    Compute criticality from image temperature (NOT RECOMMENDED - use label-based instead).
    This can cause misclassification because temperature != load/criticality.
    
    Args:
        img_path: Path to image file
        min_c: Minimum temperature
        max_c: Maximum temperature
    
    Returns:
        Criticality score in [0, 1] based on temperature
    """
    bgr = cv2.imread(img_path)
    if bgr is None:
        return 0.0
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    _, dbg = preprocess_image_to_features(
        rgb, mode="rgb_pseudocolor", min_temp_c=min_c, max_temp_c=max_c
    )
    temp = dbg["temp_c"]
    t_max = float(np.nanmax(temp))
    # Map temperature to 0..1 risk roughly: below 40C -> ~0.1, above 100C -> ~1.0
    crit = (t_max - 40.0) / (100.0 - 40.0)
    crit = 0.1 + np.clip(crit, 0.0, 1.0) * 0.9
    return float(crit)


def compute_criticality_from_label(label: str, add_noise: bool = True) -> float:
    """
    Assign criticality score based on label (not temperature).
    This ensures Low Load images get low criticality and High Load images get high criticality.
    
    Args:
        label: "Low Load", "Medium Load", or "High Load"
        add_noise: If True, add small random variation to avoid overfitting to exact values
    
    Returns:
        Criticality score in [0, 1]
    """
    if label == "Low Load":
        base_crit = 0.15  # Low criticality: 0.1-0.2 range
        if add_noise:
            # Add small random variation: 0.1 to 0.25
            noise = np.random.uniform(-0.05, 0.1)
            crit = np.clip(base_crit + noise, 0.1, 0.3)
        else:
            crit = base_crit
    elif label == "Medium Load":
        base_crit = 0.5  # Medium criticality: 0.4-0.6 range
        if add_noise:
            # Add small random variation: 0.4 to 0.6
            noise = np.random.uniform(-0.1, 0.1)
            crit = np.clip(base_crit + noise, 0.33, 0.67)
        else:
            crit = base_crit
    elif label == "High Load":
        base_crit = 0.85  # High criticality: 0.75-1.0 range
        if add_noise:
            # Add small random variation: 0.75 to 1.0
            noise = np.random.uniform(-0.1, 0.15)
            crit = np.clip(base_crit + noise, 0.7, 1.0)
        else:
            crit = base_crit
    else:
        raise ValueError(f"Unknown label: {label}")
    
    return float(crit)


def build_labels(low_dir: str | None, high_dir: str | None, medium_dir: str | None,
                 out_csv: str, min_c: float, max_c: float, use_label_based_criticality: bool = True):
    """
    Build labels CSV from folder structure.
    
    Args:
        low_dir: Path to Low Load images folder
        high_dir: Path to High Load images folder
        medium_dir: Path to Medium Load images folder (optional)
        out_csv: Output CSV path
        min_c: Minimum temperature (for feature extraction, not criticality)
        max_c: Maximum temperature (for feature extraction, not criticality)
        use_label_based_criticality: If True, assign criticality based on label (recommended).
                                     If False, compute from temperature (not recommended).
    """
    rows = []
    def add_folder(folder: str, label: str):
        base = Path(folder)
        for p in iter_images(folder):
            if use_label_based_criticality:
                # Assign criticality based on label (recommended)
                crit = compute_criticality_from_label(label, add_noise=True)
            else:
                # Compute from temperature (not recommended - can cause misclassification)
                crit = compute_criticality_from_image(str(p), min_c, max_c)
            rel = p.name  # store file name; training script will resolve search roots
            rows.append({"filepath": rel, "label": label, "criticality": crit, "source_dir": str(base)})
    if low_dir:
        add_folder(low_dir, "Low Load")
    if medium_dir:
        add_folder(medium_dir, "Medium Load")
    if high_dir:
        add_folder(high_dir, "High Load")
    if not rows:
        raise SystemExit("No images found. Provide at least one folder.")
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"✓ Wrote {len(df)} rows to {out_csv}")
    print("Label distribution:")
    print(df["label"].value_counts().to_string())
    print("\nCriticality statistics by label:")
    print(df.groupby("label")["criticality"].describe())
    if use_label_based_criticality:
        print("\n✓ Criticality assigned based on label (recommended)")
    else:
        print("\n⚠ Criticality computed from temperature (may cause misclassification)")


def main():
    ap = argparse.ArgumentParser(description="Build labels.csv from Low/Medium/High folders")
    ap.add_argument("--low_dir", type=str, default=None, help="Path to Low Load images folder")
    ap.add_argument("--medium_dir", type=str, default=None, help="Path to Medium Load images folder (optional)")
    ap.add_argument("--high_dir", type=str, default=None, help="Path to High Load images folder")
    ap.add_argument("--out_csv", type=str, default="dataset/labels_user.csv")
    ap.add_argument("--min_temp_c", type=float, default=20.0, help="Min temperature for feature extraction (not criticality)")
    ap.add_argument("--max_temp_c", type=float, default=120.0, help="Max temperature for feature extraction (not criticality)")
    ap.add_argument("--use_temp_based", action="store_true", 
                    help="Use temperature-based criticality (not recommended). Default is label-based.")
    args = ap.parse_args()

    os.makedirs(Path(args.out_csv).parent, exist_ok=True)
    build_labels(args.low_dir, args.high_dir, args.medium_dir, args.out_csv, 
                 args.min_temp_c, args.max_temp_c, 
                 use_label_based_criticality=not args.use_temp_based)


if __name__ == "__main__":
    main()


