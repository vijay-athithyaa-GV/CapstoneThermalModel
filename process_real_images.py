"""
Process real thermal images from RealImageDataset folder
Analyzes images, extracts temperature statistics, and creates labels based on thermal patterns
"""
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

from busbar.features import preprocess_image_to_features


def analyze_thermal_image(image_path: str) -> Dict:
    """
    Analyze a thermal image and extract temperature statistics
    Returns dictionary with temperature stats and inferred label
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract features and temperature matrix
    try:
        feats, debug = preprocess_image_to_features(
            img_rgb,
            mode="rgb_pseudocolor",
            min_temp_c=20.0,
            max_temp_c=120.0
        )
        temp_matrix = debug['temp_c']
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None
    
    # Calculate temperature statistics
    temp_min = float(np.min(temp_matrix))
    temp_max = float(np.max(temp_matrix))
    temp_mean = float(np.mean(temp_matrix))
    temp_std = float(np.std(temp_matrix))
    temp_median = float(np.median(temp_matrix))
    
    # Hot spot detection (top 5% of pixels)
    temp_flat = temp_matrix.flatten()
    temp_sorted = np.sort(temp_flat)
    hot_threshold = temp_sorted[int(len(temp_sorted) * 0.95)]  # 95th percentile
    hot_spots = temp_matrix > hot_threshold
    n_hot_spots = np.sum(hot_spots)
    hot_spot_max = float(np.max(temp_matrix[hot_spots])) if n_hot_spots > 0 else temp_max
    
    # Calculate hot spot percentage
    hot_spot_percentage = (n_hot_spots / temp_matrix.size) * 100
    
    # Determine label based on temperature patterns
    # This is an automatic labeling based on temperature thresholds
    if temp_max >= 90 or (hot_spot_max >= 85 and hot_spot_percentage > 2):
        label = "High Load"
        # Criticality based on max temp and hot spot characteristics
        criticality = min(0.7 + (temp_max - 90) / 30 * 0.3, 1.0)
        if hot_spot_percentage > 5:
            criticality = min(criticality + 0.1, 1.0)
    elif temp_max >= 60 or (temp_mean >= 50 and temp_std >= 15):
        label = "Medium Load"
        criticality = 0.4 + (temp_max - 60) / 30 * 0.3
    else:
        label = "Low Load"
        criticality = 0.1 + (temp_max - 30) / 30 * 0.3
    
    criticality = np.clip(criticality, 0.0, 1.0)
    
    return {
        "filepath": os.path.basename(image_path),
        "label": label,
        "criticality": float(criticality),
        "temp_min": temp_min,
        "temp_max": temp_max,
        "temp_mean": temp_mean,
        "temp_std": temp_std,
        "temp_median": temp_median,
        "hot_spot_max": hot_spot_max,
        "hot_spot_percentage": hot_spot_percentage,
        "n_hot_spots": int(n_hot_spots)
    }


def process_real_images_dataset(
    real_images_dir: str,
    output_csv: str,
    relative_to: str = None
):
    """
    Process all images in RealImageDataset folder and create labels CSV
    
    Args:
        real_images_dir: Path to RealImageDataset folder
        output_csv: Path to output CSV file
        relative_to: Base directory for relative paths in CSV (default: real_images_dir)
    """
    if relative_to is None:
        relative_to = real_images_dir
    
    real_images_path = Path(real_images_dir)
    if not real_images_path.exists():
        print(f"Error: Directory not found: {real_images_dir}")
        return None
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(real_images_path.glob(f'*{ext}')))
    
    if len(image_files) == 0:
        print(f"Error: No image files found in {real_images_dir}")
        return None
    
    print(f"Found {len(image_files)} images in {real_images_dir}")
    print("Processing images...")
    
    records = []
    for i, img_path in enumerate(sorted(image_files)):
        print(f"  Processing [{i+1}/{len(image_files)}]: {img_path.name}...", end=' ')
        
        result = analyze_thermal_image(str(img_path))
        if result is not None:
            # Calculate relative path
            rel_path = os.path.relpath(img_path, relative_to)
            result['filepath'] = rel_path.replace('\\', '/')  # Use forward slashes
            
            records.append(result)
            print(f"✓ {result['label']} (max: {result['temp_max']:.1f}°C, crit: {result['criticality']:.3f})")
        else:
            print("✗ Failed")
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    if len(df) == 0:
        print("Error: No images were successfully processed")
        return None
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Processed {len(df)} images")
    print(f"✓ Saved to: {output_csv}")
    
    # Print statistics
    print(f"\nLabel distribution:")
    print(df['label'].value_counts().to_string())
    print(f"\nTemperature statistics:")
    print(f"  Min temp: {df['temp_min'].min():.1f}°C - {df['temp_min'].max():.1f}°C")
    print(f"  Max temp: {df['temp_max'].min():.1f}°C - {df['temp_max'].max():.1f}°C")
    print(f"  Mean temp: {df['temp_mean'].min():.1f}°C - {df['temp_mean'].max():.1f}°C")
    print(f"\nCriticality range: {df['criticality'].min():.3f} - {df['criticality'].max():.3f}")
    
    return df


def merge_datasets(
    existing_csv: str,
    new_csv: str,
    output_csv: str,
    existing_root: str = None,
    new_root: str = None
):
    """
    Merge existing dataset with new real images dataset
    
    Args:
        existing_csv: Path to existing labels.csv
        new_csv: Path to new real images labels.csv
        output_csv: Path to output merged CSV
        existing_root: Root directory for existing images (for absolute paths)
        new_root: Root directory for new images (for absolute paths)
    """
    # Load existing dataset
    if os.path.exists(existing_csv):
        df_existing = pd.read_csv(existing_csv)
        print(f"Loaded existing dataset: {len(df_existing)} images")
        
        # Convert to absolute paths if needed
        if existing_root:
            df_existing['abs_path'] = df_existing['filepath'].apply(
                lambda p: str(Path(existing_root) / p)
            )
        else:
            df_existing['abs_path'] = df_existing['filepath']
    else:
        df_existing = pd.DataFrame()
        print("No existing dataset found, creating new one")
    
    # Load new dataset
    if os.path.exists(new_csv):
        df_new = pd.read_csv(new_csv)
        print(f"Loaded new dataset: {len(df_new)} images")
        
        # Convert to absolute paths if needed
        if new_root:
            df_new['abs_path'] = df_new['filepath'].apply(
                lambda p: str(Path(new_root) / p)
            )
        else:
            df_new['abs_path'] = df_new['filepath']
    else:
        print(f"Error: New dataset not found: {new_csv}")
        return None
    
    # Merge datasets
    # Keep only required columns for training
    required_cols = ['filepath', 'label', 'criticality']
    
    df_existing_clean = df_existing[required_cols].copy() if len(df_existing) > 0 else pd.DataFrame(columns=required_cols)
    df_new_clean = df_new[required_cols].copy()
    
    # Combine
    df_merged = pd.concat([df_existing_clean, df_new_clean], ignore_index=True)
    
    # Remove duplicates based on filepath
    df_merged = df_merged.drop_duplicates(subset=['filepath'], keep='last')
    
    # Save merged dataset
    df_merged.to_csv(output_csv, index=False)
    
    print(f"\n✓ Merged dataset created: {len(df_merged)} images")
    print(f"  - Existing: {len(df_existing_clean)} images")
    print(f"  - New: {len(df_new_clean)} images")
    print(f"  - After deduplication: {len(df_merged)} images")
    print(f"✓ Saved to: {output_csv}")
    
    # Print label distribution
    print(f"\nMerged label distribution:")
    print(df_merged['label'].value_counts().to_string())
    
    return df_merged


if __name__ == "__main__":
    import sys
    
    project_dir = os.getcwd()
    real_images_dir = os.path.join(project_dir, "RealImageDataset")
    real_images_csv = os.path.join(project_dir, "real_images_labels.csv")
    
    # Process real images
    print("="*70)
    print("Processing Real Thermal Images")
    print("="*70)
    df_real = process_real_images_dataset(real_images_dir, real_images_csv)
    
    if df_real is not None and len(df_real) > 0:
        # Merge with existing dataset
        print("\n" + "="*70)
        print("Merging with Existing Dataset")
        print("="*70)
        
        existing_csv = os.path.join(project_dir, "dataset", "labels.csv")
        merged_csv = os.path.join(project_dir, "dataset", "labels_merged.csv")
        
        df_merged = merge_datasets(
            existing_csv=existing_csv,
            new_csv=real_images_csv,
            output_csv=merged_csv,
            existing_root=os.path.join(project_dir, "dataset"),
            new_root=real_images_dir
        )
        
        if df_merged is not None:
            print("\n" + "="*70)
            print("Next Steps:")
            print("="*70)
            print("1. Review the labels in real_images_labels.csv")
            print("2. Manually adjust labels if needed")
            print("3. Run: python train.py (update train.py to use labels_merged.csv)")
            print("   OR update dataset/labels.csv with merged data")
            print("="*70)

