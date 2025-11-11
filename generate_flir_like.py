"""
Generate realistic FLIR-like thermal images based on reference patterns
Creates 200+ images with varied thermal patterns, hot spots, and load conditions
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter


def create_flir_palette():
    """Create FLIR-like color palette (purple/blue to orange/yellow/white)"""
    # FLIR typically uses: purple/blue (cool) -> magenta -> orange -> yellow -> white (hot)
    colors = np.array([
        [0.0, 0.0, 0.5],      # Dark blue (coldest)
        [0.2, 0.0, 0.5],      # Purple
        [0.5, 0.0, 0.5],      # Magenta
        [0.8, 0.2, 0.0],      # Red-orange
        [1.0, 0.5, 0.0],      # Orange
        [1.0, 0.8, 0.0],      # Yellow-orange
        [1.0, 1.0, 0.5],      # Yellow
        [1.0, 1.0, 1.0],      # White (hottest)
    ])
    return colors


def apply_flir_colormap(temperature_normalized):
    """Apply FLIR-like colormap to normalized temperature"""
    colors = create_flir_palette()
    n_colors = len(colors)
    
    # Map normalized [0,1] to color indices
    indices = (temperature_normalized * (n_colors - 1)).astype(int)
    indices = np.clip(indices, 0, n_colors - 1)
    
    # Create RGB image
    rgb = colors[indices]
    return (rgb * 255).astype(np.uint8)


def generate_thermal_pattern(pattern_type, H, W, seed=None):
    """Generate different thermal patterns based on FLIR image characteristics"""
    if seed is not None:
        np.random.seed(seed)
    
    # Base temperature (ambient)
    base_temp = 25 + np.random.uniform(-5, 5)
    
    if pattern_type == "hot_spot_vertical":
        # Vertical component with hot spot at top (like fuse/circuit breaker)
        temp = np.ones((H, W)) * base_temp
        
        # Vertical component (center)
        comp_width = W // 4 + np.random.randint(-W//8, W//8)
        comp_left = (W - comp_width) // 2
        comp_right = comp_left + comp_width
        
        # Component body (cooler)
        comp_height = H - H // 4
        temp[H//4:comp_height, comp_left:comp_right] = base_temp + np.random.uniform(5, 15)
        
        # Hot spot at connection (top)
        hotspot_size = comp_width // 2
        hotspot_center_x = (comp_left + comp_right) // 2
        hotspot_y = H // 8
        
        y_coords, x_coords = np.ogrid[:H, :W]
        dist = np.sqrt((x_coords - hotspot_center_x)**2 + (y_coords - hotspot_y)**2)
        hotspot = np.exp(-(dist**2) / (2 * (hotspot_size/3)**2))
        temp += hotspot * np.random.uniform(60, 90)  # Hot spot 85-115°C
        
        # Add noise
        temp += np.random.normal(0, 2, (H, W))
        
    elif pattern_type == "multiple_vertical_hot":
        # Multiple vertical components, all hot (like busbars)
        temp = np.ones((H, W)) * base_temp
        
        n_components = np.random.randint(2, 5)
        component_width = W // (n_components + 1)
        
        for i in range(n_components):
            x_start = (i + 1) * component_width
            x_end = x_start + component_width // 2
            
            # Vertical hot component
            temp[:, x_start:x_end] = base_temp + np.random.uniform(20, 50)
            
            # Hotter base
            temp[-H//4:, x_start:x_end] = base_temp + np.random.uniform(40, 60)
        
        # Add noise
        temp += np.random.normal(0, 3, (H, W))
        
    elif pattern_type == "gradient_panel":
        # Panel-like structure with temperature gradients
        temp = np.ones((H, W)) * base_temp
        
        # Horizontal bands with varying temperatures
        n_bands = np.random.randint(3, 6)
        band_height = H // n_bands
        
        for i in range(n_bands):
            y_start = i * band_height
            y_end = min((i + 1) * band_height, H)
            
            # Random temperature for this band
            band_temp = base_temp + np.random.uniform(-10, 30)
            temp[y_start:y_end, :] = band_temp
            
            # Add some geometric shapes (cooler spots)
            if np.random.random() > 0.5:
                shape_x = np.random.randint(0, W - W//4)
                shape_y = np.random.randint(y_start, y_end - band_height//2)
                shape_w = np.random.randint(W//8, W//4)
                shape_h = np.random.randint(band_height//4, band_height//2)
                temp[shape_y:shape_y+shape_h, shape_x:shape_x+shape_w] -= np.random.uniform(5, 15)
        
        # Add noise
        temp += np.random.normal(0, 2, (H, W))
        
    elif pattern_type == "hot_center_cool_edges":
        # Hot center with cooler edges
        temp = np.ones((H, W)) * base_temp
        
        center_y, center_x = H // 2, W // 2
        y_coords, x_coords = np.ogrid[:H, :W]
        
        # Radial gradient
        dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Hot center, cooler edges
        temp += (1 - dist_from_center / max_dist) * np.random.uniform(30, 60)
        
        # Add noise
        temp += np.random.normal(0, 2.5, (H, W))
        
    elif pattern_type == "striped_hot_cool":
        # Alternating hot and cool vertical stripes
        temp = np.ones((H, W)) * base_temp
        
        n_stripes = np.random.randint(4, 8)
        stripe_width = W // n_stripes
        
        for i in range(n_stripes):
            x_start = i * stripe_width
            x_end = min((i + 1) * stripe_width, W)
            
            if i % 2 == 0:
                # Hot stripe
                temp[:, x_start:x_end] = base_temp + np.random.uniform(25, 50)
            else:
                # Cool stripe
                temp[:, x_start:x_end] = base_temp - np.random.uniform(0, 10)
        
        # Add noise
        temp += np.random.normal(0, 2, (H, W))
        
    else:  # "uniform_warm"
        # Uniform warm temperature
        temp = np.ones((H, W)) * (base_temp + np.random.uniform(10, 30))
        temp += np.random.normal(0, 3, (H, W))
    
    # Smooth the temperature field
    temp = gaussian_filter(temp, sigma=1.5)
    
    return np.clip(temp, 20, 120)  # Clip to reasonable range


def determine_label_and_score(temp_matrix):
    """Determine load category and criticality based on temperature patterns"""
    max_temp = np.max(temp_matrix)
    mean_temp = np.mean(temp_matrix)
    std_temp = np.std(temp_matrix)
    
    # Hot spot detection (local maxima)
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(temp_matrix, size=5)
    hot_spots = (temp_matrix == local_max) & (temp_matrix > mean_temp + 2 * std_temp)
    n_hot_spots = np.sum(hot_spots)
    max_hot_spot_temp = np.max(temp_matrix[hot_spots]) if n_hot_spots > 0 else max_temp
    
    # Classification logic
    if max_temp >= 90 or (max_hot_spot_temp >= 85 and n_hot_spots >= 1):
        label = "High Load"
        # Criticality based on max temp and number of hot spots
        criticality = min(0.7 + (max_temp - 90) / 30 * 0.3, 1.0)
        if n_hot_spots >= 2:
            criticality = min(criticality + 0.1, 1.0)
    elif max_temp >= 60 or (mean_temp >= 50 and std_temp >= 15):
        label = "Medium Load"
        criticality = 0.4 + (max_temp - 60) / 30 * 0.3
    else:
        label = "Low Load"
        criticality = 0.1 + (max_temp - 30) / 30 * 0.3
    
    criticality = np.clip(criticality, 0.0, 1.0)
    
    return label, float(criticality)


def generate_flir_like_dataset(dataset_root: str, csv_path: str, n_images: int = 200):
    """Generate realistic FLIR-like thermal images"""
    os.makedirs(dataset_root, exist_ok=True)
    
    # Pattern types based on FLIR image characteristics
    pattern_types = [
        "hot_spot_vertical",      # Like fuse/circuit breaker with hot connection
        "multiple_vertical_hot",  # Multiple hot vertical components
        "gradient_panel",         # Panel with temperature gradients
        "hot_center_cool_edges",  # Hot center, cool edges
        "striped_hot_cool",       # Alternating hot/cool stripes
        "uniform_warm",          # Uniform warm temperature
    ]
    
    # Image dimensions (typical FLIR resolution)
    H, W = 240, 320  # Larger resolution for more detail
    
    records = []
    
    print(f"Generating {n_images} FLIR-like thermal images...")
    
    for i in range(n_images):
        # Randomly select pattern type
        pattern_type = np.random.choice(pattern_types)
        
        # Generate temperature pattern
        temp_matrix = generate_thermal_pattern(pattern_type, H, W, seed=i)
        
        # Determine label and criticality
        label, criticality = determine_label_and_score(temp_matrix)
        
        # Normalize to [0, 1] for colormap
        temp_min, temp_max = temp_matrix.min(), temp_matrix.max()
        if temp_max > temp_min:
            temp_norm = (temp_matrix - temp_min) / (temp_max - temp_min)
        else:
            temp_norm = np.zeros_like(temp_matrix)
        
        # Apply FLIR-like colormap
        rgb = apply_flir_colormap(temp_norm)
        
        # Save image
        fname = f"flir_thermal_{i:04d}.png"
        filepath = os.path.join(dataset_root, fname)
        cv2.imwrite(filepath, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        records.append({
            "filepath": fname,
            "label": label,
            "criticality": criticality,
            "pattern": pattern_type,
            "max_temp": float(temp_max),
            "mean_temp": float(np.mean(temp_matrix))
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{n_images} images...")
    
    # Create DataFrame and save
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    
    # Print statistics
    print(f"\n✓ Dataset generated: {csv_path}")
    print(f"  Total images: {len(df)}")
    print(f"\n  Label distribution:")
    print(df['label'].value_counts().to_string())
    print(f"\n  Temperature statistics:")
    print(f"    Max temp range: {df['max_temp'].min():.1f}°C - {df['max_temp'].max():.1f}°C")
    print(f"    Mean temp range: {df['mean_temp'].min():.1f}°C - {df['mean_temp'].max():.1f}°C")
    print(f"\n  Criticality range: {df['criticality'].min():.3f} - {df['criticality'].max():.3f}")
    
    return df


if __name__ == "__main__":
    project_dir = os.getcwd()
    dataset_root = os.path.join(project_dir, "dataset")
    csv_path = os.path.join(dataset_root, "labels.csv")
    
    # Generate 200+ FLIR-like images
    generate_flir_like_dataset(dataset_root, csv_path, n_images=200)

