"""
Simple script to test the model on a real thermal image
Usage: python test_real_image.py <path_to_image>
"""
import sys
import os
from pathlib import Path

import cv2
import numpy as np

from busbar.features import preprocess_image_to_features
from busbar.model import MultiHeadModel


def test_real_image(image_path: str, model_dir: str = "artifacts"):
    """
    Load a real thermal image and run inference
    
    Args:
        image_path: Path to the thermal image (PNG, JPG, etc.)
        model_dir: Directory containing trained models (default: "artifacts")
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return None
    
    # Check if model exists
    if not os.path.exists(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}")
        print("Please run 'python train.py' first to train the model.")
        return None
    
    print("="*70)
    print("Testing Real Thermal Image")
    print("="*70)
    print(f"\nImage: {image_path}")
    print(f"Model: {model_dir}")
    
    # Load image
    print("\n[1] Loading image...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not read image. Supported formats: PNG, JPG, JPEG")
        return None
    
    print(f"  ✓ Image loaded: {img.shape[1]}×{img.shape[0]} pixels, {img.shape[2]} channels")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load model
    print("\n[2] Loading trained model...")
    try:
        model = MultiHeadModel.load(model_dir)
        print("  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return None
    
    # Extract features
    print("\n[3] Extracting features from image...")
    try:
        feats, debug = preprocess_image_to_features(
            img_rgb,
            mode="rgb_pseudocolor",  # or "auto" to auto-detect
            min_temp_c=20.0,
            max_temp_c=120.0
        )
        print(f"  ✓ Features extracted: {feats.shape}")
        print(f"    Feature values: {feats}")
        
        # Show temperature statistics
        temp_matrix = debug['temp_c']
        print(f"\n  Temperature Statistics:")
        print(f"    Min: {temp_matrix.min():.1f}°C")
        print(f"    Max: {temp_matrix.max():.1f}°C")
        print(f"    Mean: {temp_matrix.mean():.1f}°C")
        print(f"    Std: {temp_matrix.std():.1f}°C")
        
    except Exception as e:
        print(f"  ✗ Error extracting features: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Run inference
    print("\n[4] Running inference...")
    try:
        y_cls, y_reg = model.predict(feats.reshape(1, -1))
        
        load_category = str(y_cls[0])
        criticality_score = float(y_reg[0])
        
        print("  ✓ Prediction complete")
        
    except Exception as e:
        print(f"  ✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\n  Load Category:     {load_category}")
    print(f"  Criticality Score: {criticality_score:.4f}")
    
    # Interpretation
    print(f"\n  Interpretation:")
    if criticality_score < 0.3:
        risk_level = "🟢 LOW RISK - Normal operation"
    elif criticality_score < 0.6:
        risk_level = "🟡 MEDIUM RISK - Monitor closely"
    elif criticality_score < 0.8:
        risk_level = "🟠 HIGH RISK - Take action"
    else:
        risk_level = "🔴 CRITICAL - Immediate attention required"
    
    print(f"    {risk_level}")
    
    # JSON output
    result = {
        "load_category": load_category,
        "criticality_score": criticality_score
    }
    
    print(f"\n  JSON Output:")
    print(f"    {result}")
    
    print("\n" + "="*70)
    
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_real_image.py <path_to_thermal_image> [model_dir]")
        print("\nExample:")
        print("  python test_real_image.py my_thermal_image.jpg")
        print("  python test_real_image.py dataset/flir_thermal_0015.png")
        print("  python test_real_image.py C:/path/to/image.png artifacts")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else "artifacts"
    
    test_real_image(image_path, model_dir)

