"""
Test criticality-based model where classification is derived from criticality score
"""
import sys
import os
from pathlib import Path

import cv2
import numpy as np

from busbar.features import preprocess_image_to_features
from busbar.model_criticality_based import CriticalityBasedModel
from busbar.thermal_validator import validate_before_processing


def test_criticality_based(image_path: str, model_dir: str = "artifacts_criticality", strict: bool = True):
    """
    Test image with criticality-based model
    
    Classification is derived from criticality score:
    - 0.0 - 0.33 → Low Load
    - 0.33 - 0.67 → Medium Load
    - 0.67 - 1.0 → High Load
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return None
    
    # Check if model exists
    if not os.path.exists(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}")
        print("Please run 'python train_criticality_based.py' first")
        return None
    
    print("="*70)
    print("Testing with Criticality-Based Model")
    print("="*70)
    print(f"\nImage: {image_path}")
    print(f"Model: {model_dir}")
    
    # Load image
    print("\n[1] Loading image...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not read image. Supported formats: PNG, JPG, JPEG")
        return None
    
    print(f"  [OK] Image loaded: {img.shape[1]}x{img.shape[0]} pixels, {img.shape[2]} channels")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Validate thermal image
    print("\n[2] Validating thermal image...")
    should_proceed, validation_msg = validate_before_processing(image_path, img_rgb, strict=strict)
    
    if not should_proceed:
        print(f"  [ERROR] {validation_msg}")
        print("\n" + "="*70)
        print("REJECTED: Image does not appear to be a thermal image")
        print("="*70)
        return None
    
    print(f"  {validation_msg}")
    
    # Load model
    print("\n[3] Loading trained model...")
    try:
        model = CriticalityBasedModel.load(model_dir)
        print("  [OK] Model loaded successfully")
        print(f"  Thresholds: Low<{model.low_threshold}, Med<{model.medium_threshold}, High>={model.medium_threshold}")
    except Exception as e:
        print(f"  [ERROR] Error loading model: {e}")
        return None
    
    # Extract features
    print("\n[4] Extracting features from image...")
    try:
        feats, debug = preprocess_image_to_features(
            img_rgb,
            mode="rgb_pseudocolor",
            min_temp_c=20.0,
            max_temp_c=120.0
        )
        print(f"  [OK] Features extracted: {feats.shape}")
        
        # Show temperature statistics
        temp_matrix = debug['temp_c']
        print(f"\n  Temperature Statistics:")
        print(f"    Min: {temp_matrix.min():.1f} degC")
        print(f"    Max: {temp_matrix.max():.1f} degC")
        print(f"    Mean: {temp_matrix.mean():.1f} degC")
        print(f"    Std: {temp_matrix.std():.1f} degC")
        
    except Exception as e:
        print(f"  [ERROR] Error extracting features: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Run inference
    print("\n[5] Running inference...")
    try:
        # Predict criticality first
        criticality = model.predict_criticality(feats.reshape(1, -1))[0]
        
        # Derive classification from criticality
        load_category = model.criticality_to_class(criticality)
        
        print("  [OK] Prediction complete")
        print(f"\n  Criticality Score: {criticality:.4f}")
        print(f"  Derived Category: {load_category}")
        
    except Exception as e:
        print(f"  [ERROR] Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\n  Criticality Score: {criticality:.4f}")
    print(f"  Load Category:     {load_category}")
    
    # Interpretation
    print(f"\n  Interpretation:")
    if criticality < model.low_threshold:
        risk_level = "[GREEN] LOW RISK - Normal operation (Not Critical)"
        print(f"    Criticality range: 0.0 - {model.low_threshold} -> Low Load")
    elif criticality < model.medium_threshold:
        risk_level = "[YELLOW] MEDIUM RISK - Monitor closely"
        print(f"    Criticality range: {model.low_threshold} - {model.medium_threshold} -> Medium Load")
    else:
        risk_level = "[RED] CRITICAL - Immediate attention required (Very Critical)"
        print(f"    Criticality range: {model.medium_threshold} - 1.0 -> High Load")
    
    print(f"    {risk_level}")
    
    # JSON output
    result = {
        "criticality_score": float(criticality),
        "load_category": load_category
    }
    
    print(f"\n  JSON Output:")
    print(f"    {result}")
    
    print("\n" + "="*70)
    print("How it works:")
    print(f"  1. Model predicts criticality score: {criticality:.4f}")
    print(f"  2. Classification derived from criticality:")
    print(f"     - {criticality:.4f} -> {load_category}")
    print("="*70)
    
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_criticality_based.py <image_path> [model_dir] [--warn-only]")
        print("\nOptions:")
        print("  --warn-only: Warn about non-thermal images but still process them")
        print("\nExample:")
        print("  python test_criticality_based.py HighLoad/IMG-20251105-WA0086.jpg")
        print("  python test_criticality_based.py my_image.jpg artifacts_criticality")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "artifacts_criticality"
    strict = "--warn-only" not in sys.argv
    
    test_criticality_based(image_path, model_dir, strict=strict)

