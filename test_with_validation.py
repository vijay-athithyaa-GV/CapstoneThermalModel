"""
Enhanced test script with thermal image validation
Rejects non-thermal images before processing
"""
import sys
import os
from pathlib import Path

import cv2
import numpy as np

from busbar.features import preprocess_image_to_features
from busbar.model import MultiHeadModel
from busbar.thermal_validator import validate_before_processing


def test_with_validation(image_path: str, model_dir: str = "artifacts", strict: bool = True):
    """
    Test image with thermal validation
    
    Args:
        image_path: Path to thermal image
        model_dir: Directory containing trained models
        strict: If True, reject non-thermal images. If False, warn but proceed.
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return None
    
    # Check if model exists
    if not os.path.exists(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}")
        print("Please run training first.")
        return None
    
    print("="*70)
    print("Testing Image with Thermal Validation")
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
    
    # Validate thermal image
    print("\n[2] Validating thermal image...")
    should_proceed, validation_msg = validate_before_processing(image_path, img_rgb, strict=strict)
    
    if not should_proceed:
        print(f"  ✗ {validation_msg}")
        print("\n" + "="*70)
        print("REJECTED: Image does not appear to be a thermal image")
        print("="*70)
        print("\nThis model is designed for thermal/IR camera images only.")
        print("Please provide a thermal image (FLIR, infrared, or pseudo-color thermal).")
        return None
    
    print(f"  {validation_msg}")
    
    # Load model
    print("\n[3] Loading trained model...")
    try:
        model = MultiHeadModel.load(model_dir)
        print("  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
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
    print("\n[5] Running inference...")
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
        print("Usage: python test_with_validation.py <image_path> [model_dir] [--warn-only]")
        print("\nOptions:")
        print("  --warn-only: Warn about non-thermal images but still process them")
        print("\nExample:")
        print("  python test_with_validation.py HighLoad/IMG-20251105-WA0086.jpg")
        print("  python test_with_validation.py my_image.jpg artifacts --warn-only")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "artifacts"
    strict = "--warn-only" not in sys.argv
    
    test_with_validation(image_path, model_dir, strict=strict)

