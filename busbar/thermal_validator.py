"""
Thermal Image Validator
Detects if an image is actually a thermal image before processing
"""
from typing import Tuple
import numpy as np
import cv2


def is_thermal_image(image: np.ndarray, threshold: float = 0.3) -> Tuple[bool, float, str]:
    """
    Validate if an image appears to be a thermal image
    
    Thermal images typically have:
    1. Specific color palettes (purple/blue to orange/yellow/white)
    2. Smooth temperature gradients
    3. Limited color diversity
    4. Specific intensity distributions
    
    Args:
        image: RGB image array (H, W, 3)
        threshold: Confidence threshold (0-1)
    
    Returns:
        (is_thermal, confidence, reason)
    """
    if image is None or image.size == 0:
        return False, 0.0, "Empty image"
    
    if image.ndim != 3 or image.shape[2] != 3:
        return False, 0.0, "Not RGB image"
    
    # Convert to float
    rgb = image.astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    
    # Check 1: Color palette analysis (thermal images have specific color ranges)
    # Thermal images typically have warm colors (red/orange/yellow) and cool colors (blue/purple)
    # but not many greens
    
    # Extract color channels
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    
    # Check for thermal-like color distribution
    # Thermal images can have:
    # - Warm colors: red/orange/yellow (high R, low G)
    # - Cool colors: blue/purple/magenta (low R, high B, or high R+B, low G)
    # - Magenta/pink: high R+B, low G (common in thermal)
    
    total_pixels = rgb.shape[0] * rgb.shape[1]
    
    # Warm colors (red/orange/yellow) - hot areas
    warm_pixels = np.sum((r > 0.5) & (g < 0.6) & (b < 0.6) & (r > g) & (r > b))
    warm_ratio = warm_pixels / total_pixels
    
    # Cool colors (blue/purple) - cold areas
    cool_pixels = np.sum((b > 0.5) & (r < 0.5) & (g < 0.5) & (b > r) & (b > g))
    cool_ratio = cool_pixels / total_pixels
    
    # Magenta/pink (common in thermal images) - medium temperatures
    magenta_pixels = np.sum((r > 0.4) & (b > 0.4) & (g < 0.5) & (abs(r - b) < 0.3))
    magenta_ratio = magenta_pixels / total_pixels
    
    # Green (rare in thermal images - usually indicates non-thermal)
    green_pixels = np.sum((g > 0.6) & (g > r + 0.2) & (g > b + 0.2))
    green_ratio = green_pixels / total_pixels
    
    # Thermal images should have warm OR cool OR magenta regions, but minimal green
    thermal_color_score = (warm_ratio + cool_ratio + magenta_ratio * 0.8) * (1.0 - min(green_ratio * 3, 0.8))
    
    # Boost score if we have both warm and cool (typical thermal gradient)
    if warm_ratio > 0.1 and cool_ratio > 0.1:
        thermal_color_score = min(thermal_color_score * 1.3, 1.0)
    
    # Also check for overall color distribution - thermal images often lack pure greens
    if green_ratio < 0.15:  # Less than 15% green is good
        thermal_color_score = min(thermal_color_score * 1.2, 1.0)
    
    # Check 2: Gradient smoothness (thermal images have smooth temperature transitions)
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    avg_gradient = (np.mean(grad_x) + np.mean(grad_y)) / 2.0
    
    # Thermal images have moderate to smooth gradients
    # Accept wider range: 0.05 to 0.25 (was too strict at 0.15)
    if 0.05 <= avg_gradient <= 0.25:
        gradient_score = 1.0
    elif avg_gradient < 0.05:
        gradient_score = avg_gradient / 0.05  # Too flat
    else:
        gradient_score = max(0.0, 1.0 - (avg_gradient - 0.25) / 0.15)  # Too sharp
    
    # Check 3: Color diversity (thermal images have limited color palette)
    # Convert to HSV and check saturation distribution
    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    saturation = hsv[..., 1]
    avg_saturation = np.mean(saturation)
    
    # Thermal images often have moderate to high saturation
    # Accept wider range: 0.3 to 0.9 (more lenient)
    if 0.3 <= avg_saturation <= 0.9:
        saturation_score = 1.0
    elif avg_saturation < 0.3:
        saturation_score = avg_saturation / 0.3  # Too desaturated
    else:
        saturation_score = max(0.5, 1.0 - (avg_saturation - 0.9) / 0.1)  # Very high saturation
    
    # Check 4: Intensity distribution (thermal images have specific brightness patterns)
    brightness = np.mean(gray)
    # Thermal images can have varying brightness - be more lenient
    # Accept range: 0.2 to 0.8 (was too strict)
    if 0.2 <= brightness <= 0.8:
        brightness_score = 1.0
    elif brightness < 0.2:
        brightness_score = brightness / 0.2  # Too dark
    else:
        brightness_score = max(0.5, 1.0 - (brightness - 0.8) / 0.2)  # Too bright
    
    # Combined confidence score
    confidence = (
        thermal_color_score * 0.5 +  # Most important: color palette (increased weight)
        gradient_score * 0.2 +        # Smooth gradients (reduced weight - less strict)
        saturation_score * 0.2 +      # Saturation patterns
        brightness_score * 0.1         # Brightness (least important)
    )
    
    # Lower threshold and add bonus for having thermal-like colors
    effective_threshold = threshold * 0.8  # Lower effective threshold
    if thermal_color_score > 0.2:  # If has some thermal colors, be more lenient
        effective_threshold = threshold * 0.7
    
    is_thermal = confidence >= effective_threshold
    
    # Generate reason
    reasons = []
    if thermal_color_score < 0.3:
        reasons.append("unusual color palette")
    if gradient_score < 0.3:
        reasons.append("sharp edges (not thermal-like)")
    if saturation_score < 0.3:
        reasons.append("unusual saturation")
    if brightness_score < 0.3:
        reasons.append("unusual brightness")
    
    reason = "Looks like thermal image" if is_thermal else f"May not be thermal: {', '.join(reasons) if reasons else 'low confidence'}"
    
    return is_thermal, float(confidence), reason


def validate_before_processing(image_path: str, image: np.ndarray, 
                              strict: bool = True, threshold: float = 0.3) -> Tuple[bool, str]:
    """
    Validate image before processing with model
    
    Args:
        image_path: Path to image file
        image: Loaded image array
        strict: If True, reject non-thermal images. If False, warn but proceed.
        threshold: Confidence threshold
    
    Returns:
        (should_proceed, message)
    """
    is_thermal, confidence, reason = is_thermal_image(image, threshold)
    
    if strict and not is_thermal:
        return False, f"REJECTED: {reason} (confidence: {confidence:.2f}). This doesn't appear to be a thermal image."
    
    if not is_thermal:
        return True, f"WARNING: {reason} (confidence: {confidence:.2f}). Proceeding anyway..."
    
    return True, f"VALIDATED: {reason} (confidence: {confidence:.2f})"

