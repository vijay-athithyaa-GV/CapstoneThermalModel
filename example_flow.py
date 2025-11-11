"""
Visual example demonstrating the complete model pipeline
Run this to see step-by-step transformations
"""
import numpy as np
import matplotlib.pyplot as plt
from busbar.features import preprocess_image_to_features

# Create a synthetic thermal image with a hot spot
H, W = 120, 160
base_temp = np.linspace(20, 40, H).reshape(H, 1) * np.ones((1, W))
hot_spot = 80 * np.exp(-0.5 * (((np.arange(H) - H/2) / 15)**2 + 
                                ((np.arange(W) - W/2) / 20)**2))
temp_matrix = base_temp + hot_spot

# Convert to RGB pseudo-color
norm = (temp_matrix - temp_matrix.min()) / (temp_matrix.max() - temp_matrix.min())
rgb_image = (plt.cm.inferno(norm)[..., :3] * 255).astype(np.uint8)

print("="*70)
print("COMPLETE MODEL PIPELINE DEMONSTRATION")
print("="*70)

print("\n[INPUT] Thermal Image")
print(f"  Shape: {rgb_image.shape}")
print(f"  Type: RGB pseudo-color image")
print(f"  Temperature range: {temp_matrix.min():.1f}°C to {temp_matrix.max():.1f}°C")

# Process through pipeline
features, debug = preprocess_image_to_features(
    rgb_image, 
    mode="rgb_pseudocolor",
    min_temp_c=20.0,
    max_temp_c=120.0
)

print("\n[STAGE 1] Temperature Matrix")
print(f"  Shape: {debug['temp_c'].shape}")
print(f"  Min temp: {debug['temp_c'].min():.2f}°C")
print(f"  Max temp: {debug['temp_c'].max():.2f}°C")
print(f"  Mean temp: {debug['temp_c'].mean():.2f}°C")

print("\n[STAGE 2] Column Signals")
print(f"  Number of signals: {debug['signals'].shape[0]}")
print(f"  Signal length: {debug['signals'].shape[1]}")
print(f"  Example signal (column 80):")
print(f"    Min: {debug['signals'][80].min():.2f}°C")
print(f"    Max: {debug['signals'][80].max():.2f}°C")
print(f"    Mean: {debug['signals'][80].mean():.2f}°C")

print("\n[STAGE 3] Hjorth Parameters")
print(f"  Activity: min={debug['activity'].min():.2f}, max={debug['activity'].max():.2f}, mean={debug['activity'].mean():.2f}")
print(f"  Mobility: min={debug['mobility'].min():.2f}, max={debug['mobility'].max():.2f}, mean={debug['mobility'].mean():.2f}")
print(f"  Complexity: min={debug['complexity'].min():.2f}, max={debug['complexity'].max():.2f}, mean={debug['complexity'].mean():.2f}")

print("\n[STAGE 4] Aggregated Features (6-D)")
feature_names = [
    "mean(activity)",
    "std(activity)",
    "mean(mobility)",
    "std(mobility)",
    "mean(complexity)",
    "std(complexity)"
]
for i, (name, value) in enumerate(zip(feature_names, features)):
    print(f"  [{i}] {name:20s} = {value:10.4f}")

print("\n[STAGE 5] Feature Vector (ready for model)")
print(f"  Shape: {features.shape}")
print(f"  Dtype: {features.dtype}")
print(f"  Values: {features}")

print("\n[MODEL INPUT]")
print(f"  Normalized 6-D feature vector")
print(f"  → Classifier: predicts load category")
print(f"  → Regressor: predicts criticality score")

print("\n[OUTPUT]")
print("  Expected format:")
print('  {')
print('    "load_category": "High Load",')
print('    "criticality_score": 0.82')
print('  }')

print("\n" + "="*70)
print("PIPELINE COMPLETE")
print("="*70)

# Visualize the pipeline
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Input image
axes[0, 0].imshow(rgb_image)
axes[0, 0].set_title("Input: RGB Thermal Image")
axes[0, 0].axis('off')

# Temperature matrix
im1 = axes[0, 1].imshow(debug['temp_c'], cmap='inferno')
axes[0, 1].set_title("Stage 1: Temperature Matrix (°C)")
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

# Sample signals
axes[0, 2].plot(debug['signals'][40], label='Column 40')
axes[0, 2].plot(debug['signals'][80], label='Column 80')
axes[0, 2].plot(debug['signals'][120], label='Column 120')
axes[0, 2].set_title("Stage 2: Sample Column Signals")
axes[0, 2].set_xlabel("Row (vertical position)")
axes[0, 2].set_ylabel("Temperature (°C)")
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Hjorth parameters
axes[1, 0].plot(debug['activity'], label='Activity', alpha=0.7)
axes[1, 0].set_title("Stage 3: Activity per Column")
axes[1, 0].set_xlabel("Column")
axes[1, 0].set_ylabel("Activity")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(debug['mobility'], label='Mobility', alpha=0.7, color='orange')
axes[1, 1].set_title("Stage 3: Mobility per Column")
axes[1, 1].set_xlabel("Column")
axes[1, 1].set_ylabel("Mobility")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Final features
axes[1, 2].bar(range(6), features, color=['blue', 'cyan', 'green', 'yellow', 'orange', 'red'])
axes[1, 2].set_title("Stage 4: Final 6-D Feature Vector")
axes[1, 2].set_xlabel("Feature Index")
axes[1, 2].set_ylabel("Feature Value")
axes[1, 2].set_xticks(range(6))
axes[1, 2].set_xticklabels([f'F{i}' for i in range(6)], rotation=45)
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('pipeline_visualization.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved to 'pipeline_visualization.png'")
plt.show()

