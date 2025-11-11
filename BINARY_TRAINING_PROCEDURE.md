# Binary Classification Training Procedure
## Training Model with Low Load and High Load Only

---

## Overview

You want to train a **binary classifier** that distinguishes between:
- **Low Load** (Class 0)
- **High Load** (Class 1)

This removes the "Medium Load" class and simplifies the problem to a 2-class classification task.

---

## Step-by-Step Procedure

### Step 1: Prepare Binary Dataset

**What happens:**
- Filter the existing dataset to keep only "Low Load" and "High Load" samples
- Remove all "Medium Load" samples
- Create a new CSV file with binary labels

**Dataset changes:**
```
Before: 229 images
  - Low Load: 70
  - Medium Load: 79  ← Remove these
  - High Load: 80

After: ~150 images
  - Low Load: 70
  - High Load: 80
```

**Code approach:**
```python
# Load merged dataset
df = pd.read_csv("dataset/labels_merged.csv")

# Filter to only Low and High Load
df_binary = df[df['label'].isin(['Low Load', 'High Load'])].copy()

# Save binary dataset
df_binary.to_csv("dataset/labels_binary.csv", index=False)
```

---

### Step 2: Feature Extraction (Same Process)

**What happens:**
- Process each image through the same preprocessing pipeline
- Extract 6-D Hjorth features (unchanged)
- Features are independent of number of classes

**Pipeline remains the same:**
```
Image → Temperature Matrix → Column Signals → 
Hjorth Parameters → 6-D Feature Vector
```

**No changes needed** - feature extraction is identical

---

### Step 3: Model Architecture Changes

**What changes:**

#### Classification Head
- **Before**: 3-class classifier (Low/Medium/High)
- **After**: 2-class classifier (Low/High)
- **Model**: Still RandomForestClassifier, but now outputs 2 classes

#### Label Encoding
- **Before**: 
  - Low Load → 0
  - Medium Load → 1
  - High Load → 2
- **After**:
  - Low Load → 0
  - High Load → 1

#### Regression Head
- **No changes** - Still predicts criticality score (0-1)
- Works the same way

---

### Step 4: Training Process

**Training steps:**

1. **Load binary dataset**
   ```python
   df = pd.read_csv("dataset/labels_binary.csv")
   # Only contains Low Load and High Load
   ```

2. **Extract features** (same as before)
   ```python
   X, y_cls_str, y_reg = build_feature_table(df)
   # X: (n_samples, 6) feature matrix
   # y_cls_str: ['Low Load', 'High Load', ...]
   # y_reg: [0.1, 0.9, 0.2, ...] criticality scores
   ```

3. **Encode labels**
   ```python
   label_encoder.fit_transform(y_cls_str)
   # Low Load → 0
   # High Load → 1
   ```

4. **Train/Test Split**
   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, y_cls_encoded, test_size=0.2, stratify=y_cls_encoded
   )
   # Stratify ensures equal distribution of both classes
   ```

5. **Train Classifier**
   ```python
   # RandomForestClassifier automatically detects 2 classes
   classifier.fit(X_train, y_train)
   # Learns to distinguish Low (0) vs High (1)
   ```

6. **Train Regressor** (unchanged)
   ```python
   regressor.fit(X_train, y_reg_train)
   # Still predicts continuous criticality score
   ```

---

### Step 5: Model Output Changes

**Classification Output:**

**Before (3-class):**
```python
predictions = ['Low Load', 'Medium Load', 'High Load']
probabilities = [[0.1, 0.3, 0.6],  # 3 probabilities
                [0.8, 0.15, 0.05],
                ...]
```

**After (2-class):**
```python
predictions = ['Low Load', 'High Load']
probabilities = [[0.2, 0.8],  # 2 probabilities
                 [0.9, 0.1],
                 ...]
# Sum to 1.0: P(Low) + P(High) = 1.0
```

**Regression Output:**
- **No change** - Still outputs criticality score (0-1)

---

### Step 6: Evaluation Metrics

**Classification Metrics:**

**Before (3-class):**
- Confusion Matrix: 3×3
- Per-class precision/recall for 3 classes
- Overall accuracy

**After (2-class):**
- Confusion Matrix: 2×2
  ```
          Predicted
         Low  High
  True Low  [TP] [FN]
       High [FP] [TN]
  ```
- Binary metrics:
  - **Precision**: TP / (TP + FP)
  - **Recall**: TP / (TP + FN)
  - **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
  - **Accuracy**: (TP + TN) / Total
  - **ROC-AUC**: Area under ROC curve

**Regression Metrics:**
- **No change** - MAE, MSE, RMSE, R²

---

## Advantages of Binary Classification

### 1. **Simpler Decision Boundary**
- Only needs to separate 2 classes
- Easier for model to learn
- Potentially higher accuracy

### 2. **Clearer Interpretation**
- Binary decision: Low or High
- No ambiguous "Medium" category
- More actionable results

### 3. **Better Class Separation**
- Larger gap between classes
- Less confusion at boundaries
- More confident predictions

### 4. **Faster Training**
- Fewer classes = faster convergence
- Simpler model structure

---

## Expected Performance

### Classification
- **Accuracy**: Likely **higher** than 3-class (90-95%+)
- **Precision/Recall**: Better per-class metrics
- **ROC-AUC**: Should be > 0.95 (excellent)

### Regression
- **Similar performance** - Criticality prediction unchanged
- **MAE**: ~0.05-0.07 (similar to current)

---

## Implementation Code Structure

```python
# 1. Filter dataset
df = pd.read_csv("dataset/labels_merged.csv")
df_binary = df[df['label'].isin(['Low Load', 'High Load'])].copy()

# 2. Extract features (same as before)
X, y_cls_str, y_reg = build_feature_table(df_binary)

# 3. Train model (same MultiHeadModel class)
model = MultiHeadModel(random_state=42)
model.fit(X_train, y_cls_train, y_reg_train)
# LabelEncoder automatically handles 2 classes

# 4. Evaluate
y_pred_cls, y_pred_reg = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_cls)
```

---

## Key Differences Summary

| Aspect | 3-Class | 2-Class (Binary) |
|--------|---------|------------------|
| **Classes** | Low, Medium, High | Low, High |
| **Dataset Size** | 229 images | ~150 images |
| **Feature Extraction** | Same (6-D) | Same (6-D) |
| **Model Type** | Multi-class RF | Binary RF |
| **Output Probabilities** | 3 values | 2 values |
| **Confusion Matrix** | 3×3 | 2×2 |
| **Expected Accuracy** | 89.1% | 90-95%+ |
| **Decision Boundary** | More complex | Simpler |

---

## Training Workflow Diagram

```
┌─────────────────────────────────────────┐
│ 1. Load Dataset (labels_merged.csv)    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 2. Filter: Keep only Low & High Load   │
│    Remove Medium Load samples           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 3. Extract Features (6-D Hjorth)        │
│    Same process as before               │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 4. Encode Labels                        │
│    Low Load → 0                         │
│    High Load → 1                        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 5. Train/Test Split (stratified)        │
│    Ensures both classes in train/test   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 6. Train Binary Classifier              │
│    RandomForest (2 classes)            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 7. Train Regressor (unchanged)          │
│    Still predicts criticality (0-1)      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 8. Evaluate                             │
│    Binary metrics (precision, recall)   │
│    ROC-AUC curve                        │
└─────────────────────────────────────────┘
```

---

## What Stays the Same

✅ **Feature extraction pipeline** - Identical  
✅ **Preprocessing steps** - No changes  
✅ **Regression head** - Unchanged  
✅ **Model architecture** - Same RandomForest  
✅ **Training procedure** - Same workflow  
✅ **Inference process** - Same API  

## What Changes

❌ **Number of classes** - 3 → 2  
❌ **Label encoding** - 3 values → 2 values  
❌ **Confusion matrix** - 3×3 → 2×2  
❌ **Output probabilities** - 3 values → 2 values  
❌ **Dataset size** - Removes Medium Load samples  

---

## Next Steps

1. **Filter dataset** - Remove Medium Load samples
2. **Retrain model** - Use binary dataset
3. **Evaluate** - Check binary classification metrics
4. **Compare** - Binary vs 3-class performance
5. **Deploy** - Use binary model for production

---

**Summary**: The training procedure is nearly identical, just with fewer classes. The model automatically adapts to 2 classes, and you'll likely see improved accuracy due to simpler decision boundaries.

