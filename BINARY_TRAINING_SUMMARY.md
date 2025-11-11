# Binary Training - Quick Summary

## What Changes When Training with 2 Classes?

### Simple Answer:

**Almost nothing changes!** The same training process works, just with fewer classes.

---

## The Process

### 1. **Data Preparation**
```
Original:  Low Load (70) + Medium Load (79) + High Load (80) = 229 images
Filtered:  Low Load (70) + High Load (80) = 150 images
           ↑ Remove Medium Load samples
```

### 2. **Feature Extraction** 
```
SAME PROCESS:
Image → Temperature → Signals → Hjorth → 6-D Features
No changes needed!
```

### 3. **Label Encoding**
```
Before:  Low=0, Medium=1, High=2  (3 classes)
After:   Low=0, High=1            (2 classes)
```

### 4. **Model Training**
```
SAME MODEL TYPE:
- RandomForestClassifier (automatically handles 2 classes)
- RandomForestRegressor (unchanged)

The model just learns:
  "Is this Low Load (0) or High Load (1)?"
```

### 5. **Output**
```
Before:  [P(Low), P(Medium), P(High)] = [0.2, 0.3, 0.5]
After:   [P(Low), P(High)] = [0.3, 0.7]
         (sums to 1.0)
```

---

## Key Points

✅ **Feature extraction**: Identical (6-D Hjorth features)  
✅ **Training code**: Same (just filter dataset first)  
✅ **Model type**: Same RandomForest  
✅ **Regression**: Unchanged (still predicts 0-1 score)  

❌ **Classes**: 3 → 2  
❌ **Dataset size**: ~229 → ~150 images  
❌ **Output**: 3 probabilities → 2 probabilities  

---

## Expected Results

- **Higher accuracy** (90-95%+) - Simpler problem
- **Better class separation** - Clear Low vs High
- **Faster training** - Fewer classes
- **Clearer decisions** - Binary choice

---

## Implementation

```python
# 1. Filter dataset
df = df[df['label'].isin(['Low Load', 'High Load'])]

# 2. Train (same code as before)
model.fit(X, y_cls, y_reg)

# 3. Predict
prediction = model.predict(X_new)
# Returns: 'Low Load' or 'High Load'
```

**That's it!** The model automatically adapts to 2 classes.

---

**Bottom Line**: Feed Low Load and High Load data, remove Medium Load, train the same way. The model handles the rest automatically!

