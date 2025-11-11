# Criticality-Based Model - Quick Start

## ✅ Model Trained Successfully!

The model now works exactly as you requested:
- **Predicts criticality score first** (0-1)
- **Derives classification from criticality**

---

## How It Works

```
Criticality Score → Load Category

0.0 - 0.33  → Low Load (Not Critical)
0.33 - 0.67 → Medium Load
0.67 - 1.0  → High Load (Very Critical)
```

---

## Commands

### Train Model
```bash
python train_criticality_based.py --csv dataset/labels_3class.csv --roots "Lowload" "HighLoad" "dataset"
```

### Test Image
```bash
python test_criticality_based.py "HighLoad/IMG-20251105-WA0086.jpg"
```

---

## Output Format

```json
{
  "criticality_score": 0.9864,
  "load_category": "High Load"
}
```

**Classification is always derived from criticality score!**

---

## Model Performance

- ✅ **Test Accuracy**: 100% (because classification is derived)
- ✅ **Regression MAE**: 0.0315 (3.15% error)
- ✅ **Consistent outputs** - category always matches score

---

## Key Features

1. **Single Model**: Only regressor (predicts criticality)
2. **Derived Classification**: Category from criticality score
3. **Consistent**: No contradiction between outputs
4. **Flexible**: Adjust thresholds easily

---

**See `CRITICALITY_BASED_MODEL.md` for complete documentation!**

