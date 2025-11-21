# 4 Key Points - Busbar Heat Detection Model

## 1. **What It Does**
- **Analyzes thermal infrared images** of electrical busbars
- **Predicts criticality score** (0.0 = no risk, 1.0 = critical)
- **Classifies thermal load** as Low/Medium/High based on criticality thresholds
- **Purpose**: Preventive maintenance and safety monitoring to detect overheating before failures

## 2. **How It Works**
- **Input**: RGB thermal image → **Output**: Criticality score + Load category
- **Processing Pipeline**:
  1. Convert image colors to temperature values (20-120°C)
  2. Extract vertical temperature profiles (column signals)
  3. Compute Hjorth parameters (Activity, Mobility, Complexity) - signal processing features
  4. Aggregate to 6-D feature vector (mean/std of each parameter)
  5. RandomForest Regressor predicts criticality (0-1)
  6. Classification derived from criticality: Low (<0.33), Medium (0.33-0.67), High (≥0.67)

## 3. **Key Features**
- **Hjorth Parameters**: Signal processing features that capture temperature patterns (variation, rate of change, complexity)
- **Single Model Architecture**: One RandomForest regressor (simpler, faster, consistent)
- **Thermal Image Validation**: Automatically validates input images before processing
- **90.9% Accuracy**: High performance on real thermal images

## 4. **Usage**
- **Test Single Image**: `python test_criticality_based.py "image.jpg"`
- **Train Model**: `python train_criticality_based.py --csv "labels.csv" --roots "Lowload" "HighLoad"`
- **REST API**: `uvicorn api:app --host 0.0.0.0 --port 8000`
- **Output Format**: `{"load_category": "High Load", "criticality_score": 0.85}`

---

**In Summary**: Thermal image → Temperature extraction → Hjorth features → RandomForest → Criticality score → Load category

