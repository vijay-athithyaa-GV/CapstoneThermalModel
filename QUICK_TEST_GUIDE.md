# Quick Guide: Testing Real Thermal Images

## 🚀 Fastest Way

```bash
python test_real_image.py <your_image_path>
```

**Example:**
```bash
python test_real_image.py my_thermal_photo.jpg
```

---

## 📋 Three Methods

### Method 1: Detailed Test Script (Best for First Time)
```bash
python test_real_image.py your_image.jpg
```
**Shows:** Full pipeline, temperature stats, detailed results

### Method 2: Quick Inference (Best for Batch)
```bash
python infer.py your_image.jpg artifacts
```
**Shows:** Just the JSON result

### Method 3: API Server (Best for Production)
```bash
# Start server
uvicorn api:app --host 0.0.0.0 --port 8000

# Test with curl
curl -X POST "http://localhost:8000/predict" -F "file=@your_image.jpg"
```

---

## ✅ What You Need

1. **Trained Model**: Already done! (in `artifacts/` folder)
2. **Your Image**: Any thermal image (PNG, JPG)
3. **Python**: Virtual environment activated

---

## 📸 Supported Images

- ✅ FLIR camera images
- ✅ RGB pseudo-color thermal images
- ✅ Grayscale thermal images
- ✅ Any image where colors = temperature

---

## 📊 Understanding Results

**Load Categories:**
- `Low Load` = Normal operation
- `Medium Load` = Monitor closely
- `High Load` = Take action

**Criticality Score:**
- 0.0-0.3 = 🟢 Low risk
- 0.3-0.6 = 🟡 Medium risk
- 0.6-0.8 = 🟠 High risk
- 0.8-1.0 = 🔴 Critical

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Image not found" | Use full path: `C:/path/to/image.jpg` |
| "Model not found" | Run `python train.py` first |
| Wrong predictions | Check image temperature range matches 20-120°C |

---

## 💡 Pro Tips

1. **Test multiple images** to verify consistency
2. **Check temperature range** - adjust if your camera uses different range
3. **Use PNG format** for best quality
4. **Batch process** multiple images with Python script

---

**Full documentation:** See `USING_REAL_IMAGES.md`

