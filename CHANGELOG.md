# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-11

### Added
- **Criticality-Based Model**: Single RandomForest regressor model for predicting criticality scores
- **Thermal Image Validation**: Automatic validation of thermal images before processing
- **Label-Based Training**: Assign criticality based on folder labels (recommended approach)
- **Comprehensive Evaluation**: Grid-based performance evaluation graphs with dotted styling
- **REST API**: FastAPI server for production deployment
- **Performance Metrics**: Accuracy, MAE, RMSE, R², Cross-Validation scores
- **Feature Extraction**: Hjorth parameters (Activity, Mobility, Complexity) for robust feature extraction
- **Model Export**: Support for joblib and ONNX formats
- **Synthetic Data Generation**: Generate realistic FLIR-like thermal images
- **Documentation**: Comprehensive documentation with multiple guides

### Features
- **6-D Feature Vector**: Mean and std of Hjorth parameters
- **Classification Thresholds**: Configurable thresholds (default: 0.33, 0.67)
- **Multi-Format Support**: PNG, JPG, JPEG image formats
- **Thermal Image Detection**: Validates input images to reject non-thermal images
- **Performance Visualization**: 4 comprehensive evaluation graphs (2x2 grid layout)
- **Cross-Validation**: 5-fold cross-validation for robust evaluation
- **Batch Processing**: Support for processing multiple images
- **API Endpoints**: REST API with `/predict` endpoint

### Performance
- **Test Accuracy**: 90.9%
- **MAE**: 0.0176
- **RMSE**: 0.0243
- **R²**: 0.9952
- **CV MAE**: 0.0679 ± 0.0301

### Documentation
- **README.md**: Comprehensive main documentation
- **MODEL_EXPLANATION.md**: Complete end-to-end model explanation
- **TRAINING_GUIDE.md**: Detailed training instructions
- **PERFORMANCE_EVALUATION_GUIDE.md**: Evaluation guide
- **MODEL_ARCHITECTURE_COMPARISON.md**: Model architecture comparison
- **CONTRIBUTING.md**: Contribution guidelines
- **LICENSE**: MIT License

### Fixed
- **Criticality Assignment**: Fixed issue where all images were assigned criticality=1.0
- **Label-Based Training**: Changed from temperature-based to label-based criticality assignment
- **Thermal Validator**: Improved validator to correctly identify thermal images (including magenta/pink palettes)
- **Graph Generation**: Fixed layout issues in performance evaluation graphs
- **Model Classification**: Fixed issue where Low Load images were misclassified as High Load

### Changed
- **Model Architecture**: Switched from MultiHeadModel (2 models) to CriticalityBasedModel (1 model)
- **Training Approach**: Changed from temperature-based to label-based criticality assignment
- **Validation Logic**: Improved thermal image validator with better color detection and thresholds
- **Graph Layout**: Improved performance evaluation graphs with better spacing and layout

### Security
- **Input Validation**: Added thermal image validation to prevent processing non-thermal images
- **Error Handling**: Improved error handling and validation throughout the codebase

## [0.9.0] - 2025-10-XX (Pre-release)

### Added
- Initial implementation of MultiHeadModel (2-model architecture)
- Basic preprocessing pipeline
- Feature extraction using Hjorth parameters
- ONNX export support
- Basic API server

### Changed
- Refactored from notebook to modular Python package
- Improved code organization and structure

### Fixed
- Various bug fixes and improvements

---

## Version History

### v1.0.0 (Current)
- Production-ready criticality-based model
- Comprehensive evaluation and documentation
- High accuracy (90.9%) on real thermal images

### v0.9.0 (Pre-release)
- Initial multi-head model implementation
- Basic functionality and features

---

## Future Enhancements

### Planned for v1.1.0
- [ ] Support for more thermal image formats
- [ ] Real-time inference optimization
- [ ] Mobile app integration
- [ ] Cloud deployment guides
- [ ] Additional evaluation metrics
- [ ] Model ensemble support

### Planned for v1.2.0
- [ ] Advanced preprocessing options
- [ ] Model visualization tools
- [ ] Automated testing framework
- [ ] Performance benchmarks
- [ ] Additional documentation

---

## Upgrade Guide

### From v0.9.0 to v1.0.0

1. **Update Model Architecture**:
   - Old: MultiHeadModel (2 models)
   - New: CriticalityBasedModel (1 model)

2. **Update Training Script**:
   - Old: `train.py`
   - New: `train_criticality_based.py`

3. **Update Inference Script**:
   - Old: `test_real_image.py`
   - New: `test_criticality_based.py`

4. **Update Labels CSV**:
   - Rebuild labels using `build_labels_from_folders.py`
   - Uses label-based criticality assignment (recommended)

5. **Retrain Model**:
   - Retrain with new script and labels
   - Models are not backward compatible

---

## Migration Notes

### Model Compatibility
- **v1.0.0 models** are not compatible with v0.9.0 code
- **v0.9.0 models** are not compatible with v1.0.0 code
- Retrain models when upgrading

### API Compatibility
- API endpoint remains the same: `/predict`
- Response format remains the same: `{load_category, criticality_score}`
- No breaking changes to API

### Dataset Compatibility
- Old CSV format still supported
- New label-based approach recommended
- Criticality assignment method changed (label-based vs temperature-based)

---

## Release Notes

### v1.0.0 Release Highlights

- ✅ **Single Model Architecture**: Simpler, faster, more consistent
- ✅ **High Accuracy**: 90.9% test accuracy on real thermal images
- ✅ **Thermal Validation**: Automatic validation of thermal images
- ✅ **Comprehensive Evaluation**: Grid-based performance graphs
- ✅ **Production Ready**: REST API, ONNX export, full documentation

---

*Last updated: November 2025*

