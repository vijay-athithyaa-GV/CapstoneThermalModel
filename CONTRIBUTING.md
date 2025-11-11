# Contributing to Busbar Heat Detection System

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## 🤝 How to Contribute

### Reporting Bugs

1. **Check if the bug already exists**: Search existing issues on GitHub
2. **Create a new issue**: Provide a clear title and description
3. **Include details**:
   - Steps to reproduce the bug
   - Expected vs actual behavior
   - Error messages (if any)
   - System information (OS, Python version, etc.)
   - Screenshots (if applicable)

### Suggesting Enhancements

1. **Check existing issues**: See if the enhancement is already suggested
2. **Create a feature request**: Describe the enhancement clearly
3. **Explain the use case**: Why is this enhancement useful?
4. **Provide examples**: Show how it would work

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**: Follow the coding guidelines
4. **Test your changes**: Ensure all tests pass
5. **Commit your changes**: Use clear commit messages
6. **Push to your fork**: `git push origin feature/your-feature-name`
7. **Create a Pull Request**: Provide a clear description

## 📝 Coding Guidelines

### Python Style

- Follow **PEP 8** style guide
- Use **type hints** where possible
- Write **docstrings** for all functions and classes
- Keep functions focused and small
- Use descriptive variable names

### Code Formatting

```python
# Good
def preprocess_image(image: np.ndarray, mode: str = "auto") -> np.ndarray:
    """
    Preprocess thermal image.
    
    Args:
        image: Input image array
        mode: Processing mode
        
    Returns:
        Processed image array
    """
    # Implementation
    pass

# Bad
def proc(img, m='auto'):
    # Process image
    pass
```

### Documentation

- Add docstrings to all public functions
- Include type hints
- Document complex algorithms
- Update README.md if needed

### Testing

- Write tests for new features
- Ensure all tests pass
- Test edge cases
- Test error handling

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python test_model.py

# Test single image
python test_criticality_based.py "path/to/image.jpg"

# Run evaluation
python evaluate_model_performance.py
```

### Writing Tests

```python
def test_feature_extraction():
    """Test feature extraction from image."""
    image = load_test_image()
    features = preprocess_image_to_features(image)
    assert features.shape == (6,)
    assert np.all(np.isfinite(features))
```

## 📋 Pull Request Process

1. **Update documentation**: Update README.md if needed
2. **Add tests**: Include tests for new features
3. **Update CHANGELOG**: Document your changes
4. **Check code style**: Run linters and formatters
5. **Test locally**: Ensure all tests pass
6. **Create PR**: Provide a clear description

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
How was this tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No new warnings
```

## 🎯 Areas for Contribution

### High Priority

- [ ] Additional evaluation metrics
- [ ] Model ensemble support
- [ ] Real-time inference optimization
- [ ] Mobile app integration
- [ ] Cloud deployment guides

### Medium Priority

- [ ] Support for more image formats
- [ ] Additional preprocessing options
- [ ] Model visualization tools
- [ ] Automated testing
- [ ] Performance benchmarks

### Low Priority

- [ ] Documentation improvements
- [ ] Code refactoring
- [ ] UI improvements
- [ ] Additional examples
- [ ] Tutorial videos

## 📚 Resources

### Documentation

- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Type Hints](https://docs.python.org/3/library/typing.html)
- [Docstring Conventions](https://www.python.org/dev/peps/pep-0257/)

### Tools

- **Linting**: pylint, flake8
- **Formatting**: black, autopep8
- **Testing**: pytest, unittest
- **Type Checking**: mypy

## 🤔 Questions?

If you have questions or need help, please:
1. Check the documentation
2. Search existing issues
3. Create a new issue with the "question" label
4. Contact the maintainers

## 🙏 Thank You!

Your contributions make this project better for everyone. Thank you for taking the time to contribute!

---

*Last updated: November 2025*

