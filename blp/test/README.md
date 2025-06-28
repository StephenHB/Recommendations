# BLP Testing Directory

This directory contains test files for the BLP (Berry, Levinsohn, and Pakes) estimation implementation.

## Files

### `test_estimation.py`
Comprehensive test suite for the BLP estimation code. This file includes:

- **Unit tests** for all major components
- **Integration tests** for the complete estimation pipeline
- **Error handling tests** to ensure robustness
- **Synthetic data generation** for testing
- **Quick test function** for basic validation

### `run_tests.py`
Simple test runner script that:
- Checks for required dependencies
- Runs the test suite with proper error handling
- Provides clear feedback on test results

### `tf_least_square.ipynb`
Jupyter notebook for testing least squares functionality.

### `tf_mle.ipynb`
Jupyter notebook for testing maximum likelihood estimation.

## How to Run Tests

### Option 1: Using the test runner (Recommended)
```bash
cd blp/test
python run_tests.py
```

### Option 2: Running the test script directly
```bash
cd blp/test
python test_estimation.py
```

### Option 3: Running specific test cases
```bash
cd blp/test
python -m unittest test_estimation.TestBLPEstimation.test_full_estimation
```

### Option 4: Running from project root
```bash
# From the project root directory
python -m blp.test.test_estimation
```

## Test Coverage

The test suite covers:

1. **Data Creation**: Synthetic data generation and validation
2. **Configuration**: Model configuration setup and validation
3. **Estimator Initialization**: Proper initialization and state management
4. **Data Processing**: Tensor conversion and shape validation
5. **Validator**: Instrument variable validation
6. **Full Estimation**: Complete 2SLS estimation pipeline
7. **Prediction**: Model prediction functionality
8. **R-squared Computation**: Model evaluation metrics
9. **Error Handling**: Proper error messages and validation

## Expected Output

When tests run successfully, you should see output like:

```
BLP Estimation Test Suite
==================================================
✓ TensorFlow version: 2.15.0
✓ Successfully imported BLP modules

=== Quick Test of BLP Estimation ===
✓ Quick test successful!
Results: income    0.001234
         price    -1.987654
         Name: estimated params, dtype: float64

==================================================
Running full test suite...

=== Testing Data Creation ===
Data shape: (1000, 6)
Columns: ['quantity', 'price', 'income', 'cost', 'wage', 'intercept']
✓ Data creation test passed

=== Testing Configuration Setup ===
Target variable: quantity
Exogenous variables: ['income']
Endogenous variables: ['price']
Instruments: ['cost', 'wage']
Regularization parameter: 0.01
✓ Configuration setup test passed

... (more test output) ...

✓ All tests completed successfully!
```

## Troubleshooting

### Missing Dependencies
If you see import errors, install the required dependencies:
```bash
pip install -r ../requirements.txt
```

### TensorFlow Issues
If TensorFlow is not available, install it:
```bash
pip install tensorflow
```

### Path Issues
Make sure you're running the tests from the correct directory:
```bash
cd blp/test
```

## Adding New Tests

To add new test cases:

1. Add new test methods to the `TestBLPEstimation` class
2. Follow the naming convention: `test_<feature_name>`
3. Include proper assertions and error handling
4. Add descriptive print statements for debugging

Example:
```python
def test_new_feature(self):
    """Test new feature functionality"""
    print("\n=== Testing New Feature ===")
    
    # Test implementation
    result = self.estimator.new_feature()
    
    # Assertions
    self.assertIsNotNone(result)
    print("✓ New feature test passed")
```

## Notes

- Tests use synthetic data to ensure reproducibility
- Random seeds are set to ensure consistent results
- All tests include proper cleanup and error handling
- The test suite is designed to be run independently 