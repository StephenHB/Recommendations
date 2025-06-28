# BLP (Berry, Levinsohn, and Pakes) Model Implementation

This folder contains a cleaned implementation of the BLP (Berry, Levinsohn, and Pakes) model for demand estimation using TensorFlow. The BLP model is a widely used method in industrial organization and marketing for estimating demand systems with endogenous prices.

## Overview

The BLP model addresses the endogeneity problem in demand estimation by using instrumental variables (IV) in a two-stage least squares (2SLS) framework. This implementation uses TensorFlow for efficient computation and provides a clean, object-oriented interface.

## Key Features

- **2SLS Estimation**: Implements two-stage least squares with TensorFlow
- **Instrumental Variables**: Supports multiple instruments for endogenous variables
- **Regularization**: Includes Tikhonov regularization to handle collinearity
- **Type Safety**: Full type hints and validation
- **Flexible Configuration**: Easy-to-use configuration system

## File Structure

```
blp/
├── config/
│   ├── __init__.py
│   └── base.py              # Configuration class
├── model/
│   ├── __init__.py
│   └── estimation.py        # Main TFEstimator class
├── test/
│   ├── __init__.py
│   ├── test_estimation.py   # Comprehensive test suite
│   ├── run_tests.py         # Test runner script
│   └── README.md            # Testing documentation
├── example_usage.py         # Example usage script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. The main dependencies are:
   - Python >= 3.9, < 3.12
   - TensorFlow 2.15.0
   - NumPy 1.26.4
   - Pandas >= 1.5.0

## Usage

### Basic Usage

```python
from blp.config.base import Base
from blp.model.estimation import TFEstimator
import pandas as pd

# 1. Configure the model
config = Base()
config.input_table_path = "your_data.csv"
config.reg_param = 0.01
config.target_name = "quantity"
config.exog_ind_names = ["income"]      # Exogenous variables
config.exog_dep_names = ["price"]       # Endogenous variables
config.instrument_variable_names = ["cost", "wage"]  # Instruments

# 2. Load data
data = pd.read_csv("your_data.csv")
data["intercept"] = 1  # Add intercept column

# 3. Run estimation
estimator = TFEstimator()
estimator.set_config(config).set_data(data)
results = estimator.fit()

print("Estimated coefficients:")
print(results)
```

### Advanced Usage

```python
# Manual setup with more control
estimator = TFEstimator()

# Set configuration
estimator.set_config(config)

# Set data
estimator.set_data(data)

# Fit the model
coefficients = estimator.fit()

# Make predictions
predictions = estimator.predict()

# Compute R-squared
r_squared = TFEstimator.compute_r_squared(actuals, predictions)
```

## Testing

Run the comprehensive test suite:

```bash
# From the blp directory
cd test
python run_tests.py

# Or run specific tests
python test_estimation.py
```

See `test/README.md` for detailed testing documentation.

## Model Details

### Two-Stage Least Squares (2SLS)

The BLP estimator implements 2SLS in two stages:

1. **First Stage**: Regress endogenous variables on instruments
   - Projects instruments onto the space of endogenous variables
   - Uses Tikhonov regularization to handle collinearity

2. **Second Stage**: Regress target variable on predicted endogenous variables
   - Uses the predicted values from the first stage
   - Provides consistent estimates under valid instruments

### Key Methods

- `fit()`: Performs 2SLS estimation
- `predict()`: Makes predictions using fitted parameters
- `compute_r_squared()`: Computes R-squared for model evaluation
- `first_stage_ols()`: First stage regression (TensorFlow function)
- `second_stage_ols()`: Second stage regression

## Configuration

The `Base` configuration class supports:

- `input_table_path`: Path to input data file
- `reg_param`: Regularization parameter (default: 0.0)
- `target_name`: Name of the target variable
- `exog_ind_names`: List of exogenous independent variables
- `exog_dep_names`: List of endogenous variables
- `instrument_variable_names`: List of instrumental variables

## Validation

The implementation includes validation for:
- Data dimensions consistency
- Instrument relevance (number of instruments >= number of endogenous variables)
- Configuration completeness
- Model fitting prerequisites

## Example

See `example_usage.py` for a complete working example with synthetic data.

## Notes

- The implementation uses TensorFlow for efficient computation
- All tensor operations are decorated with `@tf.function` for performance
- The model assumes that instruments are valid and relevant
- Regularization can be adjusted via the `reg_param` configuration

## References

- Berry, S., Levinsohn, J., & Pakes, A. (1995). Automobile prices in market equilibrium. Econometrica, 63(4), 841-890.
- Nevo, A. (2001). Measuring market power in the ready-to-eat cereal industry. Econometrica, 69(2), 307-342. 