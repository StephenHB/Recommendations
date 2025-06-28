"""
Test script for BLP estimation.py

This script tests the TFEstimator class with synthetic data and various scenarios
to ensure the BLP estimation is working correctly.
"""

import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import unittest
from pathlib import Path

# Add the parent directory to the path to import blp modules
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from blp.config.base import Base
    from blp.model.estimation import TFEstimator, Validator
    print("✓ Successfully imported BLP modules")
except ImportError as e:
    print(f"✗ Failed to import BLP modules: {e}")
    print("Make sure you have installed the required dependencies:")
    print("pip install -r blp/requirements.txt")
    sys.exit(1)


class TestBLPEstimation(unittest.TestCase):
    """Test cases for BLP estimation"""
    
    def setUp(self):
        """Set up test data and configuration"""
        # Create synthetic data for testing
        np.random.seed(42)
        n_samples = 1000
        
        # Generate exogenous variables
        self.income = np.random.normal(50000, 10000, n_samples)
        self.cost = np.random.normal(8, 1.5, n_samples)
        self.wage = np.random.normal(20, 3, n_samples)
        
        # Generate endogenous variable (price) with endogeneity
        demand_shock = np.random.normal(0, 1, n_samples)
        self.price = np.random.normal(10, 2, n_samples) + 0.5 * demand_shock
        
        # Generate target variable (quantity)
        self.quantity = 100 - 2 * self.price + 0.001 * self.income + demand_shock
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'quantity': self.quantity,
            'price': self.price,
            'income': self.income,
            'cost': self.cost,
            'wage': self.wage
        })
        
        # Add intercept column
        self.data['intercept'] = 1
        
        # Configure the model
        self.config = Base()
        self.config.reg_param = 0.01
        self.config.target_name = "quantity"
        self.config.exog_ind_names = ["income"]
        self.config.exog_dep_names = ["price"]
        self.config.instrument_variable_names = ["cost", "wage"]
        
        # Initialize estimator
        self.estimator = TFEstimator()
    
    def test_data_creation(self):
        """Test that synthetic data is created correctly"""
        print("\n=== Testing Data Creation ===")
        print(f"Data shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        print(f"Sample data:\n{self.data.head()}")
        
        self.assertEqual(self.data.shape[0], 1000)
        self.assertIn('quantity', self.data.columns)
        self.assertIn('price', self.data.columns)
        self.assertIn('income', self.data.columns)
        print("✓ Data creation test passed")
    
    def test_config_setup(self):
        """Test configuration setup"""
        print("\n=== Testing Configuration Setup ===")
        print(f"Target variable: {self.config.target_name}")
        print(f"Exogenous variables: {self.config.exog_ind_names}")
        print(f"Endogenous variables: {self.config.exog_dep_names}")
        print(f"Instruments: {self.config.instrument_variable_names}")
        print(f"Regularization parameter: {self.config.reg_param}")
        
        self.assertEqual(self.config.target_name, "quantity")
        self.assertEqual(len(self.config.exog_ind_names), 1)
        self.assertEqual(len(self.config.exog_dep_names), 1)
        self.assertEqual(len(self.config.instrument_variable_names), 2)
        print("✓ Configuration setup test passed")
    
    def test_estimator_initialization(self):
        """Test estimator initialization"""
        print("\n=== Testing Estimator Initialization ===")
        
        # Test initial state
        self.assertIsNone(self.estimator.config)
        self.assertIsNone(self.estimator.data)
        self.assertEqual(self.estimator.reg_param, 0.0)
        
        # Test setting config
        self.estimator.set_config(self.config)
        self.assertIsNotNone(self.estimator.config)
        self.assertEqual(self.estimator.reg_param, self.config.reg_param)
        
        # Test setting data
        self.estimator.set_data(self.data)
        self.assertIsNotNone(self.estimator.data)
        
        print("✓ Estimator initialization test passed")
    
    def test_data_processing(self):
        """Test data processing methods"""
        print("\n=== Testing Data Processing ===")
        
        # Test set_target
        target_tensor = self.estimator.set_target(self.data, "quantity")
        print(f"Target tensor shape: {target_tensor.shape}")
        self.assertEqual(target_tensor.shape[0], 1000)
        self.assertEqual(target_tensor.shape[1], 1)
        
        # Test set_features
        features_tensor = self.estimator.set_features(self.data, ("income", "price"))
        print(f"Features tensor shape: {features_tensor.shape}")
        self.assertEqual(features_tensor.shape[0], 1000)
        self.assertEqual(features_tensor.shape[1], 2)
        
        # Test set_instruments
        instruments_tensor = self.estimator.set_instruments(self.data, ("cost", "wage"))
        print(f"Instruments tensor shape: {instruments_tensor.shape}")
        self.assertEqual(instruments_tensor.shape[0], 1000)
        self.assertEqual(instruments_tensor.shape[1], 2)
        
        print("✓ Data processing test passed")
    
    def test_validator(self):
        """Test the Validator class"""
        print("\n=== Testing Validator ===")
        
        validator = Validator()
        
        # Create test tensors
        x = tf.convert_to_tensor(np.random.randn(100, 2), dtype=tf.float32)
        z = tf.convert_to_tensor(np.random.randn(100, 3), dtype=tf.float32)
        
        # Test valid case
        try:
            validator.validate_instrument_variables(x, z)
            print("✓ Valid instrument variables test passed")
        except ValueError as e:
            self.fail(f"Validation failed unexpectedly: {e}")
        
        # Test invalid case (wrong number of observations)
        z_wrong = tf.convert_to_tensor(np.random.randn(50, 3), dtype=tf.float32)
        with self.assertRaises(ValueError):
            validator.validate_instrument_variables(x, z_wrong)
        print("✓ Invalid instrument variables test passed")
    
    def test_full_estimation(self):
        """Test the complete estimation process"""
        print("\n=== Testing Full Estimation ===")
        
        try:
            # Set up estimator
            self.estimator.set_config(self.config).set_data(self.data)
            
            # Run estimation
            results = self.estimator.fit()
            
            print(f"Estimation results:\n{results}")
            print(f"Number of coefficients: {len(results)}")
            print(f"Coefficient names: {list(results.index)}")
            
            # Basic checks
            self.assertIsInstance(results, pd.Series)
            self.assertEqual(len(results), 2)  # income and price
            self.assertIn('income', results.index)
            self.assertIn('price', results.index)
            
            # Check that price coefficient is negative (demand curve)
            price_coef = results['price']
            print(f"Price coefficient: {price_coef}")
            self.assertLess(price_coef, 0, "Price coefficient should be negative for demand curve")
            
            print("✓ Full estimation test passed")
            
        except Exception as e:
            self.fail(f"Estimation failed: {e}")
    
    def test_prediction(self):
        """Test prediction functionality"""
        print("\n=== Testing Prediction ===")
        
        # First fit the model
        self.estimator.set_config(self.config).set_data(self.data)
        self.estimator.fit()
        
        # Test prediction
        try:
            predictions = self.estimator.predict()
            print(f"Prediction shape: {predictions.shape}")
            print(f"Sample predictions: {predictions[:5]}")
            
            self.assertEqual(predictions.shape[0], 1000)
            print("✓ Prediction test passed")
            
        except Exception as e:
            self.fail(f"Prediction failed: {e}")
    
    def test_r_squared_computation(self):
        """Test R-squared computation"""
        print("\n=== Testing R-squared Computation ===")
        
        # Create sample actuals and predictions
        actuals = tf.convert_to_tensor(np.random.randn(100), dtype=tf.float32)
        preds = tf.convert_to_tensor(np.random.randn(100), dtype=tf.float32)
        
        try:
            r_squared = TFEstimator.compute_r_squared(actuals, preds)
            print(f"R-squared: {r_squared}")
            
            self.assertIsInstance(r_squared, np.float32)
            print("✓ R-squared computation test passed")
            
        except Exception as e:
            self.fail(f"R-squared computation failed: {e}")
    
    def test_error_handling(self):
        """Test error handling"""
        print("\n=== Testing Error Handling ===")
        
        # Test fitting without config
        estimator_no_config = TFEstimator()
        with self.assertRaises(ValueError):
            estimator_no_config.fit()
        print("✓ Error handling for missing config passed")
        
        # Test fitting without data
        estimator_no_data = TFEstimator()
        estimator_no_data.set_config(self.config)
        with self.assertRaises(ValueError):
            estimator_no_data.fit()
        print("✓ Error handling for missing data passed")
        
        # Test prediction without fitting
        estimator_no_fit = TFEstimator()
        with self.assertRaises(ValueError):
            estimator_no_fit.predict()
        print("✓ Error handling for prediction without fitting passed")


def run_quick_test():
    """Run a quick test without unittest framework"""
    print("=== Quick Test of BLP Estimation ===")
    
    try:
        # Create synthetic data
        np.random.seed(42)
        n_samples = 500
        
        # Generate data
        income = np.random.normal(50000, 10000, n_samples)
        cost = np.random.normal(8, 1.5, n_samples)
        wage = np.random.normal(20, 3, n_samples)
        
        demand_shock = np.random.normal(0, 1, n_samples)
        price = np.random.normal(10, 2, n_samples) + 0.5 * demand_shock
        quantity = 100 - 2 * price + 0.001 * income + demand_shock
        
        data = pd.DataFrame({
            'quantity': quantity,
            'price': price,
            'income': income,
            'cost': cost,
            'wage': wage,
            'intercept': 1
        })
        
        # Configure and run
        config = Base()
        config.reg_param = 0.01
        config.target_name = "quantity"
        config.exog_ind_names = ["income"]
        config.exog_dep_names = ["price"]
        config.instrument_variable_names = ["cost", "wage"]
        
        estimator = TFEstimator()
        estimator.set_config(config).set_data(data)
        results = estimator.fit()
        
        print("✓ Quick test successful!")
        print(f"Results: {results}")
        return True
        
    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        return False


if __name__ == "__main__":
    print("BLP Estimation Test Suite")
    print("=" * 50)
    
    # Check if TensorFlow is available
    try:
        print(f"✓ TensorFlow version: {tf.__version__}")
    except ImportError:
        print("✗ TensorFlow not available")
        print("Please install TensorFlow: pip install tensorflow")
        sys.exit(1)
    
    # Run quick test first
    if run_quick_test():
        print("\n" + "=" * 50)
        print("Running full test suite...")
        
        # Run unittest suite
        unittest.main(argv=[''], exit=False, verbosity=2)
    else:
        print("Skipping full test suite due to quick test failure")
        sys.exit(1) 