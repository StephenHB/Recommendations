"""
Example usage of the BLP TFEstimator

This script demonstrates how to use the cleaned BLP estimator for 
2SLS (Two-Stage Least Squares) estimation.
"""

import pandas as pd
import numpy as np
from blp.config.base import Base
from blp.model.estimation import TFEstimator


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create sample data for demonstration purposes
    
    Args:
        n_samples: Number of observations to generate
        
    Returns:
        DataFrame with sample data
    """
    np.random.seed(42)
    
    # Generate exogenous variables
    price = np.random.normal(10, 2, n_samples)
    income = np.random.normal(50000, 10000, n_samples)
    
    # Generate instruments (cost shifters)
    cost = np.random.normal(8, 1.5, n_samples)
    wage = np.random.normal(20, 3, n_samples)
    
    # Generate endogenous variable (demand)
    # Price is endogenous due to correlation with unobserved demand shocks
    demand_shock = np.random.normal(0, 1, n_samples)
    price = price + 0.5 * demand_shock  # Price is correlated with demand shock
    
    # Generate target variable (quantity demanded)
    quantity = 100 - 2 * price + 0.001 * income + demand_shock
    
    # Create DataFrame
    data = pd.DataFrame({
        'quantity': quantity,
        'price': price,
        'income': income,
        'cost': cost,
        'wage': wage
    })
    
    return data


def main():
    """Main function demonstrating BLP estimator usage"""
    
    # Create sample data
    print("Creating sample data...")
    data = create_sample_data(1000)
    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print()
    
    # Configure the model
    print("Configuring the model...")
    config = Base()
    config.input_table_path = "sample_data.csv"  # In practice, this would be a real file path
    config.reg_param = 0.01  # Regularization parameter
    config.target_name = "quantity"
    config.exog_ind_names = ["income"]  # Exogenous independent variables
    config.exog_dep_names = ["price"]   # Endogenous variables
    config.instrument_variable_names = ["cost", "wage"]  # Instruments
    
    # Save sample data to CSV for demonstration
    data.to_csv("sample_data.csv", index=False)
    
    # Initialize and run the estimator
    print("Running BLP estimation...")
    estimator = TFEstimator()
    
    try:
        # Method 1: Using execute() method (reads data from file)
        results = estimator.set_config(config).execute()
        print("Estimation completed successfully!")
        print(f"Estimated coefficients:\n{results.coefficients}")
        
    except Exception as e:
        print(f"Error during estimation: {e}")
        print("Trying alternative approach...")
        
        # Method 2: Manual setup (more flexible)
        estimator = TFEstimator()
        estimator.set_config(config).set_data(data)
        results = estimator.fit()
        print("Estimation completed successfully!")
        print(f"Estimated coefficients:\n{results}")
    
    # Clean up
    import os
    if os.path.exists("sample_data.csv"):
        os.remove("sample_data.csv")


if __name__ == "__main__":
    main() 