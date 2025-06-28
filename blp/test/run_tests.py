#!/usr/bin/env python3
"""
Simple test runner for BLP estimation

This script provides an easy way to run the BLP estimation tests
with proper error handling and dependency checking.
"""

import subprocess
import sys
import os
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = ['numpy', 'pandas', 'tensorflow']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is available")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install missing dependencies:")
        print("pip install -r ../requirements.txt")
        return False
    
    return True


def run_test_script():
    """Run the main test script"""
    test_script = Path(__file__).parent / "test_estimation.py"
    
    if not test_script.exists():
        print(f"Test script not found: {test_script}")
        return False
    
    print(f"\nRunning test script: {test_script}")
    print("=" * 60)
    
    try:
        # Run the test script
        result = subprocess.run([sys.executable, str(test_script)], 
                              capture_output=False, 
                              text=True)
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("✓ All tests completed successfully!")
            return True
        else:
            print(f"\n✗ Tests failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"✗ Error running tests: {e}")
        return False


def main():
    """Main function"""
    print("BLP Estimation Test Runner")
    print("=" * 40)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / "test_estimation.py").exists():
        print("Please run this script from the test directory")
        print(f"Current directory: {current_dir}")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Run tests
    if run_test_script():
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main()) 