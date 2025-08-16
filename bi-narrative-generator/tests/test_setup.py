"""Test the project setup and basic functionality."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all required packages can be imported."""
    try:
        import pandas as pd
        import plotly.express as px
        import streamlit as st
        import openai
        import numpy as np
        print("All core packages imported successfully!")
        return True
    except ImportError as e:
        print(f"Import failed: {e}")
        return False

def test_project_structure():
    """Test that required directories exist."""
    required_dirs = [
        'src', 'templates', 'data', 'tests', 'docs', 
        'web_demo', 'config', 'data/sample_datasets'
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"Missing directories: {missing_dirs}")
        return False
    else:
        print("All required directories exist!")
        return True

def test_config_files():
    """Test that configuration files exist."""
    required_files = [
        'requirements.txt', '.env.example', 'README.md', 
        '.gitignore', 'config/config.yaml'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Missing files: {missing_files}")
        return False
    else:
        print("All required configuration files exist!")
        return True

if __name__ == "__main__":
    print("Testing project setup...")
    print("=" * 50)
    
    all_tests_passed = True
    
    all_tests_passed &= test_project_structure()
    all_tests_passed &= test_config_files()
    all_tests_passed &= test_imports()
    
    print("=" * 50)
    if all_tests_passed:
        print("SUCCESS: All tests passed! Your project is set up correctly!")
        print("READY: Ready to start building your BI Narrative Generator!")
    else:
        print("ERROR: Some tests failed. Please check the setup.")
