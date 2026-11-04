"""
RESPONSIBILITY: tests/test_preprocessing.py
---------------------------------------------
- Pytest unit tests for src/data/preprocess.py
- Tests:
    - Dataset loads successfully
    - TotalCharges converts to numeric correctly
    - customerID column is removed
    - No null values after preprocessing
    - Target column is binary (0/1)
    - train.csv and test.csv are created correctly
    - Train/test sizes are within expected range
"""

