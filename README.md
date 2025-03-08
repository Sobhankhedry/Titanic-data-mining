# Titanic Dataset Preprocessing

This repository contains a Python script to preprocess the Titanic dataset for machine learning. The script cleans missing values, encodes categorical data, extracts features, and normalizes numerical values.

## Features of the Preprocessing Pipeline

### 1. Handling Missing Values
- `Age`: Filled with the median value.
- `Fare`: Filled with the median value.
- `Embarked`: Filled with the mode (most frequent value).
- `Cabin`: Filled with "Unknown".

### 2. Encoding Categorical Variables
- `Sex`: Label encoded (Male -> 0, Female -> 1).
- `Embarked`: One-hot encoded (with `drop_first=True` to avoid multicollinearity).

### 3. Feature Engineering
- **Extracting Titles**: Extracted title from the `Name` column.
- **Cabin Floor**: Extracted the first letter of the `Cabin` column (defaulting to `X` if unknown).
- **Family Size**: Created a new feature `FamilySize` as the sum of `SibSp` and `Parch` plus one.

### 4. Dropping Unnecessary Columns
- Removed `PassengerId`, `Name`, and `Ticket` as they are not useful for predictions.

### 5. Normalization
- Used `MinMaxScaler` to scale `Age` and `Fare` between 0 and 1.

## Usage

### Prerequisites
Ensure you have Python installed along with the required dependencies:

```bash
pip install pandas scikit-learn
```

### Running the Script
Place the `train.csv` and `test.csv` files in the same directory as the script, then execute:

```bash
python preprocess.py
```

### Output
- The processed training dataset will be saved as `train_result.csv`.
- The processed test dataset will be saved as `test_result.csv`.

## Example Output

```plaintext
   Survived  Pclass  Sex   Age  SibSp  Parch      Fare  Embarked_Q  Embarked_S Title Cabin  FamilySize
0        0       3    0  0.27      1      0  0.014151           0           1   Mr     X          2
1        1       1    1  0.42      1      0  0.139136           0           0   Mrs    C          2
```

## License
This project is open-source and available under the MIT License.

## Author
Sobhan Khedry

