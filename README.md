# Titanic Survival Prediction – Data Cleaning Project

**Course:** Machine Learning & AI — Week 1  
**Focus:** ML fundamentals, data cleaning, and feature preprocessing

## Project objective

Prepare the Titanic passenger dataset for later machine-learning work. This project explores the data, handles missing values, encodes `Sex` and `Embarked`, visualizes passenger ages, and exports a cleaned CSV. It focuses on preprocessing; it does not train or evaluate a survival classifier.

## Requirements

- Python 3.9 or newer
- pandas
- scikit-learn
- matplotlib
- seaborn

## Run the project

1. Open Terminal in this project folder.
2. (Recommended) Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

4. Put the Titanic training CSV at `data/titanic.csv`. You can download the `train.csv` file from the Kaggle Titanic competition and rename it `titanic.csv`. The script also attempts to download a public copy automatically if the local file is absent; that requires internet access.
5. Run:

   ```bash
   python3 src/clean_titanic.py
   ```

## What the script does

1. Loads the CSV with Pandas and displays `.info()`, `.describe(include="all")`, and missing-value counts.
2. Saves an age-distribution histogram to `outputs/age_distribution.png`.
3. Fills missing numeric values with each column's median. Missing `Embarked` values use the mode; missing `Cabin` values use `Unknown`; remaining missing text values use the mode (or `Unknown` if entirely empty).
4. Encodes `Sex` with `LabelEncoder` and one-hot encodes `Embarked` with `OneHotEncoder`.
5. Writes `outputs/titanic_cleaned.csv` and prints the final dimensions and remaining missing-cell count.

## Project structure

```text
titanic_survival_data_cleaning/
├── data/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── src/
│   └── clean_titanic.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Expected outputs

- `outputs/age_distribution.png` — age histogram
- `outputs/titanic_cleaned.csv` — preprocessed dataset

These outputs are generated when the script runs successfully with a dataset. They are not included in the repository before the source dataset is supplied.

## Notes

- `LabelEncoder` maps the observed `Sex` labels to integer values; the mapping is printed when the script runs.
- One-hot encoding replaces `Embarked` with columns such as `Embarked_C`, `Embarked_Q`, and `Embarked_S` depending on categories in the input.
- The exact output columns depend on the input CSV. Do not upload private or licensed data to a public repository without checking its terms.
