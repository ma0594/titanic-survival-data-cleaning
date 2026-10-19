"""Titanic Survival Prediction - Data Cleaning Project (Week 1 ML & AI).

Downloads a public copy of the classic Titanic training CSV if no local file is
provided, cleans missing values, encodes Sex and Embarked, plots age distribution,
and writes a cleaned CSV to outputs/.
"""
from pathlib import Path
import sys
import urllib.request

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
LOCAL_DATASET = DATA_DIR / "titanic.csv"
DATASET_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


def get_dataset_path() -> Path:
    """Use data/titanic.csv if present; otherwise try downloading a public CSV."""
    if LOCAL_DATASET.exists():
        return LOCAL_DATASET
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("No local dataset found. Attempting to download the Titanic CSV...")
    try:
        urllib.request.urlretrieve(DATASET_URL, LOCAL_DATASET)
        print(f"Dataset downloaded to: {LOCAL_DATASET}")
        return LOCAL_DATASET
    except Exception as exc:
        raise RuntimeError(
            "Could not download the dataset. Download the Titanic training CSV "
            "(commonly named train.csv) from Kaggle and place it at "
            f"{LOCAL_DATASET}. Then run this script again. Details: {exc}"
        ) from exc


def make_one_hot_encoder():
    """Support both newer and older scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        dataset_path = get_dataset_path()
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    df = pd.read_csv(dataset_path)
    print("\n=== Dataset preview ===")
    print(df.head())
    print("\n=== Dataset information ===")
    df.info()
    print("\n=== Descriptive statistics ===")
    print(df.describe(include="all").transpose())
    print("\n=== Missing values before cleaning ===")
    print(df.isnull().sum().sort_values(ascending=False))

    # Visualize age distribution before imputation, where Age is available.
    if "Age" in df.columns:
        plt.figure(figsize=(9, 5))
        sns.histplot(data=df, x="Age", bins=30, kde=True, color="steelblue")
        plt.title("Titanic Passenger Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Passenger count")
        plt.tight_layout()
        plot_path = OUTPUT_DIR / "age_distribution.png"
        plt.savefig(plot_path, dpi=160)
        plt.close()
        print(f"\nAge distribution plot saved to: {plot_path}")
    else:
        print("\nSkipped age plot: dataset has no 'Age' column.")

    # Numeric columns: median imputation is robust to skew and outliers.
    for column in df.select_dtypes(include="number").columns:
        if df[column].isna().any():
            median_value = df[column].median()
            if pd.notna(median_value):
                df[column] = df[column].fillna(median_value)

    # Categorical columns: use mode for Embarked and a clear label for Cabin.
    if "Embarked" in df.columns and df["Embarked"].isna().any():
        mode_values = df["Embarked"].mode(dropna=True)
        if not mode_values.empty:
            df["Embarked"] = df["Embarked"].fillna(mode_values.iloc[0])
        else:
            df["Embarked"] = df["Embarked"].fillna("Unknown")
    if "Cabin" in df.columns:
        df["Cabin"] = df["Cabin"].fillna("Unknown")

    # Fill remaining text/category columns with their most frequent value;
    # if a column is entirely missing, use 'Unknown'.
    for column in df.select_dtypes(include=["object", "category", "string"]).columns:
        if df[column].isna().any():
            mode_values = df[column].mode(dropna=True)
            fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
            df[column] = df[column].fillna(fill_value)

    # Label encode Sex as requested.
    if "Sex" in df.columns:
        encoder = LabelEncoder()
        df["Sex"] = encoder.fit_transform(df["Sex"].astype(str))
        mapping = {label: int(index) for index, label in enumerate(encoder.classes_)}
        print(f"\nSex LabelEncoder mapping: {mapping}")

    # One-hot encode Embarked as requested; drop the original column.
    if "Embarked" in df.columns:
        encoder = make_one_hot_encoder()
        encoded = encoder.fit_transform(df[["Embarked"]].astype(str))
        categories = encoder.categories_[0]
        encoded_names = [f"Embarked_{category}" for category in categories]
        encoded_df = pd.DataFrame(encoded, columns=encoded_names, index=df.index).astype(int)
        df = pd.concat([df.drop(columns=["Embarked"]), encoded_df], axis=1)
        print(f"Embarked one-hot columns created: {encoded_names}")

    # Final safety check: report any missing values still present.
    missing_after = int(df.isna().sum().sum())
    print("\n=== Missing values after cleaning ===")
    print(df.isnull().sum().sort_values(ascending=False).head(15))
    print(f"Total missing cells after cleaning: {missing_after}")

    cleaned_path = OUTPUT_DIR / "titanic_cleaned.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"\nCleaned dataset saved to: {cleaned_path}")
    print(f"Final dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
