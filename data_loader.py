"""
data_loader.py
--------------
Downloads the public 'Flood Prediction Factors' dataset from Kaggle
(naiyakhalid/flood-prediction-factors) using kagglehub and preprocesses
it for the ML pipeline.

Dataset URL:
  https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-factors

If the Kaggle download fails (no API token set up), the script falls back
to generating a synthetic dataset with the same schema so the rest of the
project still runs for demo purposes.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Column reference ───────────────────────────────────────────────────────────
FEATURE_COLS = [
    "MonsoonIntensity",
    "TopographyDrainage",
    "RiverManagement",
    "Deforestation",
    "Urbanization",
    "ClimateChange",
    "DamsQuality",
    "Siltation",
    "AgriculturalPractices",
    "Encroachments",
    "IneffectiveDisasterPreparedness",
    "DrainageSystems",
    "CoastalVulnerability",
    "Landslides",
    "Watersheds",
    "DeterioratingInfrastructure",
    "PopulationScore",
    "WetlandLoss",
    "InadequatePlanning",
    "PoliticalFactors",
]
TARGET_COL = "FloodProbability"
BINARY_TARGET = "FloodLabel"   # 1 = Flood, 0 = No Flood (derived, threshold 0.5)
CACHE_PATH = "data/flood_data.csv"


# ── Download ───────────────────────────────────────────────────────────────────

def download_dataset() -> pd.DataFrame:
    """
    Try to pull the dataset from Kaggle via kagglehub.
    Falls back to synthetic data if credentials are missing.
    """
    os.makedirs("data", exist_ok=True)

    if os.path.exists(CACHE_PATH):
        print(f"[data_loader] Loading cached dataset from {CACHE_PATH}")
        return pd.read_csv(CACHE_PATH)

    # ── Attempt Kaggle download ────────────────────────────────────────────────
    try:
        import kagglehub
        print("[data_loader] Downloading from Kaggle (naiyakhalid/flood-prediction-factors)...")
        path = kagglehub.dataset_download("naiyakhalid/flood-prediction-factors")
        # Find the CSV inside the downloaded folder
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".csv"):
                    csv_path = os.path.join(root, f)
                    df = pd.read_csv(csv_path)
                    df.to_csv(CACHE_PATH, index=False)
                    print(f"[data_loader] Downloaded {len(df):,} rows → cached at {CACHE_PATH}")
                    return df
        raise FileNotFoundError("No CSV found in Kaggle download package.")

    except Exception as e:
        print(f"[data_loader] Kaggle download failed ({e}).")
        print("[data_loader] Generating synthetic fallback dataset (50,000 rows).")
        return _generate_synthetic(n=50_000)


def _generate_synthetic(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset that mirrors the Kaggle schema so the
    project runs even without Kaggle credentials.
    """
    rng = np.random.default_rng(seed)
    data = {col: rng.integers(1, 11, size=n).astype(float) for col in FEATURE_COLS}
    # FloodProbability correlates with several risk factors
    risk_score = (
        data["MonsoonIntensity"] * 0.25
        + data["TopographyDrainage"] * 0.15
        + data["Urbanization"] * 0.12
        + data["Deforestation"] * 0.10
        + data["ClimateChange"] * 0.10
        + data["DrainageSystems"] * 0.08
        + data["RiverManagement"] * (-0.08)   # good management reduces risk
        + data["DamsQuality"] * (-0.05)
        + rng.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-0.4 * (risk_score - 4.5)))   # sigmoid mapping
    prob = np.clip(prob, 0.01, 0.99)
    data[TARGET_COL] = np.round(prob, 4)
    df = pd.DataFrame(data)
    os.makedirs("data", exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    return df


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Cleans the DataFrame, derives a binary flood label, and returns
    train/test splits plus a fitted scaler.

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    scaler                            : fitted StandardScaler
    df_clean                          : cleaned DataFrame (with FloodLabel column)
    """
    df = df.copy()

    # Keep only known columns; drop rows with missing values
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    df = df[available_features + [TARGET_COL]].dropna()

    # Clip feature values to valid range [1, 10]
    for col in available_features:
        df[col] = df[col].clip(1, 10)

    # Derive binary label
    df[BINARY_TARGET] = (df[TARGET_COL] >= 0.5).astype(int)

    X = df[available_features].values
    y = df[BINARY_TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, df, available_features


if __name__ == "__main__":
    df = download_dataset()
    print(df.head())
    print(df.shape)
    print(df[TARGET_COL].describe())
