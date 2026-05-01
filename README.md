# 🌊 Flood Early Warning & Response System
### Smart Cities Course Project

A machine-learning-powered flood detection and alert system built with Python, scikit-learn, XGBoost, and Streamlit.

---

## 📁 Project Structure

```
flood_warning_system/
├── data_loader.py      # Downloads & preprocesses the public Kaggle dataset
├── train.py            # ML training pipeline (Random Forest + XGBoost)
├── app.py              # Streamlit web dashboard (4 pages)
├── requirements.txt    # Python dependencies
├── data/               # Auto-created — cached dataset stored here
├── models/             # Auto-created — trained model files saved here
└── reports/            # Auto-created — evaluation plots saved here
```

---

## 📦 Dataset

**Source:** [Flood Prediction Factors — Kaggle](https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-factors)

- 20 environmental & socioeconomic risk features (each scored 1–10)
- Target: `FloodProbability` (0–1 continuous) → converted to binary label (≥0.5 = Flood)
- Over 1 million rows (we sample 50k for speed by default)

### Dataset features include:
| Feature | Description |
|---------|-------------|
| MonsoonIntensity | Seasonal rainfall intensity |
| TopographyDrainage | Natural terrain drainage capacity |
| RiverManagement | Quality of river management |
| Deforestation | Watershed deforestation level |
| Urbanization | Urban surface impervious coverage |
| ClimateChange | Climate variability impact |
| DamsQuality | Dam structural integrity |
| ... | (20 features total) |

---

## ⚙️ Setup Instructions

### 1. Clone or unzip this project
```bash
cd flood_warning_system
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up Kaggle API credentials (for automatic dataset download)

1. Go to https://www.kaggle.com → Account → API → "Create New Token"
2. This downloads a `kaggle.json` file
3. Place it here:
   - **Windows:** `C:\Users\<you>\.kaggle\kaggle.json`
   - **Mac/Linux:** `~/.kaggle/kaggle.json`

> **No Kaggle account?** No problem — if the download fails, the system automatically generates a synthetic fallback dataset with the same schema so everything still runs.

---

## 🚀 Running the Project

### Step 1 — Train the ML model
```bash
python train.py
```
This will:
- Download the dataset from Kaggle (or use the synthetic fallback)
- Train a Random Forest and an XGBoost classifier
- Evaluate both and save the best model to `models/`
- Save evaluation plots (confusion matrix, ROC curve, feature importance) to `reports/`

Expected output:
```
============================================================
  Flood Early Warning System — ML Training Pipeline
============================================================
[train] Train: 40,000 | Test: 10,000
[train] Training Random Forest...
  Accuracy : 0.9123
  F1 Score : 0.9118
  ROC-AUC  : 0.9702
...
[train] Best model: XGBoost (AUC = 0.9718)
[train] Done. Model saved to models/
```

### Step 2 — Launch the Streamlit dashboard
```bash
streamlit run app.py
```

Open your browser to: **http://localhost:8501**

> You can also train the model from inside the app — just click the **"Train Model Now"** button in the sidebar.

---

## 🖥️ Dashboard Pages

| Page | Description |
|------|-------------|
| **Dashboard** | Simulated alert levels (Normal/Watch/Warning/Emergency) with metrics and response actions |
| **Data Explorer** | Class balance, feature distributions, correlation heatmap |
| **Model Report** | Accuracy, F1, AUC-ROC, confusion matrix, feature importance |
| **Flood Predictor** | 20 interactive sliders → live ML flood probability prediction with risk gauge |

---

## 🤖 ML Pipeline

```
Raw Dataset
    │
    ▼
Preprocessing (StandardScaler, stratified split 80/20)
    │
    ├──► Random Forest Classifier (200 trees, max_depth=12)
    │
    └──► XGBoost Classifier (200 estimators, lr=0.1)
              │
              ▼
         Best model selected by ROC-AUC on test set
              │
              ▼
    Saved to models/best_model.pkl
```

---

## 📊 Expected Results

| Metric | Typical Value |
|--------|--------------|
| Accuracy | 90–93% |
| F1 Score | 0.90–0.93 |
| ROC-AUC | 0.96–0.98 |
| 5-Fold CV | 90–92% |

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Kaggle download fails | Check `~/.kaggle/kaggle.json` exists, or proceed with synthetic data |
| `streamlit: command not found` | Use `python -m streamlit run app.py` |
| Port 8501 in use | Run `streamlit run app.py --server.port 8502` |

---

## 📚 Technologies Used

- **Python 3.10+**
- **scikit-learn** — Random Forest, preprocessing, evaluation
- **XGBoost** — Gradient boosted trees
- **Streamlit** — Interactive web dashboard
- **Plotly** — Interactive charts
- **Pandas / NumPy** — Data manipulation
- **kagglehub** — Dataset download
