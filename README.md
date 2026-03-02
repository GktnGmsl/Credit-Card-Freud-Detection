# Credit Card Fraud Detection

Kredi kartı dolandırıcılık tespiti projesi. PCA ile dönüştürülmüş 284,807 işlem üzerinde 3 farklı makine öğrenimi modeli eğitilmiş ve karşılaştırılmıştır.

## Proje Yapısı

```
├── Data/
│   ├── creditcard.csv              # Ham veri (284,807 satır × 31 sütun)
│   └── processed/
│       ├── train_smote.parquet     # SMOTE ile dengelenmiş eğitim seti (398K satır)
│       ├── train_original.parquet  # Orijinal eğitim seti (dengelenmemiş)
│       ├── val.parquet             # Validation seti (%15)
│       └── test.parquet            # Test seti (%15)
├── eda.ipynb                       # Keşifsel Veri Analizi
├── preprocessing.ipynb             # Veri ön işleme (Feature Engineering, SMOTE, Scaling)
├── model_comparison.ipynb          # Model eğitimi ve karşılaştırma notebook'u
├── training.py                     # Eğitim/değerlendirme yardımcı fonksiyonlar
├── mlflow_tracking.py              # MLflow experiment tracking & model registry
├── outputs/                        # Grafikler ve kaydedilmiş modeller
│   ├── models/                     # Joblib model dosyaları
│   └── *.png                       # Confusion matrix, ROC, PR, loss grafikleri
├── mlruns/                         # MLflow tracking verileri
└── README.md
```

## Modeller

| Model | Açıklama |
|-------|----------|
| Logistic Regression | Baseline model, saga solver |
| Random Forest | 200 trees, OOB score |
| XGBoost | 300 boosting rounds, logloss |
| Optimized (Optuna) | PR-AUC'si en yüksek modelin Bayesian optimizasyonu |

## Değerlendirme Metrikleri

- Confusion Matrix (validation + test ayrı ayrı)
- Classification Report (precision, recall, F1 — fraud sınıfı odaklı)
- Loss Curves (train/validation/test)
- ROC-AUC Curve
- Precision-Recall Curve (imbalanced data için daha bilgilendirici)
- Threshold Optimization (F1-maximizing threshold)

## Kurulum

```bash
pip install scikit-learn xgboost optuna mlflow pandas numpy matplotlib seaborn pyarrow
```

## Kullanım

### 1. Model Eğitimi (terminal/script)
```bash
python training.py
```

### 2. Model Eğitimi (notebook)
`model_comparison.ipynb` dosyasını Jupyter/VS Code'da açıp hücreleri sırayla çalıştırın.

### 3. MLflow Tracking
```bash
python mlflow_tracking.py
```

### 4. MLflow UI
```bash
cd "proje_dizini"
mlflow ui
```
Tarayıcıdan [http://127.0.0.1:5000](http://127.0.0.1:5000) adresine gidin.

## MLflow UI Ekran Görüntüsü

> `mlflow ui` komutunu çalıştırdıktan sonra ekran görüntüsünü buraya ekleyin.

![MLflow UI](outputs/mlflow_ui_screenshot.png)

## Veri Seti

- **Kaynak**: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Toplam**: 284,807 işlem, %0.172 fraud
- **Features**: V1-V28 (PCA), Time, Amount + 5 mühendislik özelliği
- **Train/Val/Test**: %70/%15/%15 stratified split
- **Dengeleme**: SMOTE (sadece train setine)
