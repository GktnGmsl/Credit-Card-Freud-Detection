# Credit Card Fraud Detection

Kredi kartı dolandırıcılık tespiti projesi. PCA ile dönüştürülmüş **284,807 işlem** üzerinde 3 farklı makine öğrenimi modeli eğitilmiş, karşılaştırılmış, **SHAP** ile açıklanmış, **FastAPI** ile servis edilmiş ve **Streamlit** dashboard ile izlenebilir hale getirilmiştir.

---

## Proje Mimarisi

```
 ┌──────────────┐     ┌────────────────────┐     ┌────────────────────┐
 │  creditcard   │────▸│   Preprocessing    │────▸│   Training &       │
 │  .csv (284K)  │     │  (Feature Eng.,    │     │   Comparison       │
 │               │     │   SMOTE, Scaling)  │     │  (LR, RF, XGB)    │
 └──────────────┘     └────────────────────┘     └────────┬───────────┘
                                                          │
                          ┌───────────────────────────────┤
                          │                               │
                          ▼                               ▼
                ┌──────────────────┐           ┌────────────────────┐
                │  Optuna          │           │   MLflow Tracking  │
                │  (Bayesian Opt.) │           │   & Model Registry │
                └────────┬─────────┘           └────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Optimized RF Model  │
              │  (outputs/models/)   │
              └──────────┬───────────┘
                         │
              ┌──────────┴───────────┐
              │                      │
              ▼                      ▼
    ┌──────────────────┐   ┌──────────────────┐
    │   FastAPI (REST) │   │  Streamlit       │
    │   POST /predict  │◀──│  Dashboard       │
    │   GET  /health   │   │  (Monitoring &   │
    │   :8000          │   │   Live Test)     │
    └──────────────────┘   │   :8501          │
                           └──────────────────┘
```

---

## Proje Yapısı

```
├── Data/
│   ├── creditcard.csv                # Ham veri (284,807 satır × 31 sütun)
│   └── processed/
│       ├── train_smote.parquet       # SMOTE ile dengelenmiş eğitim seti (398K satır)
│       ├── train_original.parquet    # Orijinal eğitim seti (dengelenmemiş)
│       ├── val.parquet               # Validation seti (%15)
│       └── test.parquet              # Test seti (%15)
├── eda.ipynb                         # Keşifsel Veri Analizi
├── preprocessing.ipynb               # Veri ön işleme (Feature Engineering, SMOTE, Scaling)
├── model_comparison.ipynb            # Model eğitimi ve karşılaştırma notebook'u
├── explainability.ipynb              # SHAP tabanlı model açıklanabilirliği
├── training.py                       # Eğitim/değerlendirme/Optuna optimizasyon pipeline
├── mlflow_tracking.py                # MLflow experiment tracking & model registry
├── api/
│   ├── __init__.py
│   ├── main.py                       # FastAPI uygulaması (POST /predict, GET /health)
│   └── schemas.py                    # Pydantic request/response şemaları
├── dashboard/
│   └── app.py                        # Streamlit monitoring dashboard
├── outputs/                          # Grafikler ve kaydedilmiş modeller
│   ├── models/                       # Joblib model dosyaları + scaler + model_config.json
│   ├── cm_*.png                      # Confusion matrix görselleri
│   ├── roc_*.png / pr_*.png          # ROC & PR eğri görselleri
│   ├── loss_*.png                    # Loss curve görselleri (train + validation)
│   ├── model_comparison.csv          # 3 baseline model metrikleri
│   └── final_comparison.csv          # 4 model metrikleri (optimized dahil)
├── mlruns/                           # MLflow tracking verileri & model registry
├── Dockerfile.api                    # API Docker image
├── Dockerfile.dashboard              # Dashboard Docker image
├── docker-compose.yml                # Çoklu servis orkestrasyon
├── .dockerignore                     # Docker build exclusions
├── .vscode/settings.json             # VS Code ayarları (Python interpreter)
├── requirements.txt                  # Tüm Python bağımlılıkları
├── requirements-api.txt              # API için minimal bağımlılıklar
├── requirements-dashboard.txt        # Dashboard için minimal bağımlılıklar
└── README.md
```

---

## Modeller

| Model | Açıklama |
|-------|----------|
| Logistic Regression | Baseline model, saga solver, C=1.0, max_iter=200 |
| Random Forest | 200 trees, OOB score, en iyi baseline (PR-AUC: 0.8449) |
| XGBoost | max_depth=6, 200 boosting rounds, logloss objective |
| Random Forest (Optimized) | Optuna + **Stratified 5-Fold CV**, 50 trial, PR-AUC maximization |

### Optuna Hiperparametre Optimizasyonu

- **Değerlendirme:** Stratified 5-Fold CV (`train_original.parquet` üzerinde, her fold ~370 fraud örneği)
- **Arama alanı:** max_depth ∈ {None, 10, 20, 30, 40, 50}, n_estimators ∈ [100, 500], min_samples_split ∈ [2, 15], min_samples_leaf ∈ [1, 8]
- **Trial sayisi:** 50 (timeout: 60 dk)
- **Objective:** Ortalama PR-AUC (5-fold)
- **Sonuç:** max_depth=30, n_estimators=403, min_samples_split=11, min_samples_leaf=1, max_features=sqrt

---

## Model Performans Tablosu (Test Seti)

| Model | ROC-AUC | PR-AUC | Accuracy | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | Best Threshold |
|-------|---------|--------|----------|-------------------|----------------|------------|----------------|
| Logistic Regression | 0.9767 | 0.7902 | 0.9762 | 0.0597 | 0.8649 | 0.1117 | 0.99 |
| Random Forest | 0.9754 | 0.8449 | 0.9995 | 0.9077 | 0.7973 | 0.8489 | 0.51 |
| XGBoost | 0.9681 | 0.8280 | 0.9991 | 0.7229 | 0.8108 | 0.7643 | 0.94 |
| **RF (Optimized)** 🏆 | **0.9807** | **0.8409** | **0.9995** | **0.8824** | **0.8108** | **0.8451** | **0.52** |

> **Not:** PR-AUC, dengesiz veri setleri için ROC-AUC'den daha bilgilendiricidir. Optimized RF modeli en yüksek ROC-AUC'ye (0.9807) sahiptir ve Stratified 5-Fold CV ile daha güvenilir hiperparametre seçimi yapılmıştır. Threshold optimizasyonu ile F1 skoru 0.8511'e ulaşmıştır.

---

## Değerlendirme Metrikleri

- Confusion Matrix (validation + test ayrı ayrı)
- Classification Report (precision, recall, F1 — fraud sınıfı odaklı)
- Loss Curves (train + validation)
- ROC-AUC Curve
- Precision-Recall Curve (imbalanced data için daha bilgilendirici)
- Threshold Optimization (F1-maximizing threshold)
- Stratified 5-Fold CV (Optuna hiperparametre seçimi için)

---

## Kurulum

### Seçenek A — Yerel Kurulum

```bash
# 1. Depoyu klonlayın
git clone <repo-url>
cd "Credit Card Freud Detection"

# 2. Bağımlılıkları kurun
pip install -r requirements.txt
```

### Seçenek B — Docker ile Kurulum

```bash
# Tek komutla API + Dashboard ayağa kalkar
docker-compose up --build
```

| Servis | Adres |
|--------|-------|
| FastAPI (Swagger) | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |

---

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
mlflow ui
```
Tarayıcıdan [http://127.0.0.1:5000](http://127.0.0.1:5000) adresine gidin.

### 5. Model Açıklanabilirliği (SHAP)
`explainability.ipynb` notebook'unu açıp hücreleri sırayla çalıştırın.

İçerikler:
- **Global feature importance** — Bar plot + beeswarm plot (ortalama |SHAP| değerleri)
- **Lokal açıklamalar** — 2 fraud + 2 normal işlem için waterfall plot

### 6. REST API (FastAPI)
```bash
uvicorn api.main:app --reload
```
Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 7. Monitoring Dashboard (Streamlit)
```bash
streamlit run dashboard/app.py
```
Tarayıcıdan [http://localhost:8501](http://localhost:8501) adresine gidin.

Dashboard bölümleri:
- **Model Metrikleri** — AUC, Precision, Recall, F1 kartları + MLflow registry versiyon, threshold, optimized params gösterimi
- **Confusion Matrix** — Model seçerek confusion matrix görseli
- **ROC & PR Eğrileri** — Birleşik ve bireysel eğri grafikleri
- **SHAP Feature Importance** — Global bar plot + beeswarm plot
- **Canlı Test Paneli** — Değer girerek API'ye istek gönderme (model versiyonu + threshold otomatik gösterilir)

---

## API Endpoint'leri

| Metot | Yol | Açıklama |
|-------|-----|----------|
| `GET` | `/health` | Sağlık kontrolü |
| `POST` | `/predict` | Fraud tahmini |
| `GET` | `/docs` | Swagger UI (otomatik) |

### API Kullanım Örnekleri (curl)

**Sağlık Kontrolü:**
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "model_version": "4"}
```

**Fraud Tahmini:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
      "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
      "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
      "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
      "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
      "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": -0.34,
      "V25": -0.07, "V26": -0.06, "V27": -0.03, "V28": -0.01,
      "Amount": 149.62, "Time": 0.0
    }
  }'
```
```json
{"is_fraud": false, "fraud_probability": 0.0312, "risk_level": "LOW"}
```

---

## Tasarım Kararları ve Gerekçeleri

### Neden SMOTE?
Veri seti aşırı dengesizdir (%0.172 fraud). Modeller eğitim sırasında azınlık sınıfını görmezden gelebilir. **SMOTE** (Synthetic Minority Over-sampling Technique), sentetik fraud örnekleri üreterek eğitim setini 1:1 oranına dengelemiştir. Sadece eğitim setine uygulanmış olup validation/test setlerinin gerçek dağılımı korunmuştur.

### Neden Random Forest Kazandı?
- **PR-AUC** (dengesiz veri için en anlamlı metrik) bazında Random Forest (0.8449) tüm baseline'ları geçmiştir
- XGBoost (0.8280) ile yakın olsa da, RF'nin **precision-recall dengesi** belirgin şekilde daha iyidir (Precision: 0.9077 vs 0.7229)
- Logistic Regression yüksek recall'a sahip ancak çok düşük precision ile pratik kullanılabilirliği kısıtlıdır

### Neden Optuna ile Bayesian Optimizasyon?
Grid/random search yerine **Optuna** kullanılmasının sebebi:
- Bayesian optimizasyon, hiperparametre uzayını daha verimli araştırır
- **Stratified 5-Fold CV** ile `train_original.parquet` üzerinde değerlendirme yapılır (her fold ~370 fraud örneği — tek val split'teki 74'ten 5x fazla)
- SMOTE verisi yerine orijinal veri kullanılır — sentetik örneklerin hem train hem val fold'una düşmesi engellenir
- 50 trial ile yeterli yakınsama sağlanmıştır (timeout: 60 dk)
- PR-AUC objective ile dengesiz veri senaryosu için optimize edilmiştir
- `max_depth` arama alanı: {None, 10, 20, 30, 40, 50} — sınırsız derinlik de denenir
- Sonuç: ROC-AUC 0.9754 → **0.9807** yükselmiştir

### Neden Threshold = 0.52?
Varsayılan 0.50 threshold yerine, F1-maximizing threshold aranmıştır. Optimized RF için **0.52** değeri precision-recall trade-off'unun optimal noktasıdır. Threshold, `model_config.json` dosyasına kaydedilir ve API startup'ta dinamik olarak yüklenir.

### Neden SHAP?
- Tree-based modeller için `TreeExplainer` kullanarak **kesin SHAP değerleri** hesaplanabilir (yaklaşık değil)
- Her işlem için "neden fraud?" sorusu yanıtlanabilir (lokal açıklanabilirlik)
- Global feature importance ile hangi PCA bileşenlerinin dolandırıcılık kararını yönlendirdiği anlaşılır
- Regülasyon gereksinimleri (GDPR "right to explanation") için uygun

### Docker Mimarisi
- **Ayrı Dockerfile'lar**: API ve Dashboard bağımsız scale edilebilir
- **Ayrı requirements**: `requirements-api.txt` ve `requirements-dashboard.txt` ile minimal Docker image boyutu
- **docker-compose**: Tek komutla tüm servisler ayağa kalkar; Dashboard health-check ile API'nin hazır olmasını bekler
- **Ham veri hariç**: `.dockerignore` ile 150MB'lık `creditcard.csv` image'dan çıkarılmış, sadece processed parquet dosyaları dahil edilmiştir
- **MLflow registry**: Her iki container'a `mlruns/` kopyalanır — model versiyonu dinamik olarak okunur

### Dinamik Versiyon Yönetimi
- `training.py` çalıştırıldığında modeller + `model_config.json` (threshold, best_params) diske kaydedilir
- `mlflow_tracking.py` çalıştırıldığında yeni bir MLflow registry versiyonu oluşturulur ve `production` alias güncellenir
- API ve Dashboard startup'ta MLflow registry'den güncel versiyonu ve `model_config.json`'dan threshold'u otomatik yükler
- Hiçbir hardcoded versiyon veya threshold yoktur

---

## Veri Seti

- **Kaynak**: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Toplam**: 284,807 işlem, %0.172 fraud (492 fraud / 284,315 normal)
- **Features**: V1-V28 (PCA), Time, Amount + 5 mühendislik özelliği (Time_in_day, Amount_log, Time_Amount, Time_Amount_sq, Amount_per_Time)
- **Train/Val/Test**: %70/%15/%15 stratified split
- **Dengeleme**: SMOTE (sadece train setine, 1:1 orana)
