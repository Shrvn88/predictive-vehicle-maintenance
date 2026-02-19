# 🚗 Predictive Vehicle Maintenance System (RUL Prediction)

End-to-end Machine Learning system for predicting **Remaining Useful Life (RUL)** of vehicle engines using NASA CMAPSS data.

This project demonstrates a **production-style ML pipeline** including:

- Feature engineering
- Model training
- REST API inference
- Interactive dashboard
- Health monitoring visualization

---
## 🎉 Output
<img width="1918" height="970" alt="image" src="https://github.com/user-attachments/assets/4aaee6cb-4267-4c9f-acc1-891cec303345" />
<img width="1919" height="793" alt="image" src="https://github.com/user-attachments/assets/cdbbca21-d137-467e-808e-0c8d6af2fc89" />
<img width="1919" height="793" alt="image" src="https://github.com/user-attachments/assets/f3edcc1d-bcab-4c61-a8d6-316404eeeb19" />
<img width="1919" height="448" alt="image" src="https://github.com/user-attachments/assets/bc48eeb1-4694-4830-a2b8-21aba90dfcb0" />



---

## 📌 Key Features

✅ Trained on NASA CMAPSS run-to-failure dataset  
✅ Gradient Boosting regression model  
✅ Rolling window feature engineering  
✅ FastAPI inference service  
✅ Streamlit dashboard  
✅ RUL degradation visualization  
✅ Engine health estimation  

---

## 🏗 Architecture
    
          Telemetry CSV
                ↓
        Feature Engineering
                ↓
    ML Model (Gradient Boosting)
                ↓
     FastAPI Prediction Service
                ↓
        Streamlit Dashboard


---

## 📊 Dataset

Uses NASA CMAPSS turbofan engine degradation dataset.

Each engine is run until failure, enabling true RUL supervision.

---


## EDA Report:

Dataset characteristics

    Multivariate run-to-failure engine telemetry

    Variable lifetimes per unit

    Smooth degradation trajectories

Key preprocessing decisions

    Removed constant sensors

    Selected sensors with RUL correlation

    RUL computed per unit using final cycle

Observed degradation

    Certain sensors show monotonic drift

    Variance increases near failure

    RUL decreases approximately linearly

---

## 🚀 How To Run

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Train model
```bash
python -m src.models.train
```
3. Start FastAPI server
```bash
uvicorn api.main:app --reload

API available at:

http://127.0.0.1:8000/docs
```
4. Start Streamlit dashboard
```bash
streamlit run dashboard/app.py

Dashboard:

http://localhost:8501
```

---

## 📈 Dashboard

- Upload telemetry CSV

- View raw sensor preview

- Generate RUL predictions

- See degradation curve

- Engine health status

---

## 🧠 ML Approach

- Rolling mean + std features

- GradientBoostingRegressor

- RUL smoothing for visualization

- Health score computed from predicted RUL

---

## 🛠 Tech Stack

- Python

- Pandas / NumPy

- Scikit-learn

- FastAPI

- Streamlit

- Matplotlib

- Joblib
