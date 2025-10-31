# <h2 align = "center">✈️ Flight Price Prediction Web App</h2>  

[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Model-XGBoost%20|%20RandomForest-brightgreen)](https://xgboost.ai/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue?logo=kaggle)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🧠 About the Project

**Flight Price Prediction App** is a **production-ready machine learning web application** that predicts flight ticket prices based on multiple features such as airline, source, destination, stops, and travel time.

This project demonstrates the **complete ML lifecycle** — from **data preprocessing and feature engineering** to **model optimization** and **deployment with Streamlit**.

> 💡 Built by [**Batuhan Başoda**](https://www.linkedin.com/in/batuhan-ba%C5%9Foda-b78799377/) — using the [Kaggle Flight Price Dataset] and deployed as a modern ML web app.

---

## 🚀 Features

- 🧩 **Data Preprocessing:** Missing value handling, encoding, and feature extraction  
- ⚙️ **Feature Engineering:** Time-based and categorical transformations  
- 🧠 **Modeling:** Trained and compared multiple regression models (RandomForest, XGBoost, etc.)  
- 🎯 **Hyperparameter Tuning:** Used RandomizedSearchCV for best model performance  
- 📊 **Real-time Predictions:** User inputs flight details → instant price prediction  
- 🌐 **Streamlit UI:** Deployed as an interactive, fast, and responsive web app  

---

## 🛠️ Tech Stack

| Category | Technology |
|-----------|-------------|
| **Language** | Python & Jupyter Notebook & Anaconda Environment|
| **ML Libraries** | Scikit-learn, XGBoost, NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Web Framework** | Streamlit |
| **Deployment** | Streamlit Cloud |
| **Dataset** | Kaggle Flight Price Dataset |

---

## 🖥️ App Preview

<p align="center">
  <img src="images/app_screenshot.png" width="700" alt="Flight Price Prediction App Screenshot">
</p>

🔗 **Live Demo:** [https://your-streamlit-app-link.streamlit.app](https://flightprice-5bbdyzcpalqtrsovwgtjvw.streamlit.app/)

---

## 🧮 Model Overview

- Data split: **80% training / 20% testing**
- Model evaluation metrics:
  - **R² Score:** 0.93  
  - **MAE:** 1,809  
  - **RMSE:** 3,913  
- Final model used: **Decision Tree Regressor** (tuned with RandomizedSearchCV)

---

## 🧰 How to Run Locally

```bash
# 1️⃣ Clone this repo
git clone https://github.com/batuhanbasoda/flight-price-predictor.git](https://github.com/Batuhan-METU/FlightPricePredictor-MachineLearningModel
cd flight-price-predictor

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run the app
streamlit run app.py
