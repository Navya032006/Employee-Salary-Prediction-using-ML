# 💼 Employee Salary Prediction Web Application 

A **Machine Learning-powered web application** that accurately predicts employee salaries based on inputs such as age, gender, education level, job title, and years of experience. Designed with a **modern UI using Streamlit**, and trained using **Linear Regression** from scikit-learn.

---

## 🚀 Features

- Predict salary using multiple input factors
- Interactive and user-friendly **Streamlit UI**
- Real-time display of:
  - Predicted Annual Salary
  - Monthly Salary
  - Hourly Rate
- Visual display of model **R² Score**
- Model caching for fast and consistent predictions
- Clean deployment via **Streamlit Cloud**

---

## 🧾 Input Features

- Age  
- Gender  
- Education Level  
- Job Title  
- Years of Experience

---

## 🛠 Tech Stack

| Layer           | Tools Used                           |
|------------------|--------------------------------------|
| **UI**           | Streamlit, HTML/CSS                  |
| **ML Model**     | Scikit-learn (Linear Regression)     |
| **Data Handling**| Pandas, NumPy                        |
| **Encoding**     | LabelEncoder                         |
| **Scaling**      | StandardScaler                       |
| **Deployment**   | Streamlit Cloud                      |

---

## 📁 Project Structure

**Overview**
```bash
Employee-salary-prediction/
├── data/Salary Data.csv          # Dataset
├── Employee-Salary-Model.ipynb   # Model training notebook
├── app.py                        # Streamlit web application
├── evaluation_plot.png           # Evaluation plot of model performance
├── model_mse.txt                 # Mean Squared Error score
├── model_score.txt               # R² Score of the model
├── requirements.txt              # Python dependencies
└── salary_predictor.pkl          # Trained model (pickle)

---
