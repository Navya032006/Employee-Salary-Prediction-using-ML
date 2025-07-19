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
```
---

## Screenshots

<p float="left">
  <img src="https://github.com/user-attachments/assets/e621249a-1624-43c3-a8b9-6ee3eed31106" width="320" />
  <img src="https://github.com/user-attachments/assets/8a1ef61f-b0e1-4f4a-b319-1f4e32d15bf3" width="320" />
  <img src="https://github.com/user-attachments/assets/e9503591-d082-4147-8ef4-019496e2f7aa" width="320" />
</p>

---

## 📊 Model Information

- **Algorithm**: Linear Regression  
- **Metric 1**: R² Score *(stored in `model_score.txt`)*  
- **Metric 2**: Mean Squared Error *(stored in `model_mse.txt`)*  
- **Categorical Encoding**: LabelEncoder  
- **Scaling**: StandardScaler

---

## ⚙️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Navya032006/Employee-Salary-Prediction-using-ML.git
   cd Employee-Salary-Prediction-using-ML
      
2. **Create a Virtual Environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
              
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
         
4. **Run the Application Locally**
   ```bash
   streamlit run app.py

---

## 🌐 Live Deployment
The project is deployed and available at:
👉 https://employee-salary-prediction-using-ml-bynavya.streamlit.app/

---

## 📦 Dataset Source
**Kaggle** :  [Salary Prediction for Beginners](https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer/data)
