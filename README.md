# 📊 Customer Churn Prediction & Analytics Dashboard

An end-to-end Machine Learning project that analyzes customer behavior and predicts whether a customer is likely to churn.

The project uses **Python, Pandas, Scikit-learn, Matplotlib, Seaborn, Joblib, and Streamlit**.

## 🚀 Project Overview

Customer churn means a customer stops using a company's products or services.

This project uses historical customer data to:

* Analyze customer behavior
* Identify churn patterns
* Visualize important customer metrics
* Train a Machine Learning model
* Predict customer churn probability
* Provide an interactive Streamlit dashboard

## 📁 Dataset

The dataset contains the following columns:

| Column            | Description                               |
| ----------------- | ----------------------------------------- |
| `Names`           | Customer name                             |
| `Age`             | Customer age                              |
| `Total_Purchase`  | Total customer purchase                   |
| `Account_Manager` | Whether an account manager is assigned    |
| `Years`           | Years with the company                    |
| `Num_Sites`       | Number of customer sites                  |
| `Onboard_date`    | Customer onboarding date                  |
| `Location`        | Customer location                         |
| `Company`         | Customer company                          |
| `Churn`           | Target variable indicating customer churn |

## 🤖 Machine Learning

The project uses a **Random Forest Classifier**.

### Machine Learning workflow

```
Customer CSV Data
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Train/Test Split
       ↓
Data Preprocessing
       ↓
Random Forest Classifier
       ↓
Model Evaluation
       ↓
Save Trained Model
       ↓
Streamlit Prediction App
```

## 📊 Dashboard Features

### 🏠 Dashboard

Displays:

* Total customers
* Churned customers
* Churn rate
* Average customer years
* Customer data preview

### 📋 Data Explorer

Allows users to:

* View the complete dataset
* Search customer/company/location data
* Check missing values
* View statistical information

### 📈 Churn Analysis

Provides visual analysis of:

* Churn distribution
* Churn vs customer years
* Purchase vs churn
* Number of sites vs churn
* Churn rate by location

### 🔮 Predict Churn

Users can enter customer information and get:

* Churn prediction
* Churn probability
* Risk level
* Customer retention indication

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit

## 📂 Project Structure

```
Customer_Churn_Project/
│
├── data/
│   └── customer_churn.csv
│
├── models/
│   └── churn_model.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
```

Go into the project directory:

```bash
cd customer-churn-prediction
```

Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

## 🧠 Train the Model

Run:

```bash
python train_model.py
```

This will train the Random Forest model and create:

```
models/churn_model.pkl
```

## 🌐 Run the Streamlit Application

Run:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 📈 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The exact performance metrics are generated when `train_model.py` is executed on the dataset.

## 🎯 Business Objective

The goal of this project is to help businesses identify customers who may be at risk of leaving.

By identifying high-risk customers, a company can potentially:

* Improve customer retention
* Offer targeted promotions
* Improve customer support
* Identify problematic customer segments
* Reduce customer churn

## 👨‍💻 Author

**Aditya Roy**

Machine Learning & Data Analytics Project

---

⭐ If you found this project useful, consider giving the repository a star!
