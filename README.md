# 🩺 Breast Cancer Prediction using Machine Learning  

## 📌 Project Overview  
This project aims to develop a **machine learning model** that predicts whether a tumor is **malignant** or **benign** based on medical features. The dataset used is the **Breast Cancer Wisconsin Dataset** from `sklearn.datasets`.  

## 🛠️ Steps in the Project  

### 1️⃣ Data Loading & Preparation  
- Loaded the **Breast Cancer Wisconsin Dataset** and converted it into a `pandas.DataFrame`.  
- Separated **features (x)** and the **target variable (y)**.  

### 2️⃣ Exploratory Data Analysis (EDA)  
- Checked dataset dimensions (**569 samples, 30 features**).  
- Verified **missing values** (none found).  
- Used `describe()` to analyze key statistics such as mean and standard deviation.  

### 3️⃣ Splitting Data into Training & Testing Sets  
- Used `train_test_split()` to split data into **80% training** and **20% testing**.  

### 4️⃣ Training a Random Forest Classifier  
- Trained an initial **Random Forest Model** (`RandomForestClassifier`) with default hyperparameters.  
- **Initial Accuracy:** **96.49%**  

### 5️⃣ Comparing Different Models  
- Tested **Logistic Regression** and **SVC (Support Vector Classifier)** alongside Random Forest.  
- **Results:**  
  - **Random Forest**: **96.49%**  
  - **Logistic Regression**: **95.61%**  
  - **SVC**: **94.74%**  

### 6️⃣ Feature Importance Analysis  
- Identified key features contributing to the prediction.  
- Most important features:  
  - **Concave points (worst)**  
  - **Area (worst)**  
  - **Radius (worst)**  
  - **Concave points (mean)**  

### 7️⃣ Hyperparameter Tuning with GridSearchCV  
- Used `GridSearchCV` to find the best **Random Forest parameters**:  
  - `n_estimators = 150`  
  - `max_depth = None`  
- **Optimized Accuracy:** **96.26%**  

### 8️⃣ Code Optimization with Classes & Functions  
- Refactored the code into a structured **class (`BreastCancerPrediction`)** for better readability and maintainability.  

---

## 📌 Key Results  
✅ The trained **Random Forest model** achieves an accuracy of **~96%**, making it highly reliable for breast cancer classification.  
✅ **Feature Importance Analysis** helped identify the most relevant medical features.  
✅ **Hyperparameter Tuning** slightly improved the model’s performance.  

---

## 🚀 Future Enhancements  
- **User Input Feature**: Allow users to enter their own data and get predictions.    
- **Further Model Optimization**: Try **deep learning (e.g., neural networks)** for comparison.  
- **Apply to Other Medical Datasets** to test generalization.  

---

## 📂 Installation & Usage  
To run this project locally, follow these steps:  

### 🔹 Install dependencies  
```bash
pip install pandas scikit-learn matplotlib