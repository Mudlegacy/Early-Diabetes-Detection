# Predictive Modeling for Early Diabetes Detection


<!-- This is a generic banner, you can create a custom one! -->

## 📖 Overview

This project, completed by **Team DataVerse**, focuses on the early detection of diabetes. Using the Pima Indians Diabetes Dataset, we performed comprehensive Exploratory Data Analysis (EDA), data preprocessing, and feature engineering to build a robust classification model. Our final model, a tuned **Random Forest Classifier**, successfully predicts whether a person has diabetes with an **accuracy of 79%** and a strong **AUC score of 0.84**, prioritizing the correct identification of diabetic patients (high recall).

---

## 🎯 Problem Statement

A mobile health clinic aims to pre-screen patients for diabetes using basic health indicators. The goal is to reduce hospital crowding and focus medical resources on at-risk individuals. This project addresses the problem by:
1.  Using EDA to uncover the factors that most significantly influence the risk of diabetes.
2.  Developing a reliable classification model to predict a patient's diabetes status based on attributes like BMI, Glucose levels, and age.

---

## 💾 Dataset

The project utilizes the **Pima Indians Diabetes Dataset**, a classic dataset for binary classification problems in healthcare.

*   **Source:** Originally from the National Institute of Diabetes and Digestive and Kidney Diseases, widely available on platforms like Kaggle.
*   **Description:** The dataset contains 768 patient records with 9 columns. The target variable, `Outcome`, is binary (1 for diabetic, 0 for non-diabetic).
*   **Key Features:** `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `BMI`, `DiabetesPedigreeFunction`, and `Age`.

---

## 🛠️ Project Pipeline

Our approach to solving this problem followed a structured pipeline:

1.  **Data Cleaning & Preprocessing:**
    *   **Missing Value Imputation:** Identified that key features like `Glucose`, `BMI`, and `BloodPressure` had `0` values, which are physiologically impossible. These were treated as missing data and imputed using the **median** value of each column to ensure robustness against outliers.
    *   **Outlier Handling:** Applied **Winsorization** to cap extreme values in key features, preventing them from skewing the model.

2.  **Exploratory Data Analysis (EDA):**
    *   Analyzed feature distributions and found the dataset was **imbalanced**, with significantly more non-diabetic patients than diabetic ones.
    *   Confirmed through correlation heatmaps and boxplots that **Glucose** is the most significant predictor of diabetes.
    *   Identified moderate correlations between the outcome and `BMI` and `Age`.

3.  **Feature Engineering & Selection:**
    *   Created a new categorical feature, `Insulin_Category` (Low, Medium, High), to better capture the impact of insulin levels on diabetes.
    *   Used **Recursive Feature Elimination (RFE)** to identify the most impactful features for the model.

4.  **Modeling & Evaluation:**
    *   **Initial Modeling:** Trained and compared three different models: Logistic Regression, Random Forest, and Support Vector Machine (SVM).
    *   **Hyperparameter Tuning:** Used `GridSearchCV` to find the optimal parameters for the top-performing models (Random Forest and Logistic Regression).
    *   **Addressing Class Imbalance:** To improve the model's ability to identify diabetic patients (class 1), we trained a Random Forest model with `class_weight="balanced"`.
    *   **Threshold Tuning:** Further refined the balanced model by adjusting the prediction threshold to **0.45**, which increased the **recall for the diabetic class to 80%**. This is critical for minimizing false negatives in a medical context.

---

## ✨ Results & Findings

*   **Final Model:** The **Balanced Random Forest Classifier** with an adjusted threshold was selected as the final model.
*   **Performance:**
    *   **Accuracy:** 76% (on the threshold-adjusted model)
    *   **AUC Score:** 0.84
    *   **Recall (Class 1 - Diabetic):** **80%** — Our model correctly identifies 80% of all diabetic patients.
*   **Key Feature Importance:** `Glucose`, `BMI`, and `Age` were consistently identified as the most important features for prediction.

---

## 🚀 How to Run this Project

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd <repo-name>
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install the required libraries:**
    *(Create a `requirements.txt` file with the libraries from your notebook)*
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook Capstone_Project.ipynb
    ```

---

## 💻 Technologies Used

*   **Programming Language:** Python
*   **Libraries:**
    *   Pandas & NumPy
    *   Matplotlib & Seaborn
    *   Scikit-learn (for modeling, preprocessing, and evaluation)
    *   Joblib (for model saving)

---

### 📞 Contact

*   **Team:** Team DataVerse
*   **Project Lead:** ALMUSTAPHA DAMILOLA USMAN- LinkedIn : www.linkedin.com/in/mudlegacy - Email : usmandmustapha3@gmail.com
*   **Project Link:** [Link to this GitHub repository]
