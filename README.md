# Fake Job / Internship Scam Detection System

## Overview
This project is an **Ensemble and Explainable Machine Learning System** designed to detect fake job and internship postings.  
It combines **multiple machine learning models, rule-based logic, and explainability techniques** to provide accurate and reliable predictions.

The system allows users to:
- Enter job descriptions manually
- Upload files (TXT, PDF, DOCX)
- Get prediction with confidence score and reasoning

## Features

- ✅ Multi-Model Machine Learning (Logistic Regression, Naive Bayes, SVM, Random Forest)
- ✅ Ensemble Learning (Voting Classifier - Soft Voting)
- ✅ Hyperparameter Tuning (GridSearchCV)
- ✅ Hybrid System (ML + Rule-Based)
- ✅ Explainable AI (Reason for prediction)
- ✅ Confidence Score Output
- ✅ File Upload Support (TXT, PDF, DOCX)
- ✅ Interactive Web Interface (Streamlit)
- ✅ Fast Inference using Pickle (No retraining during runtime)

## Machine Learning Approach

### Data Processing
- Combined multiple datasets
- Cleaned and preprocessed text
- Removed stopwords and unwanted characters
- Converted text into numerical form using **TF-IDF Vectorization**

### Models Used
- Logistic Regression (with hyperparameter tuning)
- Multinomial Naive Bayes
- Support Vector Machine (SVM)
- Random Forest

### Ensemble Learning
- Combined models using **Voting Classifier (Soft Voting)**
- Uses probability-based predictions for better accuracy

### Hybrid System
Final prediction is based on:
- Machine Learning Probability  
- Rule-Based Scoring  
- Combined Decision Logic  

## Technologies Used

- Python
- Scikit-learn
- Pandas, NumPy
- Streamlit (Frontend)
- PyPDF2 (PDF text extraction)
- python-docx (DOCX text extraction)

## Project Structure

Fake_Job_Scam_Detection/
│
├── app.py # Streamlit frontend
├── model.py # Model training and saving
├── hybrid_predict.py # Prediction logic (ML + Rules)
├── rules.py # Rule-based scoring
├── model.pkl # Saved trained model
├── vectorizer.pkl # Saved TF-IDF vectorizer
├── requirements.txt
├── README.md
├── data/
│ └── datasets

## 🚀 How to Run the Project

### 🔹 Step 1: Clone Repository

git clone https://github.com/yourusername/Fake-Job-Scam-Detection.git

cd Fake-Job-Scam-Detection

### 🔹 Step 2: Create Virtual Environment

python -m venv venv
venv\Scripts\activate # For Windows

### 🔹 Step 3: Install Dependencies

pip install -r requirements.txt

### 🔹 Step 4: Train Model (Run Once)

python model.py


👉 This will create:
- `model.pkl`
- `vectorizer.pkl`

### 🔹 Step 5: Run Application

streamlit run app.py

## Input Options

- ✍️ Text Input (Paste job description)
- 📄 File Upload:
  - `.txt`
  - `.pdf`
  - `.docx`

## Output

- ✔ Prediction:
  - Safe Job  
  - Suspicious Job  
  - High Risk Scam  
- ✔ Confidence Score (%)
- ✔ Reason for Detection (Explainable AI)

## Key Highlights

- Uses **multiple ML models instead of a single model**
- Implements **ensemble learning for better performance**
- Provides **explainable predictions**
- Supports **real-world document input**
- Optimized using **model serialization (pickle)** for fast execution
- Combines **data-driven ML + rule-based intelligence**

## Future Improvements

- 🔹 Deep Learning Models (LSTM / BERT)
- 🔹 Real-time job scraping & detection
- 🔹 User feedback-based learning system
- 🔹 Advanced dashboard & analytics
- 🔹 API deployment

## Team Contribution

- 👤 Member 1: Model Training & Machine Learning Algorithms
- 👤 Member 2: Hybrid Prediction Logic (ML + Rules)
- 👤 Member 3: Rule-Based System & Explainability
- 👤 Member 4: Frontend Development & Deployment (Streamlit)

## Conclusion

This project demonstrates how **Machine Learning + Ensemble Methods + Explainable AI + Rule-Based Systems** can be combined to build a practical and efficient system for detecting fake job postings.