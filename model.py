import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from scipy.sparse import hstack


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)


# MAIN DATASET
df1 = pd.read_csv("data/fake_job_postings.csv", encoding="latin1", engine="python", on_bad_lines="skip")

text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
df1[text_cols] = df1[text_cols].fillna("")

df1["combined_text"] = (
    df1["title"] + " " +
    df1["company_profile"] + " " +
    df1["description"] + " " +
    df1["requirements"] + " " +
    df1["benefits"]
)


# SECOND DATASET
df2 = pd.read_csv("data/balanced_jop_posting.csv", encoding="latin1")

df2["combined_text"] = df2.iloc[:, 0].astype(str)
df2["fraudulent"] = 0
df2["telecommuting"] = 0
df2["has_company_logo"] = 0
df2["has_questions"] = 0


df1 = df1[["combined_text", "fraudulent", "telecommuting", "has_company_logo", "has_questions"]]
df2 = df2[["combined_text", "fraudulent", "telecommuting", "has_company_logo", "has_questions"]]

df = pd.concat([df1, df2], ignore_index=True)

df["combined_text"] = df["combined_text"].apply(clean_text)


# TF-IDF
vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1,2), min_df=2, max_df=0.9)
X_text = vectorizer.fit_transform(df["combined_text"])

flags = df[["telecommuting", "has_company_logo", "has_questions"]]

X = hstack([X_text, flags])
y = df["fraudulent"]


# SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# HYPERPARAMETER TUNING
param_grid = {"C": [0.1, 1, 10]}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight="balanced"),
    param_grid,
    cv=3,
    scoring='f1'
)

grid.fit(X_train, y_train)
best_lr = grid.best_estimator_

print("\nBest Logistic Regression:", best_lr)


# MODEL COMPARISON
models = {
    "Logistic Regression": best_lr,
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

print("\nMODEL COMPARISON\n")

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("Precision:", round(precision_score(y_test, y_pred), 3))
    print("Recall:", round(recall_score(y_test, y_pred), 3))
    print("F1:", round(f1_score(y_test, y_pred), 3))


# FINAL ENSEMBLE (SOFT VOTING)
model = VotingClassifier(
    estimators=[
        ('lr', best_lr),
        ('nb', MultinomialNB()),
        ('svm', SVC(probability=True))
    ],
    voting='soft'
)

model.fit(X_train, y_train)


# FINAL PERFORMANCE
y_pred = model.predict(X_test)

print("\nFINAL MODEL PERFORMANCE\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))


# EXPLAINABILITY
feature_names = vectorizer.get_feature_names_out()
coefficients = best_lr.coef_[0]

top_words = sorted(zip(coefficients, feature_names), reverse=True)[:10]

print("\nTop Scam Words:")
for coef, word in top_words:
    print(word)

import pickle

# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved successfully!")