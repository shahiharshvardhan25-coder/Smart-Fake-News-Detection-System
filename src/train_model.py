
import os
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. SET PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")
EVALUATION_DIR = os.path.join(OUTPUT_DIR, "evaluation_results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
os.makedirs(EVALUATION_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATASETS
# --------------------------------------------------

fake_df = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"))
true_df = pd.read_csv(os.path.join(DATA_DIR, "True.csv"))

print("\nDATASETS LOADED SUCCESSFULLY")
print("Fake news records:", len(fake_df))
print("Real news records:", len(true_df))

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df], ignore_index=True)

print("Combined dataset shape:", df.shape)

# --------------------------------------------------
# 3. REMOVE EMPTY UNNAMED COLUMNS
# --------------------------------------------------

unnamed_columns = [
    column for column in df.columns
    if column.startswith("Unnamed")
]

df.drop(columns=unnamed_columns, inplace=True)

print("\nEMPTY UNNAMED COLUMNS REMOVED")
print("Removed columns:", len(unnamed_columns))
print("Remaining columns:", list(df.columns))

# --------------------------------------------------
# 4. DATA CLEANING
# --------------------------------------------------

print("\nDUPLICATE ROWS BEFORE CLEANING:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("DUPLICATE ROWS AFTER CLEANING:", df.duplicated().sum())

df.dropna(subset=["title", "text"], inplace=True)

df["title"] = df["title"].astype(str).str.strip()
df["text"] = df["text"].astype(str).str.strip()

df = df[
    (df["title"] != "") &
    (df["text"] != "")
]

print("\nDATASET SHAPE AFTER CLEANING:", df.shape)

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDATASET COLUMNS:")
print(df.columns.tolist())

print("\nFIRST FIVE ROWS:")
print(df.head())

# --------------------------------------------------
# 5. EXPLORATORY DATA ANALYSIS
# --------------------------------------------------

print("\nLABEL DISTRIBUTION:")
print(df["label"].value_counts())

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="label")
plt.title("Fake and Real News Distribution")
plt.xlabel("News Label (0 = Fake, 1 = Real)")
plt.ylabel("Number of Articles")
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATION_DIR, "label_distribution.png"))
plt.close()

plt.figure(figsize=(10, 5))
sns.countplot(
    data=df,
    y="subject",
    order=df["subject"].value_counts().index
)
plt.title("News Subject Distribution")
plt.xlabel("Number of Articles")
plt.ylabel("Subject")
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATION_DIR, "subject_distribution.png"))
plt.close()

df["headline_length"] = df["title"].apply(lambda x: len(x.split()))

plt.figure(figsize=(8, 5))
sns.histplot(df["headline_length"], bins=30, kde=True)
plt.title("Headline Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATION_DIR, "headline_length.png"))
plt.close()

df["article_length"] = df["text"].apply(lambda x: len(x.split()))

plt.figure(figsize=(8, 5))
sns.histplot(df["article_length"], bins=30, kde=True)
plt.title("Article Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATION_DIR, "article_length.png"))
plt.close()

print("\nEDA VISUALIZATIONS SAVED SUCCESSFULLY!")

# --------------------------------------------------
# 6. TEXT PREPROCESSING
# --------------------------------------------------

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


df["combined_text"] = df["title"] + " " + df["text"]
df["combined_text"] = df["combined_text"].apply(preprocess_text)

print("\nTEXT PREPROCESSING COMPLETED!")

print("\nSAMPLE PREPROCESSED ARTICLE:")
print(df["combined_text"].iloc[0][:500])

# --------------------------------------------------
# 7. SPLIT DATASET
# --------------------------------------------------

X = df["combined_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDATASET SPLITTING COMPLETED!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# 8. TF-IDF VECTORIZATION
# --------------------------------------------------

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF VECTORIZATION COMPLETED!")
print("Training data shape:", X_train_tfidf.shape)
print("Testing data shape:", X_test_tfidf.shape)

# --------------------------------------------------
# 9. TRAIN MODELS
# --------------------------------------------------

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_tfidf, y_train)

print("\nLOGISTIC REGRESSION TRAINING COMPLETED!")
print("Accuracy:", accuracy_score(
    y_test, logistic_model.predict(X_test_tfidf)
) * 100, "%")

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

print("\nMULTINOMIAL NAIVE BAYES TRAINING COMPLETED!")
print("Accuracy:", accuracy_score(
    y_test, nb_model.predict(X_test_tfidf)
) * 100, "%")

# --------------------------------------------------
# 10. MODEL EVALUATION
# --------------------------------------------------

def evaluate_model(model, model_name):
    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    print(f"\n{model_name} EVALUATION")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Fake News", "Real News"],
        zero_division=0
    ))

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake News", "Real News"],
        yticklabels=["Fake News", "Real News"]
    )
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
    plt.savefig(os.path.join(VISUALIZATION_DIR, filename))
    plt.close()

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


lr_results = evaluate_model(logistic_model, "Logistic Regression")
nb_results = evaluate_model(nb_model, "Multinomial Naive Bayes")

print("\nCONFUSION MATRIX VISUALIZATIONS SAVED!")

# --------------------------------------------------
# 11. MODEL COMPARISON
# --------------------------------------------------

comparison_df = pd.DataFrame([lr_results, nb_results])

print("\nMODEL COMPARISON")
print(comparison_df)

comparison_path = os.path.join(
    EVALUATION_DIR, "model_comparison.csv"
)

comparison_df.to_csv(comparison_path, index=False)

print("\nMODEL COMPARISON SAVED SUCCESSFULLY!")

# --------------------------------------------------
# 12. SELECT AND SAVE BEST MODEL
# --------------------------------------------------

best_model_name = comparison_df.loc[
    comparison_df["F1 Score"].idxmax(), "Model"
]

if best_model_name == "Logistic Regression":
    best_model = logistic_model
else:
    best_model = nb_model

print("\nSELECTED MODEL:", best_model_name)

joblib.dump(
    best_model,
    os.path.join(MODEL_DIR, "best_model.pkl")
)

print("\nBEST MODEL SAVED SUCCESSFULLY!")
print("Model location:", os.path.join(MODEL_DIR, "best_model.pkl"))

joblib.dump(
    tfidf,
    os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
)

print("\nTF-IDF VECTORIZER SAVED SUCCESSFULLY!")
print("Vectorizer location:", os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

print("\n=======================================================")
print("SMART FAKE NEWS DETECTION SYSTEM")
print("MODEL TRAINING AND EVALUATION COMPLETED!")
print("=======================================================")
