# Smart Fake News Detection System

## 1. Project Overview
The Smart Fake News Detection System is a machine learning-based application that classifies news articles as Fake News or Real News using Natural Language Processing (NLP).

The system uses TF-IDF for text feature extraction and compares Logistic Regression and Multinomial Naive Bayes algorithms.

## 2. Objectives
- To classify news articles as fake or real.
- To preprocess and analyse textual data.
- To apply NLP and machine learning techniques.
- To compare the performance of two classification models.
- To develop a user-friendly application using Streamlit.

## 3. Dataset
The project uses the ISOT Fake and Real News Dataset from Kaggle.

Dataset source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

The dataset contains labelled fake and real news articles.

## 4. Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Multinomial Naive Bayes
- Streamlit
- Joblib

## 5. Project Workflow
1. Load the fake and real news datasets.
2. Clean duplicate and empty records.
3. Perform exploratory data analysis.
4. Preprocess the news text.
5. Split the dataset into training and testing sets.
6. Apply TF-IDF vectorization.
7. Train Logistic Regression and Multinomial Naive Bayes.
8. Evaluate and compare model performance.
9. Save the selected model and vectorizer.
10. Use Streamlit to make predictions.

## 6. Model Performance

### Logistic Regression
- Accuracy: 99.10%
- Macro Precision: 99.09%
- Macro Recall: 99.11%
- Macro F1-score: 99.10%

### Multinomial Naive Bayes
- Accuracy: 94.36%
- Macro Precision: 94.35%
- Macro Recall: 94.36%
- Macro F1-score: 94.35%

Logistic Regression was selected based on its higher macro F1-score on the test dataset.

## 7. How to Run the Project

Install the required libraries:

```bash
pip install -r requirements.txt
```

Train the models:

```bash
python src/train_model.py
```

Launch the application:

```bash
streamlit run app/streamlit_app.py
```

## 8. Project Structure
- data/ – Dataset files
- src/ – Model training code
- models/ – Saved model and TF-IDF vectorizer
- app/ – Streamlit application
- outputs/ – Visualizations and evaluation results
- screenshots/ – Project screenshots
- requirements.txt – Required Python libraries

## 9. Limitations
- Predictions depend on the dataset used for training.
- The model may be affected by bias and patterns in the dataset.
- It may incorrectly classify unfamiliar topics, satire, or breaking news.
- The application does not independently verify news sources or facts.

## 10. Conclusion
The project demonstrates how Natural Language Processing and machine learning can be used to classify news articles. Logistic Regression achieved the best performance among the two tested models on the held-out test dataset.

The application provides a simple interface for testing news articles, but its predictions should not be treated as definitive fact-checking results.