# IMDB Movies Reviews Sentiment Analyzer

## Table of Contents
1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Dataset Information](#dataset-information)
4. [Problem Statement](#problem-statement)
5. [Solution Approach](#solution-approach)
    - 5.1 [Data Preprocessing](#data-preprocessing)
    - 5.2 [Feature Engineering](#feature-engineering)
    - 5.3 [Modeling](#modeling)
    - 5.4 [Real-Time Challenges](#real-time-challenges)
6. [Evaluation](#evaluation)
7. [Conclusion and Future Work](#conclusion-and-future-work)
8. [Installation](#installation)
9. [Usage](#usage)
10. [References](#references)

## Introduction
This project focuses on building a **Sentiment Analyzer** using the IMDB movie review dataset. The goal is to classify movie reviews as **positive** or **negative** sentiments by utilizing natural language processing (NLP) techniques and machine learning models. The case study also highlights challenges faced in real-world applications and how we can address them.

## Project Overview
The **Sentiment Analysis Case Study** uses the IMDB movie review dataset to demonstrate how to process raw data, perform feature engineering, build models, and evaluate them to predict sentiment. We aim to develop an industry-level solution capable of handling complexities like sarcasm, multi-polarity, and negations.

## Dataset Information
The dataset used in this project is sourced from **[Kaggle IMDB Movie Reviews Dataset](https://www.kaggle.com/)**, which contains reviews and ratings of movies.

- **Size:** 384 MB (CSV Format)
- **Main Features:**
  - `Rating`: User ratings (1 to 10 scale)
  - `Review`: User-written review in English
  - `Movie`: Movie titles
  - `Hisania`: Translations of the reviews into Portuguese

**Quick Stats:**
- **149,780 unique reviews** (in both English and Portuguese)
- **14,207 unique movies**

## Problem Statement
The objective is to create a **Sentiment Analyzer** that predicts whether a given movie review expresses a positive or negative sentiment. We aim to handle real-world challenges like sarcasm, multi-polarity (when reviews contain both positive and negative aspects), and negations, which are common in user reviews.

## Solution Approach
This project is divided into several phases, each addressing critical aspects of building a robust sentiment analysis model.

### 5.1 Data Preprocessing
1. **Mapping Ratings to Sentiment:** 
   - Convert the 1–10 rating scale into binary classes:
     - 1–4: Negative sentiment
     - 7–10: Positive sentiment
     - 5–6: Neutral (excluded from analysis)

2. **Cleaning the Text:**
   - Remove unnecessary characters, HTML tags, and stop words.
   - Convert text to lowercase for consistency.

### 5.2 Feature Engineering
Key challenges like sarcasm, multi-polarity, and negations require advanced feature engineering:
- **Sarcasm Detection:** Capture context, e.g., "My car has awesome mileage of 4 km per liter" – sarcasm detection requires domain knowledge.
- **Multi-Polarity Reviews:** Handle reviews that contain both positive and negative aspects by splitting and analyzing each sentence individually.
- **Negation Handling:** Recognize negations like "not," "never," or "can't" to avoid misclassification of sentiments (e.g., "I don't think this is a good movie" should be negative).

### 5.3 Modeling
- We use NLP-based models such as **TF-IDF**, **Bag of Words**, and **Word Embeddings** to represent text.
- Machine learning algorithms used include:
  - **Logistic Regression**
  - **Support Vector Machines (SVM)**
  - **Naive Bayes**
  - **Deep Learning (RNN, LSTM)**

### 5.4 Real-Time Challenges
Real-time sentiment analysis faces challenges such as:
- **Sarcasm:** Users often use sarcasm to express negative opinions in positive terms.
- **Multi-Polarity Sentences:** Reviews often contain mixed opinions (positive and negative).
- **Negation Words:** Simple keyword detection may fail to capture context where words like "not" or "never" reverse the sentiment.

## Evaluation
After building and training the models, we evaluate them on the test data using metrics such as:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

In addition, confusion matrices and ROC curves are used to better understand the performance.

## Conclusion and Future Work
The sentiment analyzer developed in this project successfully addresses key challenges, particularly around sarcasm and multi-polarity. Future improvements could involve:
- Expanding the model to multi-class classification (e.g., positive, neutral, negative).
- Integrating with real-time APIs (e.g., Twitter or news reviews).
- Deploying the model as a web service using AWS or other cloud platforms.

## References
- **IMDB Movie Reviews Dataset:** [Kaggle IMDB Dataset](https://www.kaggle.com/)
- **NLP Techniques:** Natural Language Processing with Python by Steven Bird
