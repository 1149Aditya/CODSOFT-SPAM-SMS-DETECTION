# CODSOFT-SPAM-SMS-DETECTION

## Overview

The Spam SMS Detection project aims to develop an AI model that classifies SMS messages as either spam or legitimate (ham). With the growing prevalence of spam messages, an effective detection system is crucial for protecting users from unwanted communications. This project utilizes natural language processing (NLP) techniques and machine learning classifiers to accurately identify spam messages.

## Table of Contents

- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [How to Run](#how-to-run)
- [Evaluation Metrics](#evaluation-metrics)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)

## Technologies Used

- **Programming Language:** Python
- **Libraries:**
  - **Pandas:** For data manipulation and analysis.
  - **NumPy:** For numerical operations.
  - **Scikit-learn:** For implementing machine learning algorithms.
  - **NLTK:** For natural language processing tasks.
  - **TF-IDF:** For converting text data into numerical format.
  - **Word Embeddings:** For capturing semantic meaning in text.

## Dataset

The dataset used in this project consists of labeled SMS messages, where each message is categorized as either "spam" or "ham" (legitimate). The dataset can be found [here](link_to_dataset) (please provide the actual link). The messages include various types of content, providing a diverse set for model training and evaluation.


## Methodology

1. **Data Preprocessing:**
   - Clean the text data by removing unnecessary characters, stop words, and applying lowercasing.
   - Split the dataset into training and testing sets.

2. **Feature Extraction:**
   - **TF-IDF:** Used to convert SMS messages into numerical format, capturing the importance of words in the context of the messages.
   - **Word Embeddings:** Optional technique to represent words in a continuous vector space, capturing semantic relationships.

3. **Model Training:**
   - Implement classifiers such as:
     - **Naive Bayes:** A probabilistic model based on Bayes' theorem.
     - **Logistic Regression:** A linear model used for binary classification.
     - **Support Vector Machines (SVM):** A powerful classifier that works well with high-dimensional spaces.

4. **Model Evaluation:**
   - Assess the performance of each model using metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC.



