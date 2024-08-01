

## Twitter Sentiment Analysis Using Machine Learning 

Developed a machine learning pipeline for sentiment analysis of tweets using a combination of preprocessing techniques, feature extraction, and multiple classification models. The project aimed to classify tweets as positive or negative based on their text content.

**Key Contributions:**
- **Data Preprocessing:** Implemented comprehensive text preprocessing including URL handling, emoji conversion, user mention replacement, non-alphabetic character removal, and lemmatization to clean and standardize tweet data.
- **Visualization:** Generated word clouds to visualize the most frequent words in positive and negative tweets, providing insights into the data distribution and common terms.
- **Feature Extraction:** Utilized TF-IDF Vectorizer to convert processed text data into numerical feature vectors, capturing the importance of terms in the corpus.
- **Model Training and Evaluation:** Trained and evaluated multiple machine learning models including Bernoulli Naive Bayes, Linear SVC, and Logistic Regression. Achieved the highest accuracy of 83% with the Logistic Regression model.
- **Model Comparison:** Compared model performance using precision, recall, f1-score, and confusion matrices to determine the best-performing model for sentiment classification.
- **Model Saving and Loading:** Saved the trained models and vectorizer using pickle for future use, enabling quick deployment and inference on new data.
- **Sentiment Prediction:** Developed functions to load models, preprocess new text data, transform it using the vectorizer, and predict sentiment, demonstrating the practical application of the trained models.

**Technologies Used:**
- **Python Libraries:** Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Seaborn, WordCloud
- **Machine Learning Models:** Bernoulli Naive Bayes, Linear SVC, Logistic Regression
- **Data Preprocessing Techniques:** Regular expressions, lemmatization, stopword removal
- **Model Persistence:** Pickle for saving and loading trained models and vectorizers

