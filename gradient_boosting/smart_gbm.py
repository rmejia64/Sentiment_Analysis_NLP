import numpy as np
import pandas as pd
import re
import contractions
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class SMARTVectorizer:
    def __init__(self, min_df=3):
        self.min_df = min_df
        self.doc_freqs = {}
        self.num_docs = 0

    def fit(self, documents):
        self.num_docs = len(documents)
        term_doc_freqs = {}

        for doc in documents:
            unique_words = set(
                word_tokenize(doc.lower())
            )  # Tokenize and convert to lower case
            for word in unique_words:
                term_doc_freqs[word] = term_doc_freqs.get(word, 0) + 1

        self.doc_freqs = {
            term: np.log10(self.num_docs / freq)
            for term, freq in term_doc_freqs.items()
            if freq >= self.min_df
        }

        return self

    def transform(self, documents):
        vectorized_documents = []

        for doc in documents:
            vector = {}
            words = word_tokenize(doc.lower())
            for word in words:
                if word in self.doc_freqs:
                    vector[word] = (1 + np.log10(words.count(word))) * self.doc_freqs[
                        word
                    ]
            vectorized_documents.append(vector)

        # Convert the list of dictionaries into a 2D array
        feature_names = list(self.doc_freqs.keys())
        vectorized_array = np.zeros((len(vectorized_documents), len(feature_names)))
        for i, vector in enumerate(vectorized_documents):
         for j, feature_name in enumerate(feature_names):
                if feature_name in vector:
                    vectorized_array[i, j] = vector[feature_name]

        return vectorized_array

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def clean(texts):
    # print("\n...cleaning data")
    cleaned_texts = []

    for text in texts:
        #  expand contractions
        expanded_text = contractions.fix(text)

        #  tokenize
        words = word_tokenize(expanded_text)

        #  remove stop words and puctuation
        stop_words = set(stopwords.words("english"))
        filtered_words = [
            word for word in words if word.lower() not in stop_words and word.isalpha()
        ]

        #  lemmantize
        lemmatizer = WordNetLemmatizer()
        filtered_words = [
            lemmatizer.lemmatize(word, pos="n") for word in filtered_words
        ]

        cleaned_text = " ".join(filtered_words)

        cleaned_texts.append(cleaned_text)

    return cleaned_texts


def feature_extraction(clean_data, data):
    # print("\n...extracting features")

    Vectorizer = SMARTVectorizer(min_df=3)
    Vectorizer.fit(clean_data)
    x = Vectorizer.transform(clean_data)
    y = data["Review_class"].values

    return x, y, Vectorizer


def train(x_train, y_train, x_test):
    # print("\n...training on data")

    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    return y_pred, model


def evaluation(y_test, y_pred):
    # print("\n...evaluating data\n")

    print("\n\n--- Model Scores ---\n\n")

    accuracy = str(accuracy_score(y_test, y_pred))
    f1 = str(f1_score(y_test, y_pred))
    precision = str(precision_score(y_test, y_pred))
    recall = str(recall_score(y_test, y_pred))

    print("accuracy: ", accuracy)
    print("f1: ", f1)
    print("precision: ", precision)
    print("recall: ", recall)

    # print("\n\n", data["Review_class"].value_counts())

    return accuracy, f1, precision, recall


def predict_sentiment(user_input, model, vectorizer):
    # print("\n...predicting sentiments")

    cleaned_input = clean([user_input])
    print("\n\nCleaned input:\n\n", cleaned_input)

    input_features = vectorizer.transform(cleaned_input)
    predicted_sentiment = model.predict(input_features)

    return predicted_sentiment[0]


def main():
    data_imdb = pd.read_csv("../datasets/imdb_labelled.txt", sep="\t")
    data_imdb.columns = ["Review_text", "Review_class"]

    data_yelp = pd.read_csv("../datasets/yelp_labelled.txt", sep="\t")
    data_yelp.columns = ["Review_text", "Review_class"]

    data_amazon = pd.read_csv("../datasets/amazon_cells_labelled.txt", sep="\t")
    data_amazon.columns = ["Review_text", "Review_class"]

    data = pd.concat([data_imdb, data_amazon, data_yelp])

    clean_data = clean(data["Review_text"])

    x, y, vectorizer = feature_extraction(clean_data, data)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0
    )
    y_pred, model = train(x_train, y_train, x_test)
    Accuracy, F1, Precision, Recall = evaluation(y_test, y_pred)

    with open("results.txt", "a") as f:
        f.write("\n\n--- SMART Weighting ---\n\n")
        f.write("accuracy: " + Accuracy + "\n")
        f.write("f1: " + F1 + "\n")
        f.write("precision: " + Precision + "\n")
        f.write("recall: " + Recall + "\n")

    print("\n\n--- Sentiment Analysis ---\n\n")
    while True:
        user_review = input("Enter your review (or type 'exit' to quit):\n\n")
        if user_review.lower() == "exit":
            print("\nGoodbye!\n")
            break
        predicted_sentiment = predict_sentiment(user_review, model, vectorizer)
        print(
            "\n\n--- Predicted sentiment ---\n\n\n",
            "Positive" if predicted_sentiment == 1 else "Negative",
            "\n\n",
        )


main()
