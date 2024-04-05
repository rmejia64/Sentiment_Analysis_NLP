# Logistical Regression with Glove word embeddings

import numpy as np
import pandas as pd
import contractions
import cProfile
import pstats
from tqdm import tqdm
from joblib import Memory
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Specify the cache directory
cache_dir = "cache"
# Create a Memory object with the cache directory
memory = Memory(location=cache_dir, verbose=0)


@memory.cache
def load_glove(file_path):
    embeddings_index = {}
    with open(file_path, "r", encoding="utf-8") as f:
        # Get total number of lines in the file for tqdm
        num_lines = sum(1 for line in open(file_path, "r", encoding="utf-8"))
        # Initialize tqdm to track progress
        pbar = tqdm(total=num_lines, desc="Loading GloVe")
        # Wrap the loop with tqdm for progress indication
        for line in f:
            pbar.update(1)  # Update progress bar
            values = line.split()
            word = values[0]
            # Check if all values except the first one can be converted to floats
            if all(is_float(x) for x in values[1:]):
                try:
                    coefs = np.asarray(values[1:], dtype="float64")
                    embeddings_index[word] = coefs
                except ValueError:
                    # Skip lines that cannot be parsed
                    pass
        pbar.close()  # Close tqdm progress bar
    return embeddings_index


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def extract_embeddings(text, embeddings_index):
    embeddings = [
        embeddings_index[word] for word in text.split() if word in embeddings_index
    ]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(300)  # Assuming GloVe embeddings are 200-dimensional


def check_coverage(data, embeddings_index):
    total_words = set()
    covered_words = set()
    missing_words = set()

    for text in data:
        words = text.split()
        total_words.update(words)
        for word in words:
            if word in embeddings_index:
                covered_words.add(word)
            else:
                missing_words.add(word)

    coverage = len(covered_words) / len(total_words)
    print("\nCoverage: {:.2%}".format(coverage))
    print("Total words:", len(total_words))
    print("Words covered by embeddings:", len(covered_words))
    print("Words missing from embeddings:", len(missing_words))
    print("Example missing words:", list(missing_words)[:10])


@memory.cache
def clean(texts):
    spell = SpellChecker()
    lemmatizer = WordNetLemmatizer()
    cleaned_texts = []
    num_texts = len(texts)
    with tqdm(total=num_texts, desc="Cleaning data") as pbar:
        for text in texts:
            expanded_text = contractions.fix(text)
            corrected_words = [spell.correction(word) for word in expanded_text.split()]
            corrected_words = [word for word in corrected_words if word is not None]
            lemmatized_words = [lemmatizer.lemmatize(word) for word in corrected_words]
            cleaned_text = " ".join(
                [
                    word
                    for word in lemmatized_words
                    if word not in stopwords.words("english") and word.isalpha()
                ]
            )
            cleaned_texts.append(cleaned_text)
            pbar.update(1)  # Update progress bar
    return cleaned_texts


def feature_extraction(clean_data, data, embeddings_index):
    X = np.array([extract_embeddings(text, embeddings_index) for text in clean_data])
    y = data["Review_class"].values
    return X, y


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


def main():
    embeddings_index = load_glove("../datasets/glove.840B.300d.txt")

    data_imdb = pd.read_csv("../datasets/imdb_labelled.txt", sep="\t")
    data_imdb.columns = ["Review_text", "Review_class"]

    data_yelp = pd.read_csv("../datasets/yelp_labelled.txt", sep="\t")
    data_yelp.columns = ["Review_text", "Review_class"]

    data_amazon = pd.read_csv("../datasets/amazon_cells_labelled.txt", sep="\t")
    data_amazon.columns = ["Review_text", "Review_class"]

    data = pd.concat([data_imdb, data_amazon, data_yelp])

    clean_data = clean(data["Review_text"])

    check_coverage(clean_data, embeddings_index)

    x, y = feature_extraction(clean_data, data, embeddings_index)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0
    )

    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    Accuracy, F1, Precision, Recall = evaluation(y_test, y_pred)

    with open("results.txt", "a") as f:
        f.write("\n\n--- Logistical Regression with Glove word embeddings ---\n\n")
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

        cleaned_review = clean([user_review])
        print("\n\nCleaned input:\n\n", cleaned_review)
        embedding = extract_embeddings(cleaned_review[0], embeddings_index)
        predicted_sentiment = model.predict([embedding])[0]

        print(
            "\n\n--- Predicted sentiment ---\n\n\n",
            "Positive" if predicted_sentiment == 1 else "Negative",
            "\n\n",
        )

if __name__ == "__main__":
    # Run main() under the profiler
    cProfile.run("main()", "profile_stats")

    # Create a Stats object from the profile stats
    stats = pstats.Stats("profile_stats")

    # Sort the statistics by the cumulative time spent in a function
    stats.sort_stats("cumulative")

    # Print the top 10 functions with the highest cumulative time
    stats.print_stats(10)
