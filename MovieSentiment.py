import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from termcolor import colored

# Download required NLTK modules
nltk.download("punkt", quiet=True)      # 'punkt' for tokenization
nltk.download("stopwords", quiet=True)  # 'stopwords' for English stopwords
nltk.download("wordnet", quiet=True)   # 'wordnet' for lemmatization

# ------------------------------------------
# STEP 1: LOAD THE DATASET
# ------------------------------------------
df = pd.read_csv("Movie_Review.csv")
df = df.dropna(subset=["text", "sentiment"])

reviews = df["text"].tolist()
labels = df["sentiment"].tolist()

# ------------------------------------------
# STEP 2: TEXT PREPROCESSING
# ------------------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    """
    Cleans and normalizes input text:
    - lowercase
    - remove punctuation
    - tokenize
    - remove stopwords
    - lemmatize
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# Apply preprocessing to all reviews
X_clean = [preprocess(t) for t in reviews]

# ------------------------------------------
# STEP 3: FEATURE EXTRACTION (TF-IDF)
# ------------------------------------------
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
X_features = vectorizer.fit_transform(X_clean)

# ------------------------------------------
# STEP 4: TRAIN/TEST SPLIT
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)

# ------------------------------------------
# STEP 5: TRAIN MODEL (Linear SVM)
# ------------------------------------------
model = LinearSVC(class_weight={'neutral': 2, 'pos': 1, 'neg': 1})
model.fit(X_train, y_train)

# ------------------------------------------
# STEP 6: EVALUATE MODEL
# ------------------------------------------
y_pred = model.predict(X_test)

print("\n==============================")
print(" Sentiment Analysis Results")
print("==============================")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------------------
# STEP 7: USER INTERFACE IMPROVED
# ------------------------------------------
def predict_sentiment():
    """
    Takes user input, cleans text, predicts sentiment, and prints result.
    """
    user_review = input("\nWrite your movie review here: ").strip()

    if user_review == "":
        print(colored("Empty input! Please type something.", "yellow"))
        return

    clean_review = preprocess(user_review)
    vector = vectorizer.transform([clean_review])
    prediction = model.predict(vector)[0]

    # Color-coded results
    color = (
    "green" if prediction == "positive" 
    else "red" if prediction == "negative" 
    else "yellow"
    )
    print("\nPredicted Sentiment:", colored(prediction.upper(), color))


# ------------------------------------------
# MENU SYSTEM
# ------------------------------------------
print("\n==============================")
print("  MOVIE SENTIMENT SYSTEM")
print("==============================")

while True:
    print("\nChoose an option:")
    print("1) Analyze a review")
    print("2) Exit")

    choice = input("Your choice: ").strip()

    if choice == "1":
        predict_sentiment()
    elif choice == "2":
        print("\nGoodbye!")
        break
    else:
        print(colored("Invalid choice. Please try again.", "yellow"))
