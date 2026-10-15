import re
import string
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#STEP 1: DATA PREPROCESSING

def step1_preprocessing(file_path='bbc-text (1).csv'):#Loads data, cleans text, removes stopwords, and tokenizes
    print("=== RUNNING STEP 1: Preprocessing ===")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded. Columns: {df.columns.tolist()}")

    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))

    def preprocess_text(text):#This function: cleans, tokenizes, and removes stopwords from a text string.
        if not isinstance(text, str):
            return []
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)#punctuation removal
        tokens = text.split()#tokenising text into individual words
        tokens = [word for word in tokens if word not in STOPWORDS]#removing stop words
        return tokens

    df['cleaned_tokens'] = df['text'].apply(preprocess_text)

    sample_text = df['text'].iloc[0]#Raw text of the first article
    print("\nOriginal Text Preview")
    print(sample_text[:100].strip() + "...\n")
    print("Preprocessed Tokens Preview (First 15 words)")
    print(df['cleaned_tokens'].iloc[0][:15])
    
    return df


#STEP 2: BAG OF WORDS

def step2_bow(df):#Manual train-test split, vocabulary building, and BoW matrix conversion    
    print("\n=== RUNNING STEP 2: Bag of Words ===")
    np.random.seed(42)
    total_rows = len(df)
    shuffled_indices = np.random.permutation(total_rows)

    split_index = int(total_rows * 0.8)
    train_df = df.iloc[shuffled_indices[:split_index]].reset_index(drop=True)
    test_df = df.iloc[shuffled_indices[split_index:]].reset_index(drop=True)

    print(f"Total documents: {total_rows} | Train: {len(train_df)} | Test: {len(test_df)}")

    #Building vocab strictly from Training Data
    unique_words = sorted(list(set(word for doc in train_df['cleaned_tokens'] for word in doc)))
    vocab_dict = {word: idx for idx, word in enumerate(unique_words)}
    vocab_list = unique_words
    print(f"Vocabulary Size (V): {len(vocab_list)}")

    def corpus_to_bow_matrix(corpus_tokens, vocab_dict, vocab_list):#Converts a corpus of token lists into a Bag of Words count matrix
        vectors = []
        for doc in corpus_tokens:
            vec = np.zeros(len(vocab_list), dtype=int)
            for word in doc:
                if word in vocab_dict:
                    vec[vocab_dict[word]] += 1
            vectors.append(vec)
        return np.array(vectors)

    X_train_bow = corpus_to_bow_matrix(train_df['cleaned_tokens'], vocab_dict, vocab_list)
    X_test_bow = corpus_to_bow_matrix(test_df['cleaned_tokens'], vocab_dict, vocab_list)

    print(f"Shape of Training BoW Matrix: {X_train_bow.shape}")

    #Single doc vector example preview
    sample_doc_idx = 0
    sample_vector = X_train_bow[sample_doc_idx]
    print(f"\nShowcase: Document Vector #{sample_doc_idx}")
    print(f"Total vector length (Vocabulary size V): {len(sample_vector)}")
    print(f"First 20 elements of the vector: {sample_vector[:20]}")

    #Finds which words actually appeared in this document (non-zero counts)
    non_zero_indices = np.nonzero(sample_vector)[0]
    print(f"Number of unique active words in this document: {len(non_zero_indices)}")
    print("Sample words present in this doc and their counts:")
    for idx in non_zero_indices[:5]:  #Shows first 5 active words
        word = vocab_list[idx]
        count = sample_vector[idx]
        print(f"  - '{word}': {count} time(s)")

    return train_df, test_df, vocab_dict, vocab_list, X_train_bow, X_test_bow


#STEP 3: TF-IDF

def step3_tfidf(X_train_bow, X_test_bow, vocab_dict, vocab_list):#Calculates TF and IDF, then combines them into TF-IDF matrices    
    print("\n=== RUNNING STEP 3: TF-IDF ===")
    N = X_train_bow.shape[0]

    def compute_tf(bow_matrix):#Calculates term frequency (TF)
        return bow_matrix.astype(float)#In Bag of Words, TF is simply the count matrix itself

    def compute_idf(bow_matrix):#Calculates inverse document frequency (IDF) using NumPy vectorization
        doc_frequencies = np.sum(bow_matrix > 0, axis=0)
        return np.log(N / doc_frequencies)

    train_tf_matrix = compute_tf(X_train_bow)
    idf_array = compute_idf(X_train_bow)

    train_idf_dict = {word: idf_array[idx] for word, idx in vocab_dict.items()}

    X_train_tfidf = train_tf_matrix * idf_array#Multiplies BoW count matrices by the IDF array to get TF-IDF matrices
    X_test_tfidf = X_test_bow * idf_array

    print(f"Training TF-IDF Matrix Shape: {X_train_tfidf.shape}")
    print(f"Testing TF-IDF Matrix Shape: {X_test_tfidf.shape}")

    return X_train_tfidf, X_test_tfidf, idf_array


#STEP 4: ANALYSIS

def step4_analysis(train_df, X_train_bow, X_train_tfidf, idf_array, vocab_list):#Category top words and TF vs IDF analysis    
    print("\n=== RUNNING STEP 4: Analysis ===")    
    #4.a For a given category, find the top 10 words with the highest average TF-IDF scores
    target_category = 'tech'
    category_mask = (train_df['category'] == target_category)

    if category_mask.any():
        mean_tfidf_scores = np.mean(X_train_tfidf[category_mask], axis=0)
        top_10_indices = np.argsort(mean_tfidf_scores)[::-1][:10]
        
        print(f"\nTop 10 Words for Category: '{target_category.upper()}'")
        for rank, idx in enumerate(top_10_indices, 1):
            print(f"{rank}. {vocab_list[idx]} (Avg TF-IDF: {mean_tfidf_scores[idx]:.4f})")
    else:
        print(f"Category '{target_category}' not found.")

    #4.b High TF vs Low IDF (and vice versa)
    total_tf_scores = np.sum(X_train_bow, axis=0)
    sorted_vocab_indices = np.argsort(idf_array)

    print("\nHigh TF / Low IDF Words (Common Corpus Terms)")
    for idx in sorted_vocab_indices[:5]:
        print(f"Word: '{vocab_list[idx]}' | Total TF: {total_tf_scores[idx]} | IDF: {idf_array[idx]:.4f}")

    print("\nLow TF / High IDF Words (Rare/Niche Terms)")
    for idx in sorted_vocab_indices[-5:][::-1]:
        print(f"Word: '{vocab_list[idx]}' | Total TF: {total_tf_scores[idx]} | IDF: {idf_array[idx]:.4f}")



def step5_classification(X_train_tfidf, X_test_tfidf, train_df, test_df):#Trains a simple classifier using my custom TF-IDF vectors to predict categories, since the objective was: to classify BBC news articles into their respective categories    
    print("\n=== RUNNING STEP 5: Classification Evaluation ===")    
    y_train = train_df['category']#Extracts target category labels
    y_test = test_df['category']
    
    classifier = LogisticRegression(max_iter=1000, random_state=42)#Initialize and train a classifier on my custom TF-IDF features
    classifier.fit(X_train_tfidf, y_train)    
    
    y_pred = classifier.predict(X_test_tfidf)#Predicts on the test set      
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Set Classification Accuracy: {accuracy * 100:.2f}%")
    
    #Prints a detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

#MAIN EXECUTION

if __name__ == "__main__":
    df = step1_preprocessing('bbc-text (1).csv')
    train_df, test_df, vocab_dict, vocab_list, X_train_bow, X_test_bow = step2_bow(df)
    X_train_tfidf, X_test_tfidf, idf_array = step3_tfidf(X_train_bow, X_test_bow, vocab_dict, vocab_list)
    step4_analysis(train_df, X_train_bow, X_train_tfidf, idf_array, vocab_list)
    step5_classification(X_train_tfidf, X_test_tfidf, train_df, test_df) 