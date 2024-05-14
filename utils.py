## This is utils

import sys
import utils
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.naive_bayes import BernoulliNB
from scipy.sparse import load_npz
from sklearn.model_selection import KFold
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import hamming_loss
from sklearn.linear_model import LogisticRegression
import tensorflow as tf  # Optional, to suppress C++ logs too
tf.get_logger().setLevel('ERROR')
from sklearn.naive_bayes import MultinomialNB
import requests
from sklearn.metrics import make_scorer, f1_score
import scipy
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn.metrics import make_scorer, hamming_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.multioutput import MultiOutputClassifier
from scipy.stats import ttest_rel
from bs4 import BeautifulSoup
import zipfile
from scipy.sparse import save_npz
import io
import spacy
import json
from sklearn.multioutput import MultiOutputClassifier
import pandas as pd
import warnings
import ast
#import spacy
#import pedantic
import sklearn
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
def load_negative_words():
    import nltk
    from nltk.corpus import words
    return None

def extract_texts_conclusions(folder_path):
    data = []  # To hold our text and conclusion pairs

    # Iterate through each file in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):  # Ensure we're reading only JSON files
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = json.load(file)
                
                # Check if the LANGUAGEISOCODE is ENG
                if content.get('LANGUAGEISOCODE') == 'ENG':
                    text = ' '.join(content.get('TEXT', []))  # Join the text list into a single string
                    conclusion = content.get('VIOLATED_ARTICLES', '')  # Get the conclusion
                    data.append({'text': text, 'articles': conclusion})  # Append to our list

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data, columns=['text', 'articles'])
    return df
def safe_convert_to_int(s):
    try:
        return int(s)
    except ValueError:
        return None  # or some other indicator that it's not a valid integer
def turn_response_into_list(file_location):
    df = pd.read_csv(file_location)
    df['articles'] = df['articles'].apply(lambda x: [safe_convert_to_int(item) for item in ast.literal_eval(x)] if isinstance(x, str) and x.startswith('[') else x)
    
    # Filter out None values that represent failed conversions
    df['articles'] = df['articles'].apply(lambda articles: [item for item in articles if item is not None])
    return df

def get_article_distribution(file_location):
    # Convert string representations of lists into actual lists and then to integers
    print('Turning Response into list')
    df = turn_response_into_list(file_location)
    # Initialize a dictionary to count occurrences of each code
    code_counts = {}
    for articles in df['articles']:
        for article in articles:
            if article in code_counts:
                code_counts[article] += 1
            else:
                code_counts[article] = 1

    # Calculate total counts to compute percentages
    total_counts = sum(code_counts.values())
    
    # Create a DataFrame to store code counts and percentages
    distribution_df = pd.DataFrame(list(code_counts.items()), columns=['Code', 'Count'])
    distribution_df['Percentage'] = distribution_df['Count'] / total_counts * 100
    print('Returning trained_df with response_var as list type and distrtibution of sum of article violations')
    return  distribution_df
def create_response_percent_threshold(distribution_df, threshold, filtered=True, negative=True):
    # Filter to keep only articles with occurrence percentage above the threshold
    filtered_distribution = distribution_df[distribution_df['Percentage'] > threshold]
    
    # Create a mapping from article code to the index in the vector
    sorted_codes = sorted(filtered_distribution['Code'])
    code_map = {code: i for i, code in enumerate(sorted_codes)}
    
    # Determine file name based on filtering and negation status
    suffix = '_no_negation' if not negative else ''
    filtered_suffix = '' if filtered else '_not_filtered'
    mapping_filename = f'results/row_to_article_{threshold}_percent{suffix}{filtered_suffix}.txt'
    
    with open(mapping_filename, 'w') as f:
        for index, code in enumerate(sorted(code_map.keys())):
            f.write(f"Row {index + 1}: Article {code}\n")
        
        print(f"Processed threshold {threshold}%, mapping saved to '{mapping_filename}'")
    
    return code_map
def create_response_low_shot(distribution_df, filtered=True, negative=True):
    filtered_distribution = distribution_df[distribution_df['Percentage'] > 0]
    sorted_codes = sorted(filtered_distribution['Code'])
    code_map = {code: i for i, code in enumerate(sorted_codes)}
    suffix = '_no_negation' if not negative else ''
    filtered_suffix = '' if filtered else '_not_filtered'
    mapping_filename = f'results/mapping_low_shot{suffix}{filtered_suffix}.txt'
    with open(mapping_filename, 'w') as f:
        for index, code in enumerate(sorted(code_map.keys())):
            f.write(f"Row {index + 1}: Article {code}\n")
    print(f" Low shot mapping saved to '{mapping_filename}'")
    return code_map


    # Function to convert article list to vector

def safe_convert_to_int(s):
    """ Convert string to integer safely. """
    try:
        return int(s)
    except ValueError:
        return None  # Return None for non-integer values

def articles_to_vector(articles, code_map):
    """ Converts article list to a vector using the code map. Handles strings representing lists. """
    if isinstance(articles, str) and articles.strip():
        try:
            articles = ast.literal_eval(articles)  # Convert string representation of list to actual list
        except (ValueError, SyntaxError):
            articles = []  # In case of syntax error, treat it as an empty list
    else:
        articles = []

    # Convert articles to integers, filtering out non-integers
    articles = [safe_convert_to_int(article) for article in articles]
    articles = [article for article in articles if article is not None and article in code_map]

    vector = [0] * len(code_map)
    for article in articles:
        if article in code_map:
            vector[code_map[article]] = 1
    return vector

def vectorize_articles(df, code_map):
    """ Vectorizes the articles column of the dataframe using the provided code map. """
    df['article_vectors'] = df['articles'].apply(lambda articles: articles_to_vector(articles, code_map))
    return df
nlp = spacy.load("en_core_web_sm")



def compare_scores(scores_old, scores_new, alpha=0.05):
    """
    Performs a paired t-test to determine if there is a significant difference in the mean scores of two models.

    Args:
    - scores_old (list or np.array): F1 scores from the old model.
    - scores_new (list or np.array): F1 scores from the new model.
    - alpha (float): Significance level.

    Returns:
    - is_significant (bool): True if the new model is significantly better, False otherwise.
    """
    t_stat, p_value = ttest_rel(scores_old, scores_new)
    print(f"T-statistic: {t_stat}, P-value: {p_value}")
    return p_value < alpha
def scoring(mean_accuracy, std_accuracy, alpha = .5):
    return mean_accuracy - alpha * std_accuracy
import time
import numpy as np
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, hamming_loss
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.multiclass import OneVsRestClassifier

def run_multilabel_logistic_cv(X, Y, cv=3, class_weight='balanced', max_iter=5000):
    """
    Run cross-validation for the multi-label logistic regression classifier using One-vs-Rest approach.
    """
    start_time = time.time()
    clf = OneVsRestClassifier(LogisticRegression(class_weight=class_weight, solver='saga', max_iter=max_iter), n_jobs=-1)
    hamming_loss_scorer = make_scorer(hamming_loss)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, Y, cv= kf, scoring=hamming_loss_scorer)
    mean_hamming_loss = np.mean(scores)
    std_hamming_loss = np.std(scores)
    training_time = time.time() - start_time
    return {
        "mean_hamming_loss": mean_hamming_loss,
        "std_hamming_loss": std_hamming_loss,
        "training_time": training_time,
        "model": clf
    }
def run_multilabel_nb_cv(X, Y, cv=3, alpha=1.0):
    """
    Run cross-validation for the multi-label Bernoulli Naive Bayes classifier using MultiOutput approach.
    Suitable for binary label data.

    Parameters:
    X : array-like, sparse matrix of shape (n_samples, n_features)
        Training data.
    Y : array-like of shape (n_samples, n_targets)
        Binary target values.
    cv : int, default=3
        Number of folds in cross-validation.
    alpha : float, default=1.0
        Smoothing parameter (0 for no smoothing).

    Returns:
    dict : A dictionary containing the mean and standard deviation of the hamming loss,
           the training time and the trained model.
    """
    start_time = time.time()
    clf = MultiOutputClassifier(BernoulliNB(alpha=alpha), n_jobs=-1)
    hamming_loss_scorer = make_scorer(hamming_loss, greater_is_better= False)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, Y, cv=cv, scoring=hamming_loss_scorer)
    mean_hamming_loss = np.mean(scores)
    std_hamming_loss = np.std(scores)
    training_time = time.time() - start_time
    return {
        "mean_hamming_loss": mean_hamming_loss,
        "std_hamming_loss": std_hamming_loss,
        "training_time": training_time,
        "model": clf
    }
def save_sparse_matrix(filename, matrix):
    save_npz(filename, matrix)
def hamming_loss_scorer(y_true, y_pred):
    """Scorer function to calculate Hamming loss."""
    return hamming_loss(y_true, y_pred)

def save_dense_matrix_as_csv(filename, dataframe):
    dataframe.to_csv(filename, index=False)

def send_prepared_data(df, tfidf_vectorizer, code_map, output_file_prefix, negative=True, filtered=True, nb = False):
    df = vectorize_articles(df, code_map)
    Y = pd.DataFrame(df['article_vectors'].tolist())
    suffix = '_no_negation' if not negative else ''
    filtered_suffix = '_filtered' if filtered else ''
    naive_suffix = '_naive' if nb else ''
    location = f'results/{output_file_prefix}_Y{suffix}{filtered_suffix}{naive_suffix}.csv'
    print(f'Printing sending transformed Y for {output_file_prefix} to {location}')
    Y.to_csv(location)

def custom_tokenizer(text):
    tokens = nlp(text)
    transformed_tokens = []
    negation_scope = False
    for token in tokens:
        word = token.text
        
        # Reset negation scope at period or other ending punctuation
        if token.is_punct:
            if word == '.':
                negation_scope = False
            continue
        
        if negation_scope:
            word = f"Not_{word}"

        if word.lower() in {"not", "no", "never", "none"} or word.lower().endswith("n't"):
            negation_scope = True
            word = f"Not_{word}"

        transformed_tokens.append(word)
    return spacy_tokenizer(' '.join(transformed_tokens))
def spacy_tokenizer(document):
    nlp = spacy.load('en_core_web_sm')
    tokens = nlp(document)
    tokens = [token.lemma_.lower().strip() for token in tokens if not token.is_punct and not token.is_space]
    return tokens
def tokenize_csv_column(location, name, tokenizer, suffix, filtered_suffix):
    df = pd.read_csv(location)
    df['tokenized_text'] = df['text'].apply(lambda x: ' '.join(tokenizer(x)))

    csv_location = f'results/{name}_phase_2_X{suffix}{filtered_suffix}.csv'
    list_location = f'results/{name}_phase_2_X_tokenized_list{suffix}{filtered_suffix}.pkl'
    
    print(f'Sending tokenized version of {name} to {csv_location}')
    df.to_csv(csv_location, index=False)
    
    result_list = df['tokenized_text'].tolist()
    with open(list_location, 'wb') as f:
        pickle.dump(result_list, f)
        print(f'Tokenized text list dumped to {list_location}')
    
    return result_list

def load_tokenized_lists(suffix, filtered_suffix):
    
    train_list_location = f'results/train_phase_2_X_tokenized_list{suffix}{filtered_suffix}.pkl'
    test_list_location = f'results/test_phase_2_X_tokenized_list{suffix}{filtered_suffix}.pkl'
    dev_list_location = f'results/dev_phase_2_X_tokenized_list{suffix}{filtered_suffix}.pkl'

    train_list = pickle.load(open(train_list_location, 'rb'))
    test_list = pickle.load(open(test_list_location, 'rb'))
    dev_list = pickle.load(open(dev_list_location, 'rb'))
    
    print(f"Loaded tokenized lists: Train ({len(train_list)} items), Test ({len(test_list)} items), Dev ({len(dev_list)} items)")
    return train_list, test_list, dev_list

def transform_and_save_tdif_matrices(dev_list, test_list, vectorizer_path,suffix, filtered_suffix):
    with open(vectorizer_path, 'rb') as file:
        tfidf_vectorizer = pickle.load(file)
    
    dev_matrix = tfidf_vectorizer.transform(dev_list)
    test_matrix = tfidf_vectorizer.transform(test_list)

    
    dev_file_name = f'results/dev_phase_2_X_tdif{suffix}{filtered_suffix}.npz'
    test_file_name = f'results/test_phase_2_X_tdif{suffix}{filtered_suffix}.npz'
    
    save_npz(dev_file_name, dev_matrix)
    save_npz(test_file_name, test_matrix)
    
    print(f"Transformed TDIF matrices saved for dev and test: {dev_file_name} and {test_file_name}")
def is_english_word(word):
    return word.lower() in english_vocab

def filter_non_english_texts(df):
    nltk.download('words')
    english_vocab = set(words.words())
    def english_word_ratio(text):
        words = text.split()
        english_count = sum(is_english_word(word) for word in words)
        return (english_count / len(words)) if words else 0

    df['english_ratio'] = df['text'].apply(english_word_ratio)
    filtered_df = df[df['english_ratio'] >= 0.5]
    return filtered_df
from scipy.sparse import issparse

def train_feature_selector(X, Y, suffix, filtered_suffix, max_iter=5000, low_shot_suffix = ''):
    # Calculate variances and determine dynamic thresholds
    variances = np.var(X.toarray(), axis=0)
    percentiles = [95, 90, 80, 70]  # Percentiles to consider
    thresholds = [np.percentile(variances, 100 - p) for p in percentiles]
    k_values = [100, 200, 300, 400, 500]

    clf = OneVsRestClassifier(LogisticRegression(class_weight='balanced', solver='saga', max_iter=max_iter), n_jobs=-1)
    best_score = float('inf')
    best_selector = None
    best_feature_indices = None

    # Initialize a DataFrame to store the results
    results_df = pd.DataFrame(columns=["Type of Feature Reduction", "Mean Hamming Loss", "Std Hamming Loss", "Num Features", "Score"])
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    # Applying each variance threshold
    for threshold, percentile in zip(thresholds, percentiles):
        selector = VarianceThreshold(threshold=threshold)
        X_reduced = selector.fit_transform(X)
        
        scores = cross_val_score(clf, X_reduced, Y, cv=kf, scoring=make_scorer(hamming_loss))
        mean_hamming_loss = np.mean(scores)
        std_hamming_loss = np.std(scores)
        num_features = X_reduced.shape[1]
        score = custom_scorer_phase_3(mean_hamming_loss, std_hamming_loss, num_features)
        results_df = results_df.append({
            "Type of Feature Reduction": f"Variance Threshold (Top {100 - percentile}%)",
            "Mean Hamming Loss": mean_hamming_loss,
            "Std Hamming Loss": std_hamming_loss,
            "Num Features": num_features,
            "Score": score
        }, ignore_index=True)

        # Check if this score is the best
        if score < best_score:
            best_score = score
            best_selector = selector
            best_feature_indices = selector.get_support(indices=True)

    # Applying each k in SelectKBest
    for k in k_values:
        selector = SelectKBest(chi2, k=k)
        X_reduced = selector.fit_transform(X, Y)
        scores = cross_val_score(clf, X_reduced, Y, cv=kf, scoring=make_scorer(hamming_loss))
        mean_hamming_loss = np.mean(scores)
        std_hamming_loss = np.std(scores)
        num_features = k
        score = custom_scorer_phase_3(mean_hamming_loss, std_hamming_loss, num_features)
        results_df = results_df.append({
            "Type of Feature Reduction": f"SelectKBest (k={k})",
            "Mean Hamming Loss": mean_hamming_loss,
            "Std Hamming Loss": std_hamming_loss,
            "Num Features": num_features,
            "Score": score
        }, ignore_index=True)

        if score < best_score:
            best_score = score
            best_selector = selector
            best_feature_indices = selector.get_support(indices=True)
    print(f'In the end we will use {best_selector} because it had the highest score of {best_score}')

    # Save the results DataFrame to a CSV file
    results_df.to_csv(f'results/grid_search_results{low_shot_suffix}{suffix}{filtered_suffix}.csv', index=False)

    return best_selector, best_feature_indices

def custom_scorer_phase_3(mean_hamming_loss, std_hamming_loss, num_features, weight_mean=0.6, weight_feature=0.3, weight_std=0.1, feature_scale=0.001):
    """
    Custom scorer for selecting the best model and feature subset.
    Aims to minimize this score.
    """
    feature_penalty = feature_scale * num_features
    score = (weight_mean * mean_hamming_loss) + (weight_std * std_hamming_loss) + (weight_feature * feature_penalty)
    return score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D

def create_cnn_model(input_dim, output_dim):
    model = Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,), sparse=True))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(Reshape((128, 1)))
    model.add(Conv1D(64, 7, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='sigmoid'))  # Multi-label classification

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
from sklearn.model_selection import KFold
def evaluate_model_cv(model, X, Y):
    """Evaluate model using cross-validation and a custom Hamming loss scorer."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # Create a scorer object
    hamming_scorer = make_scorer(hamming_loss, greater_is_better=False)
    scores = cross_val_score(model, X, Y, cv=kf, scoring=hamming_scorer, n_jobs=-1)
    mean_hamming_loss = -scores.mean()  # Negate because 'greater_is_better=False' returns negative values
    std_hamming_loss = scores.std()
    return mean_hamming_loss, std_hamming_loss
    
def combined_score(mean_hamming_loss, std_hamming_loss, weight_mean = .7, weight_std=.3):
    # A simple way to combine these is just to sum them
    # You may choose to apply different weights if one is more important than the other
    print(f'Combined score for this one is {mean_hamming_loss} * {weight_mean} (mean hemming times weight) plus std_hemming * std_weight {std_hamming_loss} *{weight_std} ')
    return mean_hamming_loss*weight_mean + std_hamming_loss* weight_std

def compare_scores_phase_3(best_score, new_score, alpha=0.05):
    """
    Compares new model scores to the best model scores to determine if the new model is significantly better.
    Args:
    - best_score (float): Best score obtained so far.
    - new_score (float): New score from the current model.
    - alpha (float): Significance level.
    Returns:
    - (bool): True if the new model's score is significantly better and lower, False otherwise.
    """
    # Check if new_score is lower; if not, no need for statistical test
    if new_score >= best_score:
        return False
    
    # Simple statistical test simulation (as there's only one score, a real statistical test isn't applicable here)
    return (best_score - new_score) > (alpha * best_score)
def filter_features(row, selected_features):
    """
    Filter words in a row based on selected features.
    
    Parameters:
        row (str): Text data in a single row.
        selected_features (set): Set of selected features.
    
    Returns:
        str: Filtered text containing only selected features.
    """
    filtered_tokens = [token for token in row.split() if token in selected_features]
    return ' '.join(filtered_tokens)

def f1_score(y_true, y_pred):
    precision = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / K.sum(K.round(K.clip(y_pred, 0, 1)) + K.epsilon())
    recall = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / K.sum(K.round(K.clip(y_true, 0, 1)) + K.epsilon())
    return 2*(precision*recall)/(precision+recall+K.epsilon())

def build_simple_model(input_dim, output_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),  # Hidden layer
        Dense(output_dim, activation='sigmoid')              # Output layer
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', f1_score])  
    
    return model


def get_class_weights(y):
    # Calculate class weights which are useful for handling imbalanced datasets
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, class_weights))
    return class_weight_dict
def csr_to_sparse_tensor(csr_matrix):
    indices = np.array(list(zip(*csr_matrix.nonzero())), dtype=np.int64)
    values = csr_matrix.data
    shape = csr_matrix.shape
    return tf.sparse.SparseTensor(indices, values, shape)



def custom_tf_cv(model, X, Y, num_folds=5, random_state=42):
    num_samples = X.shape[0]  # Get the correct number of samples

    # Generate a random permutation of indices
    np.random.seed(random_state)
    indices = np.random.permutation(num_samples)

    fold_size = num_samples // num_folds
    hamming_losses = []

    for i in range(num_folds):
        start = i * fold_size
        end = start + fold_size if i != num_folds - 1 else num_samples

        # Correctly generate train and validation indices
        train_idx = np.concatenate([indices[:start], indices[end:]])
        valid_idx = indices[start:end]

        # Create boolean masks for selecting folds
        train_mask = np.zeros(num_samples, dtype=bool)
        valid_mask = np.zeros(num_samples, dtype=bool)
        train_mask[train_idx] = True
        valid_mask[valid_idx] = True

        # Convert boolean masks to TensorFlow tensors
        train_mask_tensor = tf.constant(train_mask)
        valid_mask_tensor = tf.constant(valid_mask)

        # Use boolean mask to select train and validation sets
        X_train = tf.sparse.retain(X, train_mask_tensor)
        Y_train = Y[train_mask]
        X_valid = tf.sparse.retain(X, valid_mask_tensor)
        Y_valid = Y[valid_mask]

        # Train the model
        model.fit(X_train, Y_train, epochs=10, batch_size=32)

        # Predict and calculate Hamming loss
        predictions = model.predict(X_valid)
        hamming_loss = np.mean(np.not_equal(predictions.round(), Y_valid).astype(int))
        hamming_losses.append(hamming_loss)

    mean_hamming_loss = np.mean(hamming_losses)
    std_hamming_loss = np.std(hamming_losses)

    print(f'Mean Hamming Loss: {mean_hamming_loss}, Standard Deviation: {std_hamming_loss}')
    return mean_hamming_loss, std_hamming_loss

                                       
