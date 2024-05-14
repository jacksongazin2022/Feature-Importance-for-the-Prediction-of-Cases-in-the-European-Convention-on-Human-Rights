import utils
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, hamming_loss
import numpy as np
import datetime
from sklearn.model_selection import cross_val_score
import keras
import nikfunc
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.multioutput import MultiOutputClassifier
import json
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import warnings
from scipy.sparse import save_npz, load_npz
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.model_selection import ParameterSampler
from scipy.stats import loguniform, uniform
import argparse
## Used Chat GPT for this
def load_data():
    train_df = utils.extract_texts_conclusions('data/EN_train_Anon')
    test_df = utils.extract_texts_conclusions('data/EN_test_Anon')
    dev_df = utils.extract_texts_conclusions('data/EN_dev_Anon')
    train_df.to_csv('train_df.csv')
    dev_df.to_csv('dev_df.csv')
    test_df.to_csv('test_df.csv')
    return None
def phase_2(run=True, vectorizer=False, negative=True, filtered=True, run_language_check = True, transform = False, transform_train = False, nb = False):
    ## Loading the train, dev,test  and getting the distribution of response for train

    print('Reading test, train,dev')
    train_df_location = 'train_df.csv'
    dev_df_location = 'dev_df.csv'
    test_df_location = 'test_df.csv'
    print('Reading The files')
    train_df = pd.read_csv(train_df_location)
    dev_df = pd.read_csv(dev_df_location)
    test_df = pd.read_csv(test_df_location)

    if filtered:
        if run_language_check:
            ## If we want to filter out docs that are likely not english 
            utils.load_negative_words()
            train_df = utils.filter_non_english_texts(train_df)
            dev_df = utils.filter_non_english_texts(dev_df)
            test_df = utils.filter_non_english_texts(test_df)
            # Save the filtered data frames
            train_df_location = 'results/train_df_filtered.csv'
            dev_df_location = 'results/dev_df_filtered.csv'
            test_df_location = 'results/test_df_filtered.csv'
            train_df.to_csv(train_df_location, index=False)
            dev_df.to_csv(dev_df_location, index=False)
            test_df.to_csv(test_df_location, index=False)
        else:
            print('Loading Filtered Data Set')
            train_df_location = 'results/train_df_filtered.csv'
            dev_df_location = 'results/dev_df_filtered.csv'
            test_df_location = 'results/test_df_filtered.csv'
            train_df = pd.read_csv(train_df_location)
            dev_df = pd.read_csv(dev_df_location)
            test_df = pd.read_csv(test_df_location)

    # Construct paths dynamically based on filtering and negation
    print('Creating file locations')
    suffix = '_no_negation' if not negative else ''
    filtered_suffix = '_filtered' if filtered else '_not_filtered'
    naive_suffix = '_naive' if nb else ''
    vectorizer_save_path = f'results/tdif_vectorizer{suffix}{filtered_suffix}.pkl'
    score_file = f'results/scores_each_threshold{suffix}{filtered_suffix}{naive_suffix}.txt'
    best_code_map_file = f'results/best_code_map{suffix}{filtered_suffix}{naive_suffix}.json'


    if vectorizer:
        print('Fitting Tfidf Vectorizer on Training Data')
        ## Tokenizer can be done independetly
        tokenizer = utils.custom_tokenizer if negative else utils.spacy_tokenizer
        train_list = utils.tokenize_csv_column(train_df_location, 'train', tokenizer,suffix, filtered_suffix )
        dev_list = utils.tokenize_csv_column(dev_df_location, 'dev', tokenizer, suffix, filtered_suffix)
        test_list = utils.tokenize_csv_column(test_df_location, 'test', tokenizer,suffix, filtered_suffix)
        tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizer)
        start_time = datetime.datetime.now()
        tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_list)
        with open(vectorizer_save_path, 'wb') as file:
            pickle.dump(tfidf_vectorizer, file)
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        minutes = elapsed_time.seconds // 60
        seconds = elapsed_time.seconds % 60
        print('time to fit and transform vectorizer to training is!!!')
        print(f"{minutes} minutes {seconds} seconds")
    else:
        print('Loading Tokenized Lists')
        train_list, test_list, dev_list = utils.load_tokenized_lists(suffix, filtered_suffix)
        print(f'Length of train_list is {len(train_list)}')
        print('Loading vectorizer')
        with open(vectorizer_save_path, 'rb') as file:
            tfidf_vectorizer = pickle.load(file)
        X_matrix_location = f'results/train_phase_2_X{suffix}{filtered_suffix}.npz'
        if transform_train == True:
            print('Transforming sparse train')
            tfidf_matrix_train = tfidf_vectorizer.transform(train_list)
            save_npz(matrix = tfidf_matrix_train, file = X_matrix_location)
        else:
            print('Loading sparse train')
            tfidf_matrix_train = load_npz(X_matrix_location)
            print(f'tfidf_matrix_train has shape {tfidf_matrix_train.shape}')
        print(f'Shape of train is {tfidf_matrix_train.shape}')
        
    print('Transforming dev and test tdif')
    if transform:
        start_time = datetime.datetime.now()
        utils.transform_and_save_tdif_matrices(dev_list, test_list, vectorizer_save_path, suffix, filtered_suffix)
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        minutes = elapsed_time.seconds // 60
        seconds = elapsed_time.seconds % 60
        print('time to transform vectorizer to test and dev is!!!')
        print(f"{minutes} minutes {seconds} seconds")

    ## get distribution of Y
    distribution_df = utils.get_article_distribution(train_df_location)
    results_csv_path = f'results/scores_each_threshold{suffix}{filtered_suffix}{naive_suffix}.csv'
    if run:
        start_time = datetime.datetime.now()
        print('Running Cross Validation')
        with open(score_file, 'w') as file:
            file.write("Threshold, Mean Hamming Loss, Std Hamming Loss, Training Time\n")
        results_df = pd.DataFrame(columns=['Model_Considering', 'Negation', 'Filtered', 'Threshold', 'Mean_Hamming_Loss', 'Std_Hamming_Loss'])
        previous_scores = None
        best_score = np.inf  # Since lower is better for Hamming Loss
        best_model_details = None
        thresholds = [1, 2, 3]
        for threshold in thresholds:
            print(f'Running for Threshold {threshold}')
            code_map = utils.create_response_percent_threshold(distribution_df, threshold, filtered, negative)
            train_df = utils.vectorize_articles(train_df, code_map)
            Y = pd.DataFrame(train_df['article_vectors'].tolist())
            Y.to_csv(f'results/Y_train_threshold_{threshold}{suffix}{filtered_suffix}')
            print(Y)
            if nb: 
                results = utils.run_multilabel_logistic_cv(tfidf_matrix_train, Y)
            else:
                results = utils.run_multilabel_logistic_cv(tfidf_matrix_train, Y)
            results_df = results_df.append({
                'Model_Considering': 'Naive Bayes' if nb else 'Logistic Regression',
                'Negation': negative,
                'Filtered': filtered,
                'Threshold': threshold,
                'Mean_Hamming_Loss': results['mean_hamming_loss'],
                'Std_Hamming_Loss': results['std_hamming_loss']
            }, ignore_index=True)
            current_hamming_loss = results['mean_hamming_loss']
            with open(score_file, 'a') as file:
                file.write(f"{threshold}, {current_hamming_loss:.4f}, {results['std_hamming_loss']:.4f}, {results['training_time']:.2f}\n")

            if previous_scores is not None:
                significant = utils.compare_scores(previous_scores,current_hamming_loss)
                if significant and current_hamming_loss < best_score - 0.005:  # Hypothesis test for significant difference
                    best_score = current_hamming_loss
                    best_model_details = {
                        "threshold": threshold,
                        "hamming_loss": best_score,
                        "std_hamming": results['std_hamming_loss'],
                        "training_time": results['training_time']
                    }
                    best_code_map = code_map
                    best_Y = Y
            else:
                best_score = current_hamming_loss
                best_model_details = {
                    "threshold": threshold,
                    "hamming_loss": best_score,
                    "std_hamming": results['std_hamming_loss'],
                    "training_time": results['training_time']
                }
                best_code_map = code_map
                best_Y = Y

            previous_scores = current_hamming_loss  # Update previous scores to current for next iteration
        
        if best_code_map:
            with open(best_code_map_file, 'w') as f:
                json.dump(best_code_map, f)
                print(f"Best code map saved to '{best_code_map_file}'")
        
        print(f'Sending output of models we considered to {results_csv_path}')
        results_df.to_csv(results_csv_path)
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        minutes = elapsed_time.seconds // 60
        seconds = elapsed_time.seconds % 60
        print('time to determine the best threshold (grid search phase 2 is)!!!')
        print(f"{minutes} minutes {seconds} seconds")
        Y_location = f'results/train_phase_2_Y{suffix}{filtered_suffix}{naive_suffix}.csv'
        print(f'Also sending transformed Y to {Y_location}')
        best_Y.to_csv(Y_location)
        test_y_location =f'results/test_Y{suffix}{filtered_suffix}{naive_suffix}'
        dev_y_location = f'results/dev_Y{suffix}{filtered_suffix}{naive_suffix}'
        transform_response_dev_test(negative = negative, filtered= filtered, nb = nb, test_df_location = test_df_location,
                                    dev_df_location = dev_df_location)
def transform_response_dev_test(dev_df_location='results/dev_df_filtered.csv', test_df_location='results/test_df_filtered.csv',
                                 negative=False, filtered=True, nb=False):
    suffix = '_no_negation' if not negative else ''
    filtered_suffix = '_filtered' if filtered else '_not_filtered'
    naive_suffix = '_naive' if nb else ''
    best_code_map_file = f'results/best_code_map{suffix}{filtered_suffix}{naive_suffix}.json'
    
    with open(best_code_map_file, 'r') as file:
        code_map = json.load(file)

    dev_df = pd.read_csv(dev_df_location)
    # Assuming articles is stored as stringified list, parse it
    dev_df['articles'] = dev_df['articles'].apply(lambda x: eval(x))
    dev_df['article_vector'] = dev_df['articles'].apply(lambda articles: utils.articles_to_vector(articles, code_map))
    print(f"head is {dev_df['article_vector'].head()}")
    test_df = pd.read_csv(test_df_location)
    test_df['articles'] = test_df['articles'].apply(lambda x: eval(x))
    test_df['article_vector'] = test_df['articles'].apply(lambda articles: utils.articles_to_vector(articles, code_map))
    print(f"Test head is {test_df['article_vector'].head()}")
    test_y_location =f'results/test_Y{suffix}{filtered_suffix}{naive_suffix}'
    dev_y_location = f'results/dev_Y{suffix}{filtered_suffix}{naive_suffix}'
    test_df['article_vector'].to_csv(test_y_location)
    dev_df['article_vector'].to_csv(dev_y_location)
    print(f'Sending updated test to {test_y_location} and updated dev to {dev_y_location}')

      
def phase_3(negative = True, filtered = True, run = False, run_choosing = True):
    suffix = '_no_negation' if not negative else ''
    filtered_suffix = '_filtered' if filtered else '_not_filtered'
    print(f'Running with {suffix} and {filtered_suffix} for phase 3')
    Y_train_location = f'results/train_phase_2_Y{suffix}{filtered_suffix}.csv'
    vectorizer_save_path = f'results/tdif_vectorizer{suffix}{filtered_suffix}.pkl'
    train_list_location = f'results/train_phase_2_X_tokenized_list{suffix}{filtered_suffix}.pkl'
    test_list_location = f'results/test_phase_2_X_tokenized_list{suffix}{filtered_suffix}.pkl'
    dev_list_location = f'results/dev_phase_2_X_tokenized_list{suffix}{filtered_suffix}.pkl'
    dev_npz_location = f'results/dev_phase_2_X_tdif{suffix}{filtered_suffix}.npz'
    test_npz_location = f'results/test_phase_2_X_tdif{suffix}{filtered_suffix}.npz'
    train_npz_location = f'results/train_phase_2_X{suffix}{filtered_suffix}.npz'
   
           
    ## Loading Train Data
    Y_train = pd.read_csv(Y_train_location, index_col = 0)
    with open(vectorizer_save_path, 'rb') as file:
        vectorizer = pickle.load(file)
    if run:
        train_list = pickle.load(open(train_list_location, 'rb'))
        test_list = pickle.load(open(test_list_location, 'rb'))
        dev_list = pickle.load(open(dev_list_location, 'rb'))
        X_train_sparse = load_npz(train_npz_location)
        X_dev_sparse = vectorizer.transform(dev_list)
        save_npz(matrix = X_dev_sparse, file = dev_npz_location)
        X_test_sparse = vectorizer.transform(test_list)
        save_npz(matrix = X_test_sparse, file =test_npz_location)
    else:
        print(f' Loading Sparse Train from {train_npz_location}')
        X_train_sparse = load_npz(train_npz_location)
        print(f' Loading Sparse Dev from {dev_npz_location}')
        X_dev_sparse = load_npz(dev_npz_location)
        print(f' Loading Sparse Test from {test_npz_location}')
        X_test_sparse = load_npz(test_npz_location)
    features = vectorizer.get_feature_names_out()
    features_path = f'results/selected_features{suffix}{filtered_suffix}.pkl'
    indices_file_path = f'results/selected_feature_indices{suffix}{filtered_suffix}.pkl'
    if run_choosing:
        start_time = datetime.datetime.now()
        print('Getting best features with grid search')
        print(f'results will be in results/grid_search_results{suffix}{filtered_suffix}.txt')
        best_feature_selector, best_feature_indices = utils.train_feature_selector(X_train_sparse, Y_train, suffix, filtered_suffix)
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        minutes = elapsed_time.seconds // 60
        seconds = elapsed_time.seconds % 60
        print('TIme to determine best feature selector phase 3 is !!')
        print(f"{minutes} minutes {seconds} seconds")
        selected_features = features[best_feature_indices]
        print(f'Sending names of best features to {features_path}')
        with open(features_path, 'wb') as f:
            pickle.dump(selected_features, f)
        print(f'Sending indices of best features to {indices_file_path}')
        with open(indices_file_path, 'wb') as f:
            pickle.dump(best_feature_indices, f)
    else:
        print(f'Loading best features from {features_path}')
        with open(features_path, 'rb') as f:
            selected_features = pickle.load(f)
        print(f'We selected {len(selected_features)} tokens to keep!')
        print(f'Loading selected indices for sparse from {indices_file_path}')
        with open(indices_file_path, 'rb') as f:
            best_feature_indices = pickle.load(f)
        ## need code here to find selected indices by a mistake did not send them to a file
    X_train_phase_3_location = f'results/train_df_phase_3_X{suffix}{filtered_suffix}.npz'
    X_dev_phase_3_location =  f'results/dev_df_phase_3_X{suffix}{filtered_suffix}.npz'
    X_test_phase_3_location = f'results/test_df_phase_3_X{suffix}{filtered_suffix}.npz' 

    X_train_sparse_phase_3 = X_train_sparse[:, best_feature_indices]
    print(type(X_train_sparse_phase_3))
    print(f'Saving X_train_sparse_phase_3 to {X_train_phase_3_location}')
    save_npz(matrix = X_train_sparse_phase_3, file = X_train_phase_3_location)
    
    X_dev_sparse_phase_3 = X_dev_sparse[:, best_feature_indices]
    print(f'Saving X_dev_sparse_phase_3 to {X_dev_phase_3_location}')
    save_npz( matrix = X_dev_sparse_phase_3, file = X_dev_phase_3_location)

    X_test_sparse_phase_3 = X_test_sparse[:, best_feature_indices]
    print(f'Saving X_test_sparse_phase_3 to {X_test_phase_3_location}')
    save_npz(matrix = X_test_sparse_phase_3, file = X_test_phase_3_location)
from sklearn.multiclass import OneVsRestClassifier
def phase_4_tfidf(negative=False, filtered=True, nb = False):
    suffix = '_no_negation' if not negative else ''
    filtered_suffix = '_filtered' if filtered else '_not_filtered'
    naive_suffix = '_naive' if nb else ''
    X_dev_phase_3_location = f'results/dev_df_phase_3_X{suffix}{filtered_suffix}.npz'
    Y_dev_location = f'results/dev_Y{suffix}{filtered_suffix}.csv'
    X = load_npz(X_dev_phase_3_location)
    dev_y_location = f'results/dev_Y{suffix}{filtered_suffix}{naive_suffix}.csv'
    Y = pd.read_csv(dev_y_location, index_col = 0)
    print(Y.head())

    # Common classifier parameters
    class_weight = 'balanced'
    max_iter = 5000

    # Initialize the baseline OneVsRest model with LogisticRegression
    baseline_model = OneVsRestClassifier(LogisticRegression(class_weight=class_weight, solver='saga', max_iter=max_iter), n_jobs=-1)
    mean_hamming_loss, std_hamming_loss = utils.evaluate_model_cv(baseline_model, X, Y)
    print('Getting score for baseline model')
    best_score = utils.combined_score(mean_hamming_loss, std_hamming_loss, weight_mean=0.7, weight_std=0.3)
    file_name = f'results/phase_4_scoring_tdif{suffix}{filtered_suffix}.txt'

    # Write baseline results
    with open(file_name, 'w') as f:
        f.write(f"Baseline Model {baseline_model.estimator.get_params()}: Score {best_score}, Mean Hamming: {mean_hamming_loss}, Std Hemming: {std_hamming_loss}\n")

    param_distributions = {
    OneVsRestClassifier(LogisticRegression()): {
        'estimator__penalty': ['l2', 'none'],
        'estimator__C': loguniform(1e-4, 1e2),
        'estimator__solver': ['lbfgs', 'sag', 'saga']
    },
    OneVsRestClassifier(RidgeClassifier()): {
        'estimator__alpha': loguniform(1e-2, 1e2)
    },
    MultiOutputClassifier(SGDClassifier()): {
        'estimator__loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
        'estimator__penalty': ['l2', 'l1', 'elasticnet'],
        'estimator__alpha': loguniform(1e-4, 1e0),
        'estimator__l1_ratio': uniform(0, 1)
    },
    MultiOutputClassifier(RandomForestClassifier()): {
        'estimator__n_estimators': randint(100, 500),
        'estimator__max_depth': randint(10, 50),
        'estimator__min_samples_split': randint(2, 11),
        'estimator__min_samples_leaf': randint(1, 11),
        'estimator__max_features': ['auto', 'sqrt', 'log2', None]
    }
}


    results = []
    n_iter = 10
    best_model_row_number = None
    # Evaluate each model configuration
    i = 0
    with open(file_name, 'a') as f:  # Open file in append mode
        start_time = datetime.datetime.now()
        for model, params in param_distributions.items():
            param_list = list(ParameterSampler(params, n_iter=n_iter, random_state=42))
            for param in param_list:
                i+=1
                print(f'Testing model {model} with params {param}')
                current_model = model.set_params(**param)
                mean_hamming_loss, std_hamming_loss = utils.evaluate_model_cv(model, X, Y)
                score = utils.combined_score(mean_hamming_loss, std_hamming_loss, weight_mean=0.7, weight_std=0.3)
                print(f'{model} with {param} parameters got us Mean Hamming {mean_hamming_loss} with  ')
                # Log the results for each model tested
                f.write(f"Model: {type(model.estimator).__name__}, Params: {param}, Score: {score}, Mean Hamming: {mean_hamming_loss}, Std Hemming: {std_hamming_loss}\n")
                results.append((type(model.estimator).__name__,param, score, mean_hamming_loss, std_hamming_loss))
                if utils.compare_scores_phase_3(best_score, score):
                    best_score = score
                    best_model = model
                    if best_model_row_number == None:
                        best_model_row_number = 0
                    else:
                        best_model_row_number = len(results)-1
                    print(f'at this point this is our best model in results')
                    print(results[len(results)-1])
                    print(f'row numebr should be {len(results)-1+1} ')

    # Fit the best model to development data and save
    print('Fitting Best Model to Dev')
    print(f'row number for ben model is {best_model_row_number}')
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    minutes = elapsed_time.seconds // 60
    seconds = elapsed_time.seconds % 60
    print(f'Grid search time for phase 4 to fit {i} models is)!!!')
    print(f"{minutes} minutes {seconds} seconds")
    start_time = datetime.datetime.now()
    best_model.fit(X, Y)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    minutes = elapsed_time.seconds // 60
    seconds = elapsed_time.seconds % 60
    print(f'Grid search time for phase 4 to our best model {best_model} models is)!!!')
    print(f"{minutes} minutes {seconds} seconds")
    model_location = f'results/phase_4_model{suffix}{filtered_suffix}.pkl'
    print(f'Sending best model: {best_model.__class__.__name__} with params {best_model.estimator.get_params()} to {model_location}')
    with open(model_location, 'wb') as f:
        pickle.dump(best_model, f)

    # Optionally, save the results dataframe to a CSV for easy analysis
    results_df = pd.DataFrame(results, columns=['Model', 'param', 'Score', 'Mean Hamming Loss', 'Std Hamming Loss'])
    results_df.to_csv(f'results/phase_4_results{suffix}{filtered_suffix}.csv', index=False)
def build_model(input_dim, output_dim):
    return utils.build_simple_model(input_dim, output_dim)

def prepare_Y_for_NN(Y):
    print('Converting Dev Y into NP array for NN model')
    Y = np.array([np.array(eval(x)) for x in Y['article_vector']])
    return Y

def phase_4_neural_tdif(filtered=True, negative=False, nb=False):
    # Define file paths
    naive_suffix = '_naive' if nb else ''
    suffix = '_no_negation' if not negative else ''
    filtered_suffix = '_filtered' if filtered else '_not_filtered'
    results_df_location = f'results/phase_4_results{suffix}{filtered_suffix}.csv'
    model_location = f'results/phase_4_model{suffix}{filtered_suffix}.pkl'

    # Load and prepare data
    X_dev_phase_3_location = f'results/dev_df_phase_3_X{suffix}{filtered_suffix}.npz'
    dev_y_location = f'results/dev_Y{suffix}{filtered_suffix}{naive_suffix}.csv'
    X = load_npz(X_dev_phase_3_location)
    Y = pd.read_csv(dev_y_location, index_col=0)
    
    Y_nn = prepare_Y_for_NN(Y)

    # Define and compile the neural network model
    start_time = datetime.datetime.now()
    model = KerasClassifier(build_fn=lambda: build_model(X.shape[1], Y_nn.shape[1]), epochs=10, batch_size=32, verbose=0)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    minutes = elapsed_time.seconds // 60
    seconds = elapsed_time.seconds % 60
    print(f'Time to fit our neural network models is)!!!')
    print(f"{minutes} minutes {seconds} seconds")
    # Evaluate the model using cross-validation
    hamming_loss_scorer = make_scorer(hamming_loss)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, Y_nn, scoring=hamming_loss_scorer, cv=kf)
    mean_hamming_loss = np.mean(scores)
    print(f'Mean Hamming Loss for NN is {mean_hamming_loss}')
from sklearn.metrics import hamming_loss, make_scorer, recall_score, confusion_matrix
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def sensitivity_score(y_true, y_pred):
    # This is the same as recall
    return recall_score(y_true, y_pred, average='samples')
def phase_5(negative=False, filtered=True, nb=False):
    # Define file paths
    naive_suffix = '_naive' if nb else ''
    suffix = '_no_negation' if not negative else ''
    filtered_suffix = '_filtered' if filtered else '_not_filtered'
    best_code_map_file = f'results/best_code_map{suffix}{filtered_suffix}{naive_suffix}.json'
    X_test_phase_3_location = f'results/test_df_phase_3_X{suffix}{filtered_suffix}.npz'
    test_y_location = f'results/test_Y{suffix}{filtered_suffix}{naive_suffix}.csv'

    # Load the best code map
    with open(best_code_map_file, 'r') as file:
        code_map = json.load(file)
    X_dev_phase_3_location = f'results/dev_df_phase_3_X{suffix}{filtered_suffix}.npz'
    dev_y_location = f'results/dev_Y{suffix}{filtered_suffix}{naive_suffix}.csv'
    X = load_npz(X_dev_phase_3_location)
    Y = pd.read_csv(dev_y_location, index_col=0)
    
    Y_nn = prepare_Y_for_NN(Y)

    # Define and compile the neural network model
    model = KerasClassifier(build_fn=lambda: build_model(X.shape[1], Y_nn.shape[1]), epochs=10, batch_size=32, verbose=0)
    model.fit(X,Y_nn)

    # Load data
    X_test = load_npz(X_test_phase_3_location)
    Y_test = pd.read_csv(test_y_location, index_col=0)

    # Prepare Y for NN model or for prediction comparison
    Y_test_nn = prepare_Y_for_NN(Y_test)  # This function needs to be properly defined

  

    # Predict
    start_time = datetime.datetime.now()
    Y_pred = model.predict(X_test)
    Y_pred = (Y_pred > 0.5).astype(int)
    overall_hamming = hamming_loss(Y_test_nn, Y_pred)
    print(f'Overall Hamming Loss: {overall_hamming:.4f}')

    results = []

    # Calculate statistics and build results DataFrame
    for code, index in code_map.items():
        actual = Y_test_nn[:, index]
        pred = Y_pred[:, index]
        cm = confusion_matrix(actual, pred)
        TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, 0)
        accuracy = accuracy_score(actual, pred)
        sensitivity = recall_score(actual, pred, zero_division=0)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        # Save the confusion matrix to a specified location
        cm_df = pd.DataFrame(cm, index=['Actual No', 'Actual Yes'], columns=['Predicted No', 'Predicted Yes'])
        cm_filename = f'results/conf_matrix_{suffix}{filtered_suffix}{naive_suffix}_{code}.csv'
        cm_df.to_csv(cm_filename)

        results.append({
            'Article Code': code,
            'Accuracy': accuracy,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'True Positives': TP,
            'True Negatives': TN,
            'False Positives': FP,
            'False Negatives': FN,
            'Num Predicted Positive': TP + FP,
            'Num Predicted Negative': TN + FN,
            'Actual Positive': TP + FN,
            'Actual Negative': TN + FP,
            'Num Correct': TP + TN,
            'Num Incorrect': FP + FN
        })
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    minutes = elapsed_time.seconds // 60
    seconds = elapsed_time.seconds % 60
    print(f'Time to get predictions and sens and spec for neurla is!!')
    print(f"{minutes} minutes {seconds} seconds")
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'results/detailed_statistics_per_article{suffix}{filtered_suffix}{naive_suffix}.csv', index=False)

    # Calculate overall Hamming loss


    # Calculate overall Hamming loss
from scipy.sparse import csr_matrix, issparse
def are_sparse_matrices_equal(mat1, mat2):
    # Check if both inputs are sparse matrices
    if not (issparse(mat1) and issparse(mat2)):
        return False
    
    # Check if they have the same shape
    if mat1.shape != mat2.shape:
        return False
    
    # Check if they have the same number of non-zero elements
    if mat1.nnz != mat2.nnz:
        return False
    
    # Convert both matrices to 'coo' format for easier comparison
    mat1_coo = mat1.tocoo()
    mat2_coo = mat2.tocoo()
    
    # Sort indices to ensure they are in the same order
    order1 = np.lexsort((mat1_coo.row, mat1_coo.col))
    order2 = np.lexsort((mat2_coo.row, mat2_coo.col))
    
    # Check if rows, columns, and data are the same
    if not (np.array_equal(mat1_coo.row[order1], mat2_coo.row[order2]) and
            np.array_equal(mat1_coo.col[order1], mat2_coo.col[order2]) and
            np.array_equal(mat1_coo.data[order1], mat2_coo.data[order2])):
        return False
    
    return True
def analyze_random_forest(filtered=True, negative=False, nb=False):
    import numpy as np
    import pandas as pd
    import pickle
    from scipy.sparse import load_npz
    from sklearn.metrics import hamming_loss, confusion_matrix, accuracy_score, recall_score

    naive_suffix = '_naive' if nb else ''
    suffix = '_no_negation' if not negative else ''
    filtered_suffix = '_filtered' if filtered else '_not_filtered'

    X_test_phase_3_location = f'results/test_df_phase_3_X{suffix}{filtered_suffix}.npz'
    test_y_location = f'results/test_Y{suffix}{filtered_suffix}{naive_suffix}.csv'
    X_test = load_npz(X_test_phase_3_location)
    Y_test = pd.read_csv(test_y_location, index_col=0)

    model_location = f'results/phase_4_model{suffix}{filtered_suffix}.pkl'
    with open(model_location, 'rb') as file:
        model = pickle.load(file)
    start_time = datetime.datetime.now()
    Y_pred = np.array(model.predict(X_test))

    Y_test = nikfunc.predcoverter(np.array(Y_test))
    Y_pred  = nikfunc.predcoverter(Y_pred)
    
    sum = 0
    for i in range(0,2997):
        vector = abs(Y_test[i,0,:]- Y_pred[i,0,:])
        sum += np.sum(vector)
    hamming_loss = sum/(9*2997)
    print(hamming_loss)
    best_code_map_file = f'results/best_code_map{suffix}{filtered_suffix}{naive_suffix}.json'
    with open(best_code_map_file, 'r') as file:
        code_map = json.load(file)
    results = []

    # Calculate statistics and build results DataFrame
    for code, index in code_map.items():
        actual = Y_test[:, 0, index].flatten()  # Flattening if there's an extra dimension
        pred = Y_pred[:, 0, index].flatten()

        cm = confusion_matrix(actual, pred)
        TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, 0)
        accuracy = accuracy_score(actual, pred)
        sensitivity = recall_score(actual, pred, zero_division=0)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        # Save the confusion matrix to a specified location
        cm_df = pd.DataFrame(cm, index=['Actual No', 'Actual Yes'], columns=['Predicted No', 'Predicted Yes'])
        cm_filename = f'results/conf_matrix_{suffix}{filtered_suffix}{naive_suffix}_{code}.csv'
        cm_df.to_csv(cm_filename)

        results.append({
            'Article Code': code,
            'Accuracy': accuracy,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'True Positives': TP,
            'True Negatives': TN,
            'False Positives': FP,
            'False Negatives': FN,
            'Num Predicted Positive': TP + FP,
            'Num Predicted Negative': TN + FN,
            'Actual Positive': TP + FN,
            'Actual Negative': TN + FP,
            'Num Correct': TP + TN,
            'Num Incorrect': FP + FN
        })

    results_df = pd.DataFrame(results)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    minutes = elapsed_time.seconds // 60
    seconds = elapsed_time.seconds % 60
    print(f'Time to get predictions convert the predictions into right form  and sens and spec for random is!!')
    print(f"{minutes} minutes {seconds} seconds")
    results_location = 'results/rf_test_performance.csv'
    print(f'Sending RF test results to {results_location}')
    results_df.to_csv(results_location)

    

def main():
    parser = argparse.ArgumentParser(description='Process the phases and settings.')
    parser.add_argument('--run_phase_2', action='store_true', help='Run phase 2')
    parser.add_argument('--run_phase_3', action='store_true', help='Run phase 3')
    parser.add_argument('--run_phase_4', action='store_true', help='Run phase 4 TFIDF')
    parser.add_argument('--negation', action='store_true', help='Use negation or not')
    parser.add_argument('--filtered', action='store_true', help='Data is filtered or not')
    parser.add_argument('--use_nb', action='store_true', help='Use Naive Bayes or Logistic Regression')
    parser.add_argument('--run_language_check', action='store_true', help='Actually perform language checking')
    parser.add_argument('--transform', action='store_true', help='Transform dev and test tfidf')
    parser.add_argument('--transform_train', action='store_true', help='Transform train data tfidf')
    parser.add_argument('--vectorizer', action='store_true', help='Create new vectorizer')
    parser.add_argument('--run_phase_4_tfidf', action='store_true', help='Run phase 4 using TF-IDF and traditional ML models')
    parser.add_argument('--run_phase_4_neural', action='store_true', help='Run phase 4 using neural network models')
    parser.add_argument('--run_phase_5', action='store_true', help='Run phase 5 for final evaluation')
    parser.add_argument('--run_rf_analysis', action='store_true', help='Run RandomForest analysis after phase 4')
    args = parser.parse_args()


    warnings.filterwarnings('ignore')
    print('For phase 2 we already transformed our train_tdif, dev_tdif, and test_tdif so not need to train again ')
    if args.run_phase_2:
        phase_2(run=True, vectorizer= args.vectorizer, negative=args.negation, filtered=args.filtered,
                run_language_check= args.run_language_check, transform_train= args.transform_train, nb=args.use_nb, transform=args.transform)
    print('For Phase 3, we should just load the transformed data in so run= False. Transforming will take more time. This parameter is just here for experimentation and flexibility')
    print('Run Choosing means we will actually do the cross-validation with different feature selectors. This should basically be true unless in weird circumstances')
    run = False
    run_choosing = True
    if args.run_phase_3:
        print('Running Phase 3')
        phase_3(negative=args.negation, filtered=args.filtered, run=run, run_choosing=run_choosing)
    if args.run_phase_4_tfidf:
        print('Running Phase 4 TDIF')
        phase_4_tfidf(negative=args.negation, filtered=args.filtered)
    if args.run_rf_analysis:
        print('Running RandomForest Analysis')
        analyze_random_forest(filtered=args.filtered, negative=args.negation, nb=args.use_nb)
    if args.run_phase_4_neural:
        print('Running Phase 4 Neural')
        phase_4_neural_tdif()
    if args.run_phase_5:
        print('Running Phase 5')
        phase_5(negative=args.negation, filtered=args.filtered, nb=args.use_nb)
    




if __name__ == "__main__":
    main()


