from nikfunc import *
import os, psutil, time
import numpy as np
import pandas as pd
from scipy.sparse import save_npz, load_npz
import pickle
import utils
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, hamming_loss
import numpy as np
from sklearn.model_selection import cross_val_score
import keras
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
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import hamming_loss, make_scorer, recall_score, confusion_matrix


np.random.seed(1)


def build_model(input_dim, output_dim):
    return utils.build_simple_model(input_dim, output_dim)

def prepare_Y_for_NN(Y):
    print('Converting Dev Y into NP array for NN model')
    Y = np.array([np.array(eval(x)) for x in Y['article_vector']])
    return Y

def phase_5(negative=False, filtered=True, nb=False):
    x_test_location = "results/test_df_phase_3_X_no_negation_filtered.npz"
    y_test_location = "results/test_Y_no_negation_filtered.csv"
    X = load_npz(x_test_location)
    X = X.toarray()
    y_test = pd.read_csv(y_test_location, index_col = 0)
    Y_nn = prepare_Y_for_NN(y_test)

    vectorizer_save_path = 'results/tdif_vectorizer_no_negation_filtered.pkl'

    with open(vectorizer_save_path, 'rb') as file:
        vectorizer = pickle.load(file)
    
    indice_path = 'results/selected_feature_indices_no_negation_filtered.pkl'

    with open(indice_path, 'rb') as file:
        indices = pickle.load(file)

    featnames = vectorizer.get_feature_names_out()[indices]

    output_names = ["2", "3", "5", "6", "8", "10", "11", "13", "14"]

    # Define and compile the neural network model
    model = KerasClassifier(build_fn=lambda: build_model(X.shape[1], Y_nn.shape[1]), epochs=10, batch_size=32, verbose=0)

    model.fit(X, Y_nn)

    # start = time.time()

    dictofdfs = calculate_multioutput_shapley_permutation_SPEC(model, X, featnames, output_names)

    # process = psutil.Process(os.getpid())
    # print('Memory usage in Mega Bytes: ', process.memory_info().rss/(1024**2))  # in bytes 
    # print(f'Time Taken: {time.time() - start}')

    df_merged = pd.DataFrame()

    for idx, (key, df) in enumerate(dictofdfs.items(), 0):
        df.rename(columns={'index': 'Feature', 'Permutation Importance': f"V{output_names[idx]} Score"}, inplace=True)
        if df_merged.empty:
            df_merged = df
        else:
            df_merged = pd.merge(df_merged, df, on='Feature', how='outer')

        # df_merged = pd.merge(df_merged, df, on='Feature', how='outer')

    latex_table = df_merged.to_latex(index=False, caption="Feature Importances", label="tab:feature_importances")
    print(latex_table)

    corr_matrix = df_merged.corr(method='pearson')
    latex_table = corr_matrix.to_latex(index=False, caption="Feature Importances", label="tab:feature_importances")
    print(latex_table)

    return 

def main():

    phase_5()

    # x_test_location = "results/test_df_phase_3_X_no_negation_filtered.npz"
    # y_test_location = "results/test_Y_no_negation_filtered.csv"

    # x_test = load_npz(x_test_location)
    # y_test = pd.read_csv(y_test_location, index_col = 0)

    # x_test = x_test.toarray()
    # y_test = y_test.values

    # vectorizer_save_path = 'results/tdif_vectorizer_no_negation_filtered.pkl'

    # with open(vectorizer_save_path, 'rb') as file:
    #     vectorizer = pickle.load(file)
    
    # indice_path = 'results/selected_feature_indices_no_negation_filtered.pkl'

    # with open(indice_path, 'rb') as file:
    #     indices = pickle.load(file)

    # featnames = vectorizer.get_feature_names_out()[indices]

    # output_names = ["2", "3", "5", "6", "8", "10", "11", "13", "14"]

    # model = phase_5()

    # model.fit(x_test, y_test)

    # dictofdfs = calculate_multioutput_shapley_permutation(model, x_test, featnames, output_names)

    # print(dictofdfs)
    
    # dfpi = permimport(model, x_test, y_test, featnames)

    # dfpi = dfpi.drop('std', axis=1)

    # dfpi.rename(columns={'feature': 'Feature', 'weight': 'PI Score'}, inplace=True)

    # df_merged = dfpi

    # for idx, (key, df) in enumerate(dictofdfs.items(), 0):
    #     df.rename(columns={'index': 'Feature', 'Permutation Importance': f"V{output_names[idx]} Score"}, inplace=True)
    #     df_merged = pd.merge(df_merged, df, on='Feature', how='outer')

    # corr_matrix = df_merged.corr(method='pearson')

    # latex_table = corr_matrix.to_latex(index=False, caption="Feature Importances", label="tab:feature_importances")
    # print(latex_table)




    # latex_table = df_merged[:50].to_latex(index=False, caption="Feature Importances", label="tab:feature_importances")
    # print(latex_table)

    # latex_table = dfrf[:10].to_latex(index=False, caption="RF Feature Importances", label="tab:feature_importances")
    # print(latex_table)

    # latex_table = dfpi[:10].to_latex(index=False, caption="General Feature Importances", label="tab:feature_importances")
    # print(latex_table)

    # for idx, (key, df) in enumerate(dictofdfs.items(), 0):
    #     latex_table = df[:10].to_latex(index=False, caption=f"Violation {output_names[idx]} Feature Importances", label="tab:feature_importances")
    #     print(latex_table)

if __name__ == '__main__':
    main()
