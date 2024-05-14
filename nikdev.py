from nikfunc import *
import eli5 
import sklearn
import shap
import spacy
import anchor
from anchor import anchor_text
import numpy as np
from sklearn.ensemble import RandomForestRegressor 
from eli5.sklearn import PermutationImportance 
import matplotlib.pylab as plt
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz, load_npz
import pickle
import utils


  
np.random.seed(1)

def prepare_Y_for_NN(Y):
    print('Converting Dev Y into NP array for NN model')
    Y = np.array([np.array(eval(x)) for x in Y['article_vector']])
    return Y


def main():
    negative = False
    filtered = True
    nb = False
    suffix = '_no_negation' if not negative else ''
    filtered_suffix = '_filtered' if filtered else '_not_filtered'
    naive_suffix = '_naive' if nb else ''
    x_test_location = "results/test_df_phase_3_X_no_negation_filtered.npz"
    y_test_location = "results/test_Y_no_negation_filtered.csv"
    x_train_location = "results/train_df_phase_3_X_no_negation_filtered.npz"

    x_train = load_npz(x_train_location)
    x_test = load_npz(x_test_location)
    y_test = pd.read_csv(y_test_location, index_col = 0)

    vectorizer_save_path = 'results/tdif_vectorizer_no_negation_filtered.pkl'

    with open(vectorizer_save_path, 'rb') as file:
        vectorizer = pickle.load(file)

    indice_path = 'results/selected_feature_indices_no_negation_filtered.pkl'
    with open(indice_path, 'rb') as file:
        indices = pickle.load(file)

    featnames = vectorizer.get_feature_names_out()[indices]
    
    x_test = x_test.toarray()
    y_test = y_test.values

    model_location = f'results/phase_4_model{suffix}{filtered_suffix}.pkl'

    with open(model_location, 'rb') as file:
        model = pickle.load(file)   

    output_names = ["2", "3", "5", "6", "8", "10", "11", "13", "14"]

    dfrf = rfmultifeatimport(model, featnames)

    dfpi = permimport(model, x_test, y_test, featnames)

    dfpi = dfpi.drop('std', axis=1)

    dfpi.rename(columns={'feature': 'Feature', 'weight': 'PI Score'}, inplace=True)

    df_merged = pd.merge(dfrf, dfpi, on='Feature', how='outer')

    dictofdfs = calculate_multioutput_shapley_permutation(model, x_test, featnames, output_names)

    for idx, (key, df) in enumerate(dictofdfs.items(), 0):
        df.rename(columns={'index': 'Feature', 'Permutation Importance': f"V{output_names[idx]} Score"}, inplace=True)
        df_merged = pd.merge(df_merged, df, on='Feature', how='outer')

    latex_table = df_merged.to_latex(index=False, caption="RF General Feature Importances", label="tab:feature_importances")
    print(latex_table)

    corr_matrix = df_merged.corr(method='pearson')

    latex_table = corr_matrix.to_latex(index=False, caption="RF General Feature Importances", label="tab:feature_importances")
    print(latex_table)


if __name__ == '__main__':
    main()
