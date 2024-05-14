import eli5 
import sklearn
import shap
import spacy
import anchor
from anchor import anchor_text
import numpy as np
from sklearn.ensemble import RandomForestRegressor 
from eli5.sklearn import PermutationImportance 
from sklearn.datasets import fetch_california_housing
import matplotlib.pylab as plt
import pandas as pd
from sklearn.model_selection import train_test_split 
import csv, os, random
import numpy as np
from collections import OrderedDict
from typing import List, Set, Optional, Union
from sklearn.feature_extraction.text import CountVectorizer
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
from itertools import combinations
from sklearn.base import clone
import math 
import random
from collections import OrderedDict
import ast

np.random.seed(1)

def rfmultifeatimport(rf, featnames):
    # Collect feature importances from each tree in the Random Forest
    feat_impts = [tree.feature_importances_ for tree in rf.estimators_]
    
    # Calculate the mean importance for each feature across all trees
    across_trees = np.mean(feat_impts, axis=0)
    
    # Create a DataFrame from the feature names and their corresponding importances
    df_feature_importance = pd.DataFrame({
        'Feature': featnames,
        'RF Score': across_trees
    })
    
    # Sort the DataFrame by importance in descending order
    df_feature_importance = df_feature_importance.sort_values(by='RF Score', ascending=False).reset_index(drop=True)
    
    return df_feature_importance

def permimport(model, x_test, y_test, featnames):
    perm = PermutationImportance(model, random_state=1).fit(x_test, y_test)
    explanation = eli5.explain_weights_df(perm, feature_names=featnames)
    return explanation 

def sklearnpermimport(model, x, y, featnames):
    r = permutation_importance(model, x, y, n_repeats=30, random_state=0)
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{featnames[i]:<8}"
            f"{r.importances_mean[i]:.3f}"
            f" +/- {r.importances_std[i]:.3f}")
    return

def calculate_multioutput_shapley_permutation_SPEC(model, X, feature_names, output_names):
    """
    Calculate approximate Shapley values by comparing model outputs with the original and permuted versions of each feature
    for each output in a multi-output model.

    Args:
        model: A fitted multi-output model with a predict() method.
        X (numpy.ndarray): Dataset (n_samples, n_features).
        y (numpy.ndarray): Target values (used for fitting if needed, not directly in this function).
        feature_names (list): Names of the features in X.
        output_names (list): Names of each output in the model.

    Returns:
        dict: A dictionary of DataFrames, one for each output, containing features and their Shapley values sorted by importance.
    """
    n_samples, n_features = X.shape
    n_outputs = len(output_names)
    # Initialize a dictionary to store DataFrames for each output
    output_impacts = {name: pd.DataFrame(index=feature_names, columns=['Permutation Importance'], data=0.0) for name in output_names}

    # Predict with all features for all outputs
    base_predictions = np.array(model.predict(X))

    for i in range(n_features):  # Iterate over each feature
        X_modified = X.copy()
        np.random.shuffle(X_modified[:, i])  # Correctly permute the feature in-place

        # Predict with the modified dataset
        modified_predictions = model.predict(X_modified)

        for j in range(n_outputs):  # Iterate over each output
            # Calculate the average impact of permuting the feature on this output
            impact = np.mean(np.abs(base_predictions[:,j] - modified_predictions[:,j]))
            output_impacts[output_names[j]].loc[feature_names[i], 'Permutation Importance'] = impact

    # Sort each DataFrame by Shapley values
    for key in output_impacts:
        output_impacts[key] = output_impacts[key].sort_values(by='Permutation Importance', ascending=False).reset_index()

    return output_impacts


def calculate_multioutput_shapley_permutation_GEN(model, X, feature_names, output_names):
    """
    Calculate approximate Shapley values by comparing model outputs with the original and permuted versions of each feature
    for each output in a multi-output model.

    Args:
        model: A fitted multi-output model with a predict() method.
        X (numpy.ndarray): Dataset (n_samples, n_features).
        y (numpy.ndarray): Target values (used for fitting if needed, not directly in this function).
        feature_names (list): Names of the features in X.
        output_names (list): Names of each output in the model.

    Returns:
        dict: A dictionary of DataFrames, one for each output, containing features and their Shapley values sorted by importance.
    """
    n_samples, n_features = X.shape
    n_outputs = len(output_names)
    # Initialize a dictionary to store DataFrames for each output
    output_impacts = {name: pd.DataFrame(index=feature_names, columns=['Permutation Importance'], data=0.0) for name in output_names}

    # Predict with all features for all outputs
    base_predictions = np.array(model.predict(X))

    for i in range(n_features):  # Iterate over each feature
        X_modified = X.copy()
        np.random.shuffle(X_modified[:, i])  # Correctly permute the feature in-place

        # Predict with the modified dataset
        modified_predictions = model.predict(X_modified)

        for j in range(n_outputs):  # Iterate over each output
            # Calculate the average impact of permuting the feature on this output
            # impact = np.mean(np.abs(base_predictions[:,j] - modified_predictions[:,j]))
            impact = np.mean(np.abs(base_predictions - modified_predictions))
            output_impacts[output_names[j]].loc[feature_names[i], 'Permutation Importance'] = impact

    # Sort each DataFrame by Shapley values
    for key in output_impacts:
        output_impacts[key] = output_impacts[key].sort_values(by='Permutation Importance', ascending=False).reset_index()

    return output_impacts

def predcoverter(preds):
    # Flatten the matrix of strings and convert each string to a list
    flattened_and_converted = [ast.literal_eval(s) for sublist in preds for s in sublist]

    # Convert the list of lists into a NumPy array
    np_array = np.array(flattened_and_converted)

    # Reshape the array to the original matrix shape with the additional dimension for the lists
    np_array = np_array.reshape(len(preds), len(preds[0]), -1)

    # Output the resulting array
    return np_array

def calculate_multioutput_shapley_permutation(model, X, feature_names, output_names):
    """
    Calculate approximate Shapley values by comparing model outputs with the original and permuted versions of each feature
    for each output in a multi-output model.

    Args:
        model: A fitted multi-output model with a predict() method.
        X (numpy.ndarray): Dataset (n_samples, n_features).
        y (numpy.ndarray): Target values (used for fitting if needed, not directly in this function).
        feature_names (list): Names of the features in X.
        output_names (list): Names of each output in the model.

    Returns:
        dict: A dictionary of DataFrames, one for each output, containing features and their Shapley values sorted by importance.
    """
    n_samples, n_features = X.shape
    n_outputs = len(output_names)
    # Initialize a dictionary to store DataFrames for each output
    output_impacts = {name: pd.DataFrame(index=feature_names, columns=['Permutation Importance'], data=0.0) for name in output_names}

    # Predict with all features for all outputs
    base_predictions = np.array(model.predict(X))

    base_predictions = predcoverter(base_predictions)

    for i in range(n_features):  # Iterate over each feature
        X_modified = X.copy()
        np.random.shuffle(X_modified[:, i])  # Correctly permute the feature in-place

        # Predict with the modified dataset
        modified_predictions = model.predict(X_modified)

        modified_predictions = predcoverter(modified_predictions)

        for j in range(n_outputs):  # Iterate over each output
            # Calculate the average impact of permuting the feature on this output
            impact = np.mean(np.abs(base_predictions[:,0,j] - modified_predictions[:,0,j]))
            output_impacts[output_names[j]].loc[feature_names[i], 'Permutation Importance'] = impact

    # Sort each DataFrame by Shapley values
    for key in output_impacts:
        output_impacts[key] = output_impacts[key].sort_values(by='Permutation Importance', ascending=False).reset_index()

    return output_impacts

def shapley_values_multioutput_approx(model, X, y, n_samples, featnames, output_names):
    """
    Calculate approximate Shapley values for each feature for each output of a given multi-output model and data,
    using a sample of combinations.

    Args:
        model: A fitted sklearn-like multi-output model with a predict_proba() method.
        X (numpy.ndarray): Dataset (n_samples, n_features).
        y (numpy.ndarray): Target values (n_samples, n_outputs).
        n_samples (int): Number of random samples to use for estimating the Shapley values.

    Returns:
        numpy.ndarray: Shapley values for each feature for each output across all samples (n_samples, n_outputs, n_features).
    """
    n_samples_data, n_features = X.shape
    n_outputs = y.shape[1] if len(y.shape) > 1 else 1
    shapley_values = np.zeros((n_samples_data, n_outputs, n_features))

    # Iterate over each output
    for output_index in range(n_outputs):
        # Work with each feature as the feature of interest
        for j in range(n_features):
            # Sample subsets randomly
            subset_indices = [i for i in range(n_features) if i != j]  # Remove the feature of interest from potential subsets
            for _ in range(n_samples):
                k = random.randint(0, len(subset_indices))  # Random subset size
                subset = random.sample(subset_indices, k)  # Random subset

                if not subset:
                    continue

                subset_with_feature = subset + [j]
                X_subset = X[:, subset]
                X_subset_with_feature = X[:, subset_with_feature]

                # Fit models for subsets
                model_subset = clone(model).fit(X_subset, y[:, output_index:output_index+1])
                model_subset_with_feature = clone(model).fit(X_subset_with_feature, y[:, output_index:output_index+1])

                # Predict probabilities for each subset
                proba_without_feature = model_subset.predict_proba(X_subset)
                proba_with_feature = model_subset_with_feature.predict_proba(X_subset_with_feature)

                y_pred_without_feature = proba_without_feature[0][:, 1]
                y_pred_with_feature = proba_with_feature[0][:, 1]

                # Calculate marginal contribution
                marginal_contribution = np.abs(y_pred_with_feature - y_pred_without_feature)
                # marginal_contribution = marginal_contribution.flatten()  # Ensure it is flat

                # Add to Shapley value for the current output
                shapley_values[:, output_index, j] += marginal_contribution / n_samples  # Normalize by number of samples
    
    sorted_dfs = {}
    for output_index, output_name in enumerate(output_names):
        # Ensure mean is taken over the correct axis
        mean_shapley_values = np.mean(shapley_values, axis=0)  # Corrected axis for averaging
        df = pd.DataFrame({
            'Feature': featnames,
            'Shapley Value': mean_shapley_values
        })
        sorted_dfs[output_name] = df.sort_values(by='Shapley Value', ascending=False).reset_index(drop=True)

    return sorted_dfs

    






# def rfmultifeatimport(rf, featnames):
#     feat_impts = []
#     for clf in rf.estimators_:
#         feat_impts.append(clf.feature_importances_)
#     accross_trees = np.mean(feat_impts, axis=0)
#     df_feature_importance = {}
#     for i in range(len(featnames)):
#         df_feature_importance[featnames[i]] = accross_trees[i]
        
#     # df_feature_importance = df_feature_importance.sort_values('importance', ascending=False)
#     df_feature_importance = sorted(df_feature_importance, key=df_feature_importance.get, reverse=True)
#     return df_feature_importance

# def compute_shap_values(model, X):
#     """
#     Compute Shapley values for each feature in a MultiOutputClassifier model.
    
#     Args:
#     model (MultiOutputClassifier): A fitted MultiOutputClassifier model.
#     X (numpy.array or pandas.DataFrame): Data used for computing Shapley values.
    
#     Returns:
#     dict: A dictionary where keys are the output labels and values are the SHAP values matrices.
#     """
#     # Initialize a dictionary to hold SHAP values for each output
#     shap_values_output = {}

#     # We iterate over each classifier in the MultiOutputClassifier
#     for i, estimator in enumerate(model.estimators_):
#         # Create an explainer for the current estimator
#         explainer = shap.Explainer(estimator.predict, X, feature_perturbation="interventional")

#         # Compute SHAP values
#         shap_values = explainer(X)

#         # Store the SHAP values in the dictionary with appropriate labels if available
#         if model.classes_ is not None:
#             label = model.classes_[i]
#         else:
#             label = f'Output_{i}'
        
#         shap_values_output[label] = shap_values.values

#     return shap_values_output


# def compute_shap_multioutput(model, X_train, y_train):
#     # Get the number of features
#     n_features = X_train.shape[1]
#     n_outputs = y_train.shape[1]
    
#     # Shuffle the feature indices, I'm not convinced this does anything?
#     feature_indices = np.random.permutation(n_features)
    
#     # Initialize an empty array to store SHAP values
#     shap_values = np.zeros((len(X_train), n_features, n_outputs))
#     print(shap_values.shape)
    
#     # Compute SHAP values for each feature
#     for i, feature_index in enumerate(feature_indices):
#         # Create a copy of X_train with the shuffled feature
#         X_train_shuffled = X_train.copy()
#         np.random.shuffle(X_train_shuffled[:, feature_index])
        
#         # Predict with the model on the shuffled data
#         y_pred_shuffled = model.predict(X_train_shuffled)
        
#         # Compute the difference in predictions for each output
#         for j in range(n_outputs):
#             # print(type(y_pred_shuffled))
#             # print(type(y_train))
#             # print(y_pred_shuffled.shape)
#             # print(y_train.shape)
#             # print(y_pred_shuffled[:5])
#             # print(y_train[:5])
#             # print(y_pred_shuffled[:, j])
#             # print(type(y_pred_shuffled[:, j]))
#             # print(y_train[:, j])
#             # print(type(y_train[:, j]))
#             # print(y_pred_shuffled[:, j][1])
#             # print(type(y_pred_shuffled[:, j][1]))
#             # print(np.array(y_pred_shuffled[:, j][0]))
#             # print(type(np.array(y_pred_shuffled[:, j][0])))
#             # print(np.array(y_train[:, j][0]))
#             # print(type(np.array(y_train[:, j][0])))            
#             lengthoflist = len(y_train[:, j][1])
#             diff = []
#             for k in range(lengthoflist):
#                 print(k)
#                 print(np.array(y_pred_shuffled[:, j][k]))
#                 print(type(np.array(y_pred_shuffled[:, j][k])))
#                 print(np.array(y_pred_shuffled[:, j][k]).shape)
#                 print(np.array(y_train[:, j][k]))
#                 print(type(np.array(y_train[:, j][k])))  
#                 print(np.array(y_train[:, j][k])[1])

#                 diff[k] = np.subtract(np.array(y_pred_shuffled[:, j][k]), np.array(y_train[:, j][k]))

#             # tensor1 = tf.convert_to_tensor(y_pred_shuffled[:, j])
#             # tensor2 = tf.convert_to_tensor(y_train[:, j])

#             # diff = tensor1 - tensor2
#             # diff = y_pred_shuffled[:, j] - y_train[:, j]  # Difference in predictions
#             shap_values[:, feature_index, j] = diff  # Assign the difference as SHAP value
#     mean_shap_values_per_feature = np.mean(shap_values, axis=0) #Assess the value of each feature accress all samples
#     return mean_shap_values_per_feature

# class ShapleyValues:
#     def __init__(self, model):
#         self.model = model
    
#     def fit(self, X, y):
#         self.X = X
#         self.y = y
#         self.num_features = X.shape[1]
        
#         # Compute model predictions
#         self.predictions = self.model.predict(X)
        
#     def compute_shapley_values(self, num_permutations=100):
#         shapley_vals = np.zeros((self.num_features, self.y.shape[1]))  # Initialize Shapley values matrix
        
#         for j in range(self.num_features):
#             for k in range(self.y.shape[1]):
#                 marginal_contributions = []
#                 feature_indices = list(range(self.num_features))
#                 feature_indices.remove(j)
                
#                 for _ in range(num_permutations):
#                     # Generate random permutation of features
#                     np.random.shuffle(feature_indices)
                    
#                     # Compute prediction with feature j included
#                     pred_with_j = self.model.predict(self.X[:, [j] + feature_indices])
                    
#                     # Compute prediction with feature j excluded
#                     pred_without_j = self.model.predict(self.X[:, feature_indices])
                    
#                     # Compute marginal contribution of feature j
#                     marginal_contribution = np.abs(pred_with_j[:, k] - pred_without_j[:, k])
#                     marginal_contributions.append(marginal_contribution)
                
#                 # Compute Shapley value for feature j and output k
#                 shapley_vals[j, k] = np.mean(marginal_contributions)
        
#         return shapley_vals

# # print(permimport(rf, x_test, y_test, featnames = housing.feature_names))

# def interpshaps(shapvals, featnames):
#     df_shap_values = pd.DataFrame(data=shap_values.values, columns=featnames)
    
#     df_feature_importance = pd.DataFrame(columns=['feature', 'importance'])
    
#     for col in df_shap_values.columns:
#         importance = df_shap_values[col].abs().mean()
#         df_feature_importance.loc[len(df_feature_importance)] = [col, importance]
        
#     df_feature_importance = df_feature_importance.sort_values('importance', ascending=False)

#     return df_feature_importance


# def shapimport(model, x_train, x_test, featnames):
#     # # If the model is OneVsRestClassifier, extract the underlying estimator
#     # if isinstance(model, OneVsRestClassifier):
#     #     model = model.estimator

#     explainer = shap.Explainer(model, x_train)
#     shap_values = explainer(x_test)

#     # shap.plots.bar(shap_values) 
#     # plt.tight_layout()
#     # plt.savefig('shap_bar_plot.pdf', format='pdf')

#     # shap.plots.bar(shap_values.abs.mean(0))
#     # plt.tight_layout()
#     # plt.savefig('shap_bar2_plot.pdf', format='pdf')

#     # shap.plots.waterfall(shap_values[0], show=False)
#     # plt.tight_layout()
#     # plt.savefig('shap_waterfall_plot.pdf', format='pdf')

#     # # Extract feature names
#     # feat_names = shap_values.data[0].data[0].get_feature_names_out()
    
#     df_shap_values = pd.DataFrame(data=shap_values.values, columns=featnames)
    
#     df_feature_importance = pd.DataFrame(columns=['feature', 'importance'])
    
#     for col in df_shap_values.columns:
#         importance = df_shap_values[col].abs().mean()
#         df_feature_importance.loc[len(df_feature_importance)] = [col, importance]
        
#     df_feature_importance = df_feature_importance.sort_values('importance', ascending=False)

#     return df_feature_importance

# # def shapimport(model, x_train, x_test, featnames):
# #     explainer = shap.Explainer(model, x_train)
# #     shap_values = explainer(x_test)

# #     shap.plots.bar(shap_values) 
# #     plt.tight_layout()
# #     plt.savefig('shap_bar_plot.pdf', format='pdf')

# #     shap.plots.bar(shap_values.abs.mean(0))
# #     plt.tight_layout()
# #     plt.savefig('shap_bar2_plot.pdf', format='pdf')

# #     shap.plots.waterfall(shap_values[0], show=False)
# #     plt.tight_layout()
# #     plt.savefig('shap_waterfall_plot.pdf', format='pdf')

# #     df_shap_values = pd.DataFrame(data=shap_values.values,columns=featnames)
# #     df_feature_importance = pd.DataFrame(columns=['feature','importance'])
# #     for col in df_shap_values.columns:
# #         importance = df_shap_values[col].abs().mean()
# #         df_feature_importance.loc[len(df_feature_importance)] = [col,importance]
# #     df_feature_importance = df_feature_importance.sort_values('importance',ascending=False)

# #     return(df_feature_importance)

# def shaptext(model, x_train, x_test):
#     explainer = shap.Explainer(model, x_train)
#     shap_values = explainer(x_test)
#     # plot a sentence's explanation
#     shap.plots.text(shap_values[0])
#     plt.tight_layout()
#     plt.savefig('shap_text_plot.pdf', format='pdf')
#     return()



# print(shapimport(rf, x_train, x_test, featnames = housing.feature_names))



# compute the SHAP values for the model

# explainer = shap.Explainer(rf, x_train)
# shap_values = explainer(x_test)
# df_shap_values = pd.DataFrame(data=shap_values.values,columns=housing.feature_names)
# df_feature_importance = pd.DataFrame(columns=['feature','importance'])
# for col in df_shap_values.columns:
#     importance = df_shap_values[col].abs().mean()
#     df_feature_importance.loc[len(df_feature_importance)] = [col,importance]
# df_feature_importance = df_feature_importance.sort_values('importance',ascending=False)
# print(df_feature_importance)
# print(housing.feature_names)
# print(shap_values.values)
# shap_importance = pd.Series(shap_values.values, housing.feature_names).abs().sort_values(ascending=False)
# print(shap_importance)

# shap.plots.bar(shap_values, x_test, feature_names = housing.feature_names) 
# plt.tight_layout()
# plt.savefig('shap_bar_plot.pdf', format='pdf')


# shap.plots.bar(shap_values.abs.mean(0))
# plt.tight_layout()
# plt.savefig('shap_bar2_plot.pdf', format='pdf')

# shap.plots.waterfall(shap_values[0], show=False)
# plt.tight_layout()
# plt.savefig('shap_waterfall_plot.pdf', format='pdf')

# Doesn't work

# shap.plots.heatmap(shap_values, show=False)
# plt.tight_layout()
# plt.savefig('shap_heatmap_plot.pdf', format='pdf')

# text only




# # get anchor conditionals
# nlp = spacy.load('en_core_web_sm')
# explainer = anchor_text.AnchorText(nlp, ['negative', 'positive'], use_unk_distribution=False)
# text = x_train[1]
# pred = explainer.class_names[predict_lr([text])[0]]
# alternative =  explainer.class_names[1 - predict_lr([text])[0]]
# exp = explainer.explain_instance(text, predict_lr, threshold=0.95, verbose=False, onepass=True)
# print('Anchor: %s' % (' AND '.join(exp.names())))
# print('Precision: %.2f' % exp.precision())
# print()
# print('Examples where anchor applies and model predicts %s:' % pred)
# print()
# print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
# print()
# print('Examples where anchor applies and model predicts %s:' % alternative)
# print()
# print('\n'.join([x[0] for x in exp.examples(only_different_prediction=True)]))
















# compute the SHAP values for the model

# explainer = shap.Explainer(rf)
# shap_values = explainer.shap_values(x_test)
# explainer = shap.Explainer(rf, x_train)
# shap_values = explainer(x_test)


# Plot SHAP summary plot

# shap.plots.bar(shap_values) 
# plt.tight_layout()
# plt.savefig('shap_bar_plot.pdf', format='pdf')

# shap.plots.bar(shap_values.abs.mean(0))
# plt.tight_layout()
# plt.savefig('shap_bar2_plot.pdf', format='pdf')

# shap.plots.waterfall(shap_values[0], show=False)
# plt.tight_layout()
# plt.savefig('shap_waterfall_plot.pdf', format='pdf')

# shap.plots.heatmap(shap_values, show=False)
# plt.tight_layout()
# plt.savefig('shap_heatmap_plot.pdf', format='pdf')

# # plot a sentence's explanation
# shap.plots.text(shap_values[0])
# plt.tight_layout()
# plt.savefig('shap_text_plot.pdf', format='pdf')





# # get anchor conditionals
# nlp = spacy.load('en_core_web_sm')
# explainer = anchor_text.AnchorText(nlp, ['negative', 'positive'], use_unk_distribution=False)
# text = x_train[1]
# pred = explainer.class_names[predict_lr([text])[0]]
# alternative =  explainer.class_names[1 - predict_lr([text])[0]]
# exp = explainer.explain_instance(text, predict_lr, threshold=0.95, verbose=False, onepass=True)
# print('Anchor: %s' % (' AND '.join(exp.names())))
# print('Precision: %.2f' % exp.precision())
# print()
# print('Examples where anchor applies and model predicts %s:' % pred)
# print()
# print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
# print()
# print('Examples where anchor applies and model predicts %s:' % alternative)
# print()
# print('\n'.join([x[0] for x in exp.examples(only_different_prediction=True)]))
