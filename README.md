# Instructions for `final_project.py`

## Acknowledgements 

This study was developed by Jackson Gazin and Nikolas Lindauer and the guidance of our wonderful professor, Dr. Khuri.

## Overview
`final_project.py` is the primary script in this project, designed to execute a multistage text analysis and machine learning pipeline. This script allows users to manage the execution flow and set specific configurations using command-line arguments. It imports functions from `utils.py`to run many of the functions.

'nikfunc.py' contains several functions for evaluating feature importance. These are implemented on the random forest in 'nikdev.py' and on the neural network in 'neuraldev.py'. Each prints a latex script which contains the feature importance for each token in the model. Run both to get the output.

## Running the Script

To run the script from beginning to end (without re-vectorizing) with specific configurations, ensure you are in the directory containing `final_project.py` and use the following command in your terminal:

```bash
python3 final_project.py --run_phase_2 --run_phase_3 ---run_phase_4_tfidf --run_phase_4_neural --run_phase_5 --filtered --use_nb --run_rf_analysis
```

This command runs the script with English-only filtering (`--filtered`), utilizes the Naive Bayes classifier (`--use_nb`), and includes all phases up to the final evaluation and RandomForest analysis. The `--vectorizer` argument is not included, meaning it will load an existing vectorizer from a pickle file, and `--negation` is also omitted, based on previous findings that negating words did not increase accuracy. Note, we also did not include the `--transform`  or `--transform_train` so we load in the transform data sets in Phase 2 (for the Tf-IDF) rather than fitting and transforming again. If you want to get fitting times for these and the vectorizer, you can include these, but it will increase running time. 

### Command-Line Arguments Explained

- `--run_phase_2`: Execute phase 2 of the pipeline.
- `--run_phase_3`: Execute phase 3 of the pipeline.
- `--run_phase_4`: Run phase 4 of the pipeline specifically for TF-IDF processing.
- `--run_phase_5`: Execute phase 5 for the final evaluation of the models.
- `--filtered`: Ensure the data is filtered to include only English text.
- `--use_nb`: Use the Naive Bayes classifier. If not set, the default classifier (e.g., Logistic Regression) is used.
- `--negation`: Apply negation handling if this flag is set; otherwise, negation is not considered.
- `--run_language_check`: Perform a language check to filter out non-English text.
- `--transform`: Transform the development and test datasets using the TF-IDF vectorizer (Phase 2). If false, we will just load the transformed files in.
- `--transform_train`: Transform the training dataset using the TF-IDF vectorizer (Phase 2). If false, we will just load them in. Set to false in main 
- `--vectorizer`: Create a new TF-IDF vectorizer from scratch.
- `--run_phase_4_tfidf`: Execute phase 4 using TF-IDF and traditional machine learning models.
- `--run_phase_4_neural`: Execute phase 4 using neural network models.
- `--run_rf_analysis`: Conduct a RandomForest analysis after phase 4.


