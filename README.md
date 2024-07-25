# Machine learning-based sample misidentification error detection in clinical laboratory tests

link

## Features
- Error Detection: AI based specific functions to predict and evaluate errors in lab results.
- Implementation of machine learning models including Deep Neural Networks (DNN), Random Forests (RF), Logistic Regression (LR), and XGBoost (XGB) with hyperparameter optimization using Optuna.
- Data preprocessing including scaling, transformation, and oversampling to handle imbalanced datasets.

## Requirements
Before running this project, install the required libraries as listed in requirements.txt. You can install these using pip:

```python
pip install -r requirements.txt
```

## Usage
The main functionalities are encapsulated in the script.

### 1) Data Preparation
Load your data into a Pandas DataFrame.  
The data should be preprocessed according to the model requirements.  
The data should include results from at least one specimen taken at different time points, and should be organized in the following order:   
current and previous  

![image](https://github.com/user-attachments/assets/fa9bd2ee-c85d-411b-a214-0f3911f29445)  
example of data format

```python
x_train_all, x_test_all, study_all = [],[],[]

# error simulation : shuffle ratio (DO NOT exceed 0.5)
shuffle_ratio = 0.01

# Load data
db=pd.read_pickle('../DB/sample_db.pkl')

# Generate dataset
x_train_all, x_test_all = gen_dbset(db, n_splits=5)
```

### 2) Model Training
- Choose a model name to use.
- Call the gen_dbset function to generate training and testing datasets.
- Use the nested_train_model function to perform model training with hyperparameter optimization.
- For model training only, you may adjust the shuffle ratio up to 0.5 or less. 
- n_trials is the number of hyperparameter tuning attempts (correct if it takes a lot of time).

```python
# Train and optimize model 
model_str = 'dnn' # Select model : 'dnn', 'xgb', 'rf', 'lr'
study_all, best_models = nested_train_model(x_train_all, model_str, n_trials =100, shuffle_ratio = shuffle_ratio)
```

### 3) Performance Evaluation
- Call nested_evaluate_model with the test datasets and trained models.
- This function performs iterative simulations by shuffling test data (based on the test_shuffle_ratio parameter) to reflect potential variability in new, unseen data.

```python
# Evaluation model with permutation test
result_all=nested_evaluate_model(x_test_all, best_models, model_str, n_iter = 1000, test_shuffle_ratio=0.01)
avg_scores=np.mean(result_all['scores'], axis=0)
metrics_names = ["AUC", "AUPRC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV"]
for name, score in zip(metrics_names, avg_scores):
    print(f"{name}: {score:.6f}")
```

### 4) For External and Internal Validation
To verify the performance of external data, two things are done as follows:  
1. External validation of internally developed models
   - use only test set of external validation dataset
2. Model development and validation on external data (=optimal performance model for external data)

```python
# Generate exteranl validation dataset
external_x_train_all, external_x_test_all = gen_dbset(external_db, n_splits=5)

# 1. external validation (use internally developed model)
result_all=nested_evaluate_model(external_x_test_all, best_models, model_str, n_iter = 1000, test_shuffle_ratio=shuffle_ratio)

# 2. Optimal performance model for external data
study_all, best_models = nested_train_model(external_x_train_all, model_str, n_trials =100, shuffle_ratio = shuffle_ratio)
result_all=nested_evaluate_model(external_x_test_all, best_models, model_str, n_iter = 1000, test_shuffle_ratio=shuffle_ratio)
```


