# Machine learning-based sample misidentification error detection in clinical laboratory tests

link

## Features
- Error Detection: AI based specific functions to predict and evaluate errors in lab results.
- Implementation of machine learning models including Deep Neural Networks (DNN), Random Forests (RF), Logistic Regression (LR), and XGBoost (XGB) with hyperparameter optimization using Optuna.
- Data preprocessing including scaling, transformation, and oversampling to handle imbalanced datasets.

## Requirements
Before running this project, install the required libraries as listed in requirements.txt. You can install these using pip:

```bash
pip install -r requirements.txt

## Usage
The main functionalities are encapsulated in the script modeling_script.py. Here's how to run the models:

### Data Preparation
Load your data into a Pandas DataFrame.
The data should be preprocessed according to the model requirements.


### Model Training
- Set the model type and other parameters.
- Call the gen_dbset function to generate training and testing datasets.
- Use the nested_train_model function to perform model training with hyperparameter optimization.
- Evaluate the model using the predict_score function to get performance metrics.

```bash
study_all, x_train_all, x_test_all = [],[],[]
# Select model : 'dnn', 'xgb', 'rf', 'lr'
model_str = 'xgb'
# error simulation : shuffle ratio (DO NOT exceed 0.5)
shuffle_ratio = 0.01

# Load data
db=pd.read_pickle('../DB/sample_db.pkl')

# Generate dataset
x_train_all, x_test_all = gen_dbset(db, 'xgb' , n_splits=5)

# Train and optimize model 
study_all, best_models = nested_train_model(x_train_all, model_str, n_trials =100, shuffle_ratio = shuffle_ratio)
