# credit-risk-classification
Univ of Denver Data Analytics | July 2024 | Andrew Mack

## Overview of the Analysis

* The purpose of this analysis was to train and evaluate a model based on loan risk using pre-classified loan data, for potential use in future prediction of the creditworthiness of borrowers based on the numerical features in the dataset. This README summarizes the data and its handling, decision making involving modeling and supervised learning, subsequent analysis, and conclusions on the efficacy of the model.

### Step 1: Read the lending_data.csv data from the Resources folder into a Pandas DataFrame
* The dataset consisted of historical lending activity from a peer-to-peer lending services company. Over 77,000 records comprised the data with columns for:

    - loan size
    - interest rate
    - borrower income
    - debt-to-income (a ratio)
    - the number of accounts
    - derogatory marks
    - total debt
    - loan status (binary)
    
![alt-text](https://github.com/andrewjmack/credit-risk-classification/blob/main/Credit_Risk/Screenshots/data_types.png "Initial Data")

* After reading the CSV file data into a Pandas Dataframe, the data types were reviewed along with the generation of a summary statistics table for the benefit of initial exploration and understanding of the data. An initial observation was that the mean value of 'derogatory remarks' (negative items on a borrower's credit report such as missed payments, bankruptices or foreclosures [1]) was greater than the median value of 0, with those central tendencies initially indicating that a majority of borrowing records the set had no negative credit marks; this is unsurprising given a practical expectation that a lending institution would likely struggle to remain in business if increased loan risk and delinquency were in the majority.

![alt-text](https://github.com/andrewjmack/credit-risk-classification/blob/main/Credit_Risk/Screenshots/summ_stats.png "Summary Statistics")

### Step 2: Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns
* A supervised machine learning model requires labeled (classified) data upon which training and testing can be performed.
* 'Loan status' was designated as the dependent (y) variable. It is a classification of 0 or 1 as to whether a loan in the dataset is considered to have been issued to a creditworthy or risky borrower, respectively; more specifically, if the loan is at risk of default.
* The other columnar data points were all deemed to be features and designated (X).
* X and y data were placed into separate frames/series for training/testing.

### Step 3: Split the data into training and testing datasets by using train_test_split
* From sklearn.model_selection, train_test_split was imported.
* X_train, X_test, y_train, and y_test data were created with a random state.
* In this case, training and test data were 75% and 25%, respectively, of the pre-split data.

###  Step 4: Fit a logistic regression model by using the training data (`X_train` and `y_train`)
* LogisticRegression ("LR") was imported from the SciKitLearn library.
    * A LR model was utilized in particular as a statistical method for predicting binary outcomes from data, as the intention of the model is to predict whether a potential borrower is either creditworthy or NOT creditworthy. E.g., a value of "0" in the “loan_status” column means that the loan is healthy; a value of "1" indicates the loan has a high risk of defaulting. LR is preferred for binary classification due its use of a sigmoid (squashing) function.
* A LR model was instantiated with a random state parameter of 1.
* The model was fit (trained) on the X_train and y_train data subsets.
* The model was then validated with the smaller subset of testing data.

### Step 5: Make predictions on the testing data labels by using the testing feature data (X_test) and the fitted model
* The model predicted labels for unclassified data.

### Step 6: Evaluating the model’s performance
* A confusion matrix and classification report were generated as critical aids in judging the predictive power and  understand the positi
![alt-text](https://github.com/andrewjmack/credit-risk-classification/blob/main/Credit_Risk/Screenshots/confusion_matrix.png "Confusion Matrix")

![alt-text](https://github.com/andrewjmack/credit-risk-classification/blob/main/Credit_Risk/Screenshots/classification_report.png "Classification Report")

![alt-text](https://github.com/andrewjmack/credit-risk-classification/blob/main/Credit_Risk/Screenshots/pos_neg.png)

## Results

* Accuracy:
- How often the model is correct; the ratio of correctly predicted observations to the total number of observations.
- Description of Model Accuracy, Precision, and Recall scores.
* Precision:
-  The ratio of correctly predicted positive observations to the total predicted positive observations: "For all the individuals who were classified by the model as being a credit risk, how many actually were a credit risk?"
- Description of Model Accuracy, Precision, and Recall scores.
* Recall:
- The ratio of correctly predicted positive observations to all predicted observations for that class: "For all the loans that are a credit risk, how many were classified by the model as being a credit risk?"
- Description of Model Accuracy, Precision, and Recall scores.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. Include your justification for recommending the model for use by the company.
For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

## Resources
[1]: "What are derogatory marks and what do they mean?" Capital One: https://www.capitalone.com/learn-grow/money-management/derogatory-credit/
- Initial dataset provided by EdX/Univ of Denver
- Course class sessions, slide content and activities
