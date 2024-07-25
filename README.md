# credit-risk-classification
Univ of Denver Data Analytics | July 2024 | Andrew Mack

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* The purpose of this analysis was to train and evaluate a model based on loan risk.
* The dataset consisted of historical lending activity from a peer-to-peer lending services company to build the model to identify the creditworthiness of borrowers. Over 77,000 records comprised the data with columns for:
    - loan size
    - interest rate
    - borrower income
    - debt-to-income (a ratio)
    - the number of accounts
    - derogatory marks
    - total debt
    - loan status (binary)
    
![alt-text](https://github.com/andrewjmack/credit-risk-classification/blob/main/Credit_Risk/Screenshots/data_types.png "Initial Data")

* Explain what financial information the data was on, and what you needed to predict.


* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).

* Describe the stages of the machine learning process you went through as part of this analysis.

* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
* Logistic Regression was utilized in particular as a statistical method for predicting binary outcomes from data, as the intention of the model is to predict whether a potential borrower is either creditworthy or uncreditworthy. E.g., a value of "0" in the “loan_status” column means that the loan is healthy; a value of "1" indicates the loan has a high risk of defaulting.  

## Results



Using a bulleted, describe the accuracy, precision and recall scores of all machine learning model(s).

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.

* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. Include your justification for recommending the model for use by the company.
For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.
