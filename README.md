# credit-risk-classification
Univ of Denver Data Analytics | July 2024 | Andrew Mack

## Overview of the Analysis

* The purpose of this analysis was to train and evaluate a model based on loan risk using pre-classified loan data, for potential use in future prediction of the creditworthiness of borrowers based on the numerical features in the dataset. This README summarizes the data and its handling, decision making involving modeling and supervised learning, subsequent analysis, and conclusions on the efficacy of the model.
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


Explain what financial information the data was on, and what you needed to predict.


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

## Resources
[1]: "What are derogatory marks and what do they mean?" Capital One: https://www.capitalone.com/learn-grow/money-management/derogatory-credit/
- Initial dataset provided by EdX/Univ of Denver
- Course class sessions, slide content and activities
