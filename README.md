# credit-risk-classification
Univ of Denver Data Analytics | July 2024 | Andrew Mack

## Table of Contents: 

1. [Repo Contents](#repo-contents)

2. [Overview of the Analysis](#overview-of-the-analysis)

3. [Results](#results)

4. [Summary](#summary)

5. [Resources](#resources)

## Repo Contents

This repo consists of:
- a README document containing process details and analysis of a supervised learning model for credit risk classification
- a Credit_Risk folder containing:
    - a Jupyter Notebook where data was transformed and the model was trained and tested
    - a Resources folder with the original data source (.CSV)
    - a folder containing Screenshots for inclusion in the README analysis

## Overview of the Analysis

* The purpose of this analysis was to train and evaluate a model based on loan risk using pre-classified loan data, for potential use in future prediction of the creditworthiness of borrowers based on the numerical features in the dataset. This README summarizes the data and its handling, decision making involving modeling and supervised learning, subsequent analysis, and conclusions on the efficacy of the model. The Jupyter Notebook should be accessed for more granular details such as function parameters choices, etc.

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
* The model was fit (trained) on the X_train and y_train data subsets.
* The model was then validated with the smaller subset of testing data.

### Step 5: Make predictions on the testing data labels by using the testing feature data (X_test) and the fitted model
* Following training and testing, the model predicted labels for the unclassified data.

### Step 6: Evaluating the model’s performance
* A confusion matrix and classification report were generated as critical aids in judging the predictive power of the model through scoring, accuracy, precision, recall, et al.

![alt-text](https://github.com/andrewjmack/credit-risk-classification/blob/main/Credit_Risk/Screenshots/confusion_matrix.png "Confusion Matrix")

![alt-text](https://github.com/andrewjmack/credit-risk-classification/blob/main/Credit_Risk/Screenshots/classification_report.png "Classification Report")

![alt-text](https://github.com/andrewjmack/credit-risk-classification/blob/main/Credit_Risk/Screenshots/pos_neg.png)

## Results

Based on the confusion matrix, the model accurately predicted 18,663 healthy loans ("actual 0" vs. "predicted 0"; true negative) while incorrectly classifying only 56 unhealthy loans as ("actual 1" vs. "predicted 0"; false negative). 102 healthy loans were incorrectly classified as unhealthy ("actual 0" vs. "predicted 1"; false positive) while 563 unhealthy loans were correctly classified ("actual 1" vs. "predicted 1"; true positive).

* Accuracy:
- How often the model is correct; the ratio of correctly predicted observations to the total number of observations.
- The overall accuracy score of 0.99 is strong; while not perfect (1.0), the score is quite high, though cannot be relied upon solely.
- Why? Because accuracy can be susceptible to imbalanced classes. In the case of
identifying poor loan candidates, the number of good loans greatly outweighs the
number of at-risk loans. In this case, it can be seen how easy it is for a model to care more about the good loans, as these have the largest impact on accuracy.

* Precision:
-  The ratio of correctly predicted positive observations to the total predicted positive observations: "For all the individuals who were classified by the model as being a credit risk, how many actually were a risk?" High precision relates to a low false positive rate.
- While comprising a small portion of the data, there were 56 false positives resulting in a precision of only 0.85, meaning these applicants would be incorrectly turned away as risky based on the model alone.

* Recall:
- The ratio of correctly predicted positive observations to all predicted observations for that class: "For all the loans that are a credit risk, how many were classified by the model as being a risk?"
- A recall of 0.99 for negatives (approved loans) vs. recall of 0.91 for positives (loans flagged as risky) is likely due to the imbalance of data.

## Summary

The high degree of accuracy is impressive but should be "taken with a grain of salt" due to the nature of this data set. The precision of the data set is perhaps more meaningful; in a vacuum, the assymetry of model precision favoring the correct prediction of creditworthy applicants while imprecisely turning away creditworthy applicants rather than risk greater exposure lending to unworthy applicants may seem like good business.

However, there are several real-world considerations to be made, among which are:

* False positives that lead to the denial of creditworthy applicants risks losing customers, damaging the institution's reputation, and the potential for drawing unwanted regulatory scrutiny (with banking/finance a highly regulated industry and at risk for damaging legal penalties)
 
### **Recommendations:**

    - An outside audit of the model to inform further refinement
    - Comparison relative to best-in-class industry classification scores for similar modeling efforts
    - Financial modeling and forecasting against these outcomes to ensure that internal stakeholders have a comfort level with the business ramifications at the predicted rate of writeoffs (i.e., model-approved "healthy" loans that result in delinquency due to a false negative)
    - Manual due diligence of loan applications pre- and post- to understand if other factors outside the data have not been considered
    - As economically viable, fitting and testing with larger dataset(s) or different ratios of training vs testing data at the risk of over or underfitting the model

I recommend pursuing the above steps before releasing the model "into the world," with specific goals of improving the precision and recall of the model, given the known challenges to accuracy with imbalanced clases (such as in loan datasets) and in spite of an overemphasis against the risk of bad loans.

If the model were implemented without pursuing these recommendations, at minimum any loan applications rejected by the model should undergo addditional due diligence to safeguard against a false positive, which should not be overly onerous given the overwhelming imbalance in loans classified as true negatives.

## Resources
[1]: "What are derogatory marks and what do they mean?" Capital One: https://www.capitalone.com/learn-grow/money-management/derogatory-credit/
- Initial dataset provided by EdX/Univ of Denver
- Course class sessions, slide content and activities
<img src="https://capsule-render.vercel.app/api?type=waving&color=BDBDC8&height=150&section=header" />
