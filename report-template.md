# Module 12 Report

## Overview of the Analysis

* The main purpose of this experiment was to compare the perfomance of two logistic regression models (one trained on original date and the other on resampled data) in predicting the credit risk of loan applicants (loan_status) based on their loan_size, interest_rate, borrower_income, debt_to_income, num_of_accounts	derogatory_marks and total_debt.

* The data used contained the following information:
	* loan_size: amount money applicant borrowed.
	* interest_rate: the loan's interest rate.
	* borrower_income: the applicant's income.
	* debt_to_income: total debt to income ratio.
	* num_of_accounts: Number of bank account the applicant has
	* derogatory_marks: applicant's derogatory marks
	* total_debt: applicants total debt.
	* loan_status: The loan status either healthy or high-risk.
	
	The data contains 77,536 observations. We need to predict whether the loan requested by an applicant (loan_status) is high-risk or healthy.

* Out of the 77,536 observations, 75036 were healthy while only 2500 were high-risk, which shows a great imbalance in the data.

* The analysis involved the following stages of machine learning process.
	* Splitting the data into training and testing. 70% of the data was used for training and 30% was used in testing. The train_test_split function was used to split the data, it is imported from the sklearn library.
	* Fitting a logistic regression model by using the original training data. A logistic regression model was created and fitted with the original training data(with massive imbalance).
	* Making predictions on the test data. The fitted model above was use to predict the loan status of the test features. The results were saves as a varible y_pred.
	* Evaluating the model’s performance by using four metrics: balanced accuracy score, confusion matrix, classification report. The y_test (actual labels) together with the y_pred (predicted labels) were used to evaluate the models performance. 
	* Resampling the training data. The RandomOverSampler module from the imbalanced-learn library was used to balance the labels: 52521 healthy loans and 52521 high-risk loans.
	* Fitting another logistic regression model by using the resampled training data. Another logistic regression model was fitted using the new resampled data.
	* Making predictions on the test data.The fitted model above was use to predict the loan status of the test features. The results were saves as a varible y_pred_ros.
	* Evaluating the model’s performance by using the same metrics as the first model. The y_test (actual labels) together with the y_pred_ros (predicted labels by the second logistic regression model) were used to evaluate the models performance.
	
## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1: Logistic Regression with Original Training Data
	* Balanced Accuracy Score: 0.951
	* Precision for Healthy Loans (0): 1.00
	* Precision for High-Risk Loans (1): 0.85
	* Recall for Healthy Loans (0): 0.99
	* Recall for High-Risk Loans (1): 0.91
	* F1 Score for Healthy Loans (0): 1.00
	* F1 Score for High-Risk Loans (1): 0.88



* Machine Learning Model 2: Logistic Regression with Resampled Training Data
	* Balanced Accuracy Score: 0.994
	* Precision for Healthy Loans (0): 1.00
	* Precision for High-Risk Loans (1): 0.85
	* Recall for Healthy Loans (0): 0.99
	* Recall for High-Risk Loans (1): 0.99
	* F1 Score for Healthy Loans (0): 1.00
	* F1 Score for High-Risk Loans (1): 0.92

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

* The results of the analysis show that the logistic regression model fitted with resampled training data performed better than the logistic regression model fitted with original training data on all metrics except precision for the high-risk class. The resampled model had higher balanced accuracy score, recall, specificity, F1 score, geometric mean, and index balanced accuracy than the original model for both classes, which means that it was more accurate, sensitive, specific, balanced, and adjusted in predicting both healthy and high-risk loans.

* Based on these results, I recommend using the logistic regression model fitted with resampled training data to predict the credit risk of loan applicants. This model seems to perform best because it has higher balanced accuracy score and F1 score than the original model for both classes, which are good metrics to compare the performance of models because they balance both precision and recall.

* The performance depends on the problem, that we are trying to solve. For example, in this case, we are more interested in minimizing false negatives (i.e predicting healthy loans when they are actually high-risk loans), hence we should use a model that has high recall for the high-risk class.I belive that recall is more important than precision for the high-risk class. Therefore, I still recommend using the resampled model, which has higher recall for the high-risk class than the original model.

