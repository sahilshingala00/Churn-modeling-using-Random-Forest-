# Churn-modeling-using-Random-Forest

Technical Configration:
Python version : python 3.7
Environment : Spider
Source of Dataset: https://www.kaggle.com/shrutimechlearn/churn-modelling


About DataSet:
Data Size: 10000
Field consider in Data set are ->
1)CustomerId	
2)Surname
3)CreditScore	
4)Geography	
5)Gender
6)Age
7)Tenure
8)Balance	
9)NumOfProducts	
10)HasCrCard	
11)IsActiveMember	
12)EstimatedSalary
13)Exited

Machine Learning Part:
Algorithm : Random Forest
Estimator size : 1000
Test data Ratio: 20%
libraries : Numpy & Pandas

Steps of execution:

1)Importing the libraries
2)Data Preprocessing
3)Import the dataset
4)Encoding categorical data
5)1.Label Encoding the "Gender" column
6)2.One Hot Encoding the "Geography" column
7)Feature Scaling
8)Splitting the dataset into the Training set and Test set
9)Training the Random Forest Classification model on the Training set
10)Prediction
11)Making the Confusion Matrix

Results:
Accuracy : 86-88%
