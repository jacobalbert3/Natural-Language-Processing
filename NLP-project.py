#PROJECT TITLE: PREDICTING EXAM TRUE/FALSE ANSWERS USING NATURAL LANGUAGE PROCESSING
#AUTHOR: Jacob Albert


#PROJECT DESCRIPTION
    # Often, when reading True/False questions, there are subtle clues based on the 
    # wording of the question. The goal of this project is to see if, using machine learning,
    # a program can predict answer to T/F questions better than an average guess (50%) correctness
    
#RESULTS
    #with a p value, less than .001, we reject the null hypothesis and confirm that, given
    #the relevant data, the program can predict T/F answers better than an average guess of 50%.



#PROCESS OUTLINED BELOW
#importing the different libraries we need
import numpy as np
import pandas as pd
from scipy.stats import norm
import math
#used to split data into training/testing data
from sklearn.model_selection import train_test_split
#convert the written data into numerical values (computers dont understand words, but numbers)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#use test data to evaluate the model in order to find how well it is performing
from sklearn.metrics import accuracy_score


#Explanation of dataset: 
    #I created a dataset with sample true/false questions from several sources including quizlet. In order to provide 
    #a broad range of topics, I included T/F questions on the following subjects: biology, computer science, biophysics, psychology, chemistry, linguistics, European history, Accounting, genetics, 
    #finance, geography, music, environmental science, philosophy, government, astronomy, economics, law, sales, religion, film, education
    #criminology, advertising, sociology, geology, consumers 


#STEP 1: DATA COLLECTION (FROM CSV)
#use the pandas dataframe
questions_data = pd.read_csv('/Users/jalbert21/Desktop/code/nlpproject.csv')

#Label Encoding
#Label false answers as 0 and true answers as 1
questions_data.loc[questions_data['Answer']=='False', 'Answer',] = 0
questions_data.loc[questions_data['Answer']=='True', 'Answer',] = 1


#seperating the data as text and label
X = questions_data['Question']
Y = questions_data['Answer']

#STEP 2: SPLITTING THE DATA INTO TRAINING DATA AND TEST DATA

#Test ratio is arbitrary (percentage of data set leaving for testing rather than training)
TEST_RATIO = .45
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = TEST_RATIO, random_state = 2)


#STEP 3: CONVERT TEXT DATA INTO NUMERICAL VALUES 
feature_extraction = TfidfVectorizer(min_df=1, lowercase= True)
#fit the data 
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)


#STEP 4: training the model (logistic legression)

model = LogisticRegression()

#training the logistic regression model with the training data
model.fit(X_train_features, Y_train)

#STEP 5: Evaluating the trained model:

#prediction on training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on Training Data: ', accuracy_on_training_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on Testing Data: ', accuracy_on_test_data)


#function used to calculate the p value given the hypothesis test and a threshold (in this case will be .5)
def hypothesis_test(accuracy, threshold):
    # standard error 
    n = questions_data.shape[0] * TEST_RATIO
    standard_error = math.sqrt(((threshold * (1 - threshold))/n))
    result = (accuracy - threshold)/standard_error
    print('z score is equal to: ', result )
    print('p value is equal to', 1 - norm.cdf(result))

hypothesis_test(accuracy_on_test_data, .5)



#ATTEMPING TO RATIONALIZE
#EXAMPLE: Some words in true or false questions may imply false regardless of the information provided. When words like "all" , "always", and "none
    #appear, the sentence has a higher chance of being false. For example in the below code, you can see how the program correctly predicts
    # that the following 3 sentences are false, even though these questions are arbitrary.  

#EXPECTING FALSE ANSWERS:

input_example = ["All people are bad", "people are always bad", "none of the people are good"]
#convert text to feature vectors
input_data_features = feature_extraction.transform(input_example)
#making prediction:
prediction = model.predict(input_data_features)
print('expected result:[0 0 0], actual result: ', prediction)