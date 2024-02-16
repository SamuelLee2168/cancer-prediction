import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.sidebar.subheader('Table of Contents')
st.sidebar.write('1. ','<a href=#introduction>Introduction</a>', unsafe_allow_html=True)
st.sidebar.write('2. ','<a href=#code-of-the-k-nearest-neighbors-machine-learning-model>Code of the K Nearest Neighbors Machine Learning Model</a>', unsafe_allow_html=True)
st.sidebar.write('3. ','<a href=#interactive-prediction-of-breast-cancer>Interactive Prediction of Breast Cancer</a>', unsafe_allow_html=True)




st.title("Predicting Breast Cancer using KNN")
st.header("Introduction")
st.markdown("In this case study, I will build a k nearest neighbors machine learning model that will predict whether a patient has breast cancer using a series of factors.")
st.markdown("I will train and test my model by using a dataset of patients who were suspected to have breast cancer.")
st.markdown("Source:")
st.markdown("https://www.kaggle.com/uciml/breast-cancer-wisconsin-data")
st.markdown("Here is the dataset:")
cancer = pd.read_csv('data.csv').drop('Unnamed: 32',axis=1)
cancer
st.markdown('Each row of the dataset represents a patient who was suspected to have breast cancer.')
st.markdown('The "diagnosis" column is what diagnosis the patient actually got in the end. "M" stands for malignant, which means the patient did have cancer, while "B" stants for benign, meaning that the patient did not have cancer.')
st.markdown('The other columns are data of each patient like the radius mean or texture mean of the cell nuclei. The machine learning model will use these factors to predict whether a patient has breast cancer.')

st.header("Code of the K Nearest Neighbors Machine Learning Model")

data_cleaning_code = """
#Imports the data
cancer_data = pd.read_csv('data.csv').drop('Unnamed: 32',axis=1)

#This function will standardize a series.
def standardize(series):
    return (series-np.average(series))/np.std(series)

#Setting the list of factors for prediction
factors = ['radius_mean', 'texture_mean', 'perimeter_mean','area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean','concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

#Standardizing all of the features
standardized = pd.DataFrame()
standardized['id'] = cancer['id']
standardized['diagnosis'] = cancer['diagnosis']
for column_name in factors:
    standardized[column_name] = standardize(cancer[column_name])
    
#This function will shuffle the dataset and split 80% of the data into a training set and 20% of the data into a test set.
def pick_train_and_test():
    row_num = standardized.shape[0]
    shuffled = standardized.sample(row_num).reset_index().drop('index',axis=1)
    train_set = shuffled.iloc[0:int(row_num*0.8)]
    test_set = shuffled.iloc[int(row_num*0.8):row_num+1]
    return (train_set,test_set)

#Using the pick_train_and_test function
train_set,test_set = pick_train_and_test()
"""

#Imports the data
cancer_data = pd.read_csv('data.csv').drop('Unnamed: 32',axis=1)

#This function will standardize a series.
def standardize(series):
    return (series-np.average(series))/np.std(series)

#Setting the list of factors for prediction
factors = ['radius_mean', 'texture_mean', 'perimeter_mean','area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean','concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

#Standardizing all of the factors
standardized = pd.DataFrame()
standardized['id'] = cancer['id']
standardized['diagnosis'] = cancer['diagnosis']
for column_name in factors:
    standardized[column_name] = standardize(cancer[column_name])
    
#This function will shuffle the dataset and split 80% of the data into a training set and 20% of the data into a test set.
def pick_train_and_test():
    row_num = standardized.shape[0]
    shuffled = standardized.sample(row_num).reset_index().drop('index',axis=1)
    train_set = shuffled.iloc[0:int(row_num*0.8)]
    test_set = shuffled.iloc[int(row_num*0.8):row_num+1]
    return (train_set,test_set)

#Using the pick_train_and_test function
train_set,test_set = pick_train_and_test()


with st.expander("Code for data cleaning"):
    st.code(data_cleaning_code)
    
    st.markdown("Training set:")
    train_set
    st.markdown("Test set:")
    test_set

functions_for_prediction = """
#The distance function will calculate the euclidean distance between 2 arrays 
def distance(features1,features2):
    return np.sqrt(sum((features1-features2)**2))

#The distance_for_dataframe function will calculate and return an array 
#of the euclidean distance between one given row and the rest of the dataset
def distance_for_dataframe(tested_row,train_rows):
    distances = []
    for i in np.arange(train_rows.shape[0]):
        distances.append(distance(tested_row,train_rows.iloc[i]))
    return distances

#The classify function takes a tested row, some train rows, 
#an array of classifications, and a k value. 
#It will classify whether the patient is malignant of benign.
def classify(tested_row,train_rows,train_classes,k):
    #This chunk of code creates a dataframe where:
    #the "diagnosis" column is the classification for the other patients in the training rows
    #the "distance" column is the distance between the other patients and the given patient
    distances = pd.DataFrame()
    distances['diagnosis'] = train_classes
    distances['distance'] = distance_for_dataframe(tested_row,train_rows)
    
    #This chunk of code finds the k amount of patients most similar 
    #to the inputted patient and returns the most common classification of the nearest patients.
    top_distances = distances.sort_values('distance').head(k)
    number_of_malignant = sum(top_distances['diagnosis'] == 'M')
    number_of_benign = sum(top_distances['diagnosis'] == 'B')
    if number_of_malignant >= number_of_benign:
        return 'M'
    else:
        return 'B'
"""


with st.expander("Functions for predicting breast cancer"):
    st.code(functions_for_prediction)
        

accuracy_code = """
#This function calculates the accuracy of a given k by using the 
#classify function on every row of the test set, 
#and comparing the estimate to the actual classification.
def calculate_accuracy(k):
    estimates = []
    for i in np.arange(test_set.shape[0]):
        estimate = classify(test_set.iloc[i].drop('id').drop('diagnosis'),train_set.drop('id',axis=1).drop('diagnosis',axis=1),train_set['diagnosis'],k)
        estimates.append(estimate)
    return sum(estimates==test_set['diagnosis'])/test_set.shape[0]

#This chunk of code calculates the accuracy of different k values ranging from 1-50. 
accuracies_for_different_k = []
for k in np.arange(1,51,1):
    accuracies_for_current_k = []
    #The accuracy of each k value is calculated 3 times for extra sample size.
    for i in np.arange(3):
        #The train and test set is re-shuffled before each accuracy test.
        train_set,test_set = pick_train_and_test() 
        accuracies_for_current_k.append(calculate_accuracy(k))
    accuracies_for_different_k.append(np.average(accuracies_for_current_k))
    
#This chunk of code organizes the accuracies_for_different_k dictionary and sorts them.
accuracy_dataframe = pd.DataFrame()
accuracy_dataframe['k'] = np.arange(1,51,1)
accuracy_dataframe['accuracy'] = accuracies_for_different_k
accuracy_dataframe = accuracy_dataframe.sort_values('accuracy',ascending=False)
"""


with st.expander("Code that finds the best K"):
    st.code(accuracy_code)
    st.markdown("Here is the dataframe of the accuracies for each k value:")
    accuracy_dataframe = pd.DataFrame()
    accuracies_for_different_k = [0.9093567251461988,
 0.8888888888888888,
 0.9473684210526315,
 0.9385964912280702,
 0.935672514619883,
 0.915204678362573,
 0.9532163742690059,
 0.9239766081871346,
 0.9415204678362573,
 0.9532163742690059,
 0.9473684210526315,
 0.935672514619883,
 0.9385964912280702,
 0.9473684210526315,
 0.935672514619883,
 0.9532163742690059,
 0.9473684210526315,
 0.956140350877193,
 0.9269005847953217,
 0.9444444444444445,
 0.912280701754386,
 0.9532163742690059,
 0.9385964912280702,
 0.9327485380116959,
 0.9327485380116958,
 0.9532163742690059,
 0.9415204678362573,
 0.9619883040935672,
 0.9239766081871346,
 0.9473684210526315,
 0.9269005847953217,
 0.9327485380116959,
 0.9385964912280702,
 0.9269005847953217,
 0.9239766081871345,
 0.9385964912280702,
 0.9473684210526315,
 0.9239766081871346,
 0.9619883040935672,
 0.915204678362573,
 0.935672514619883,
 0.9473684210526315,
 0.9385964912280702,
 0.9239766081871345,
 0.915204678362573,
 0.9473684210526315,
 0.9532163742690059,
 0.935672514619883,
 0.9181286549707602,
 0.935672514619883]
    
    
    accuracy_dataframe['k'] = np.arange(1,51,1)
    accuracy_dataframe['accuracy'] = accuracies_for_different_k
    accuracy_dataframe = accuracy_dataframe.sort_values('accuracy',ascending=False)
    accuracy_dataframe
    st.markdown("Note: I did not run the code shown above for you since it will take a long time to run. I ran the code myself and printed the values here.")
    
final_accuracy_test_code = """
#To make sure there is enough samples, I will run the accuracy test 20 times 
#using 28 as the k value.
accuracies = []
for i in np.arange(20):
    train_set,test_set = pick_train_and_test()
    accuracies.append(calculate_accuracy(28))
"""

with st.expander("Final accuracy test"):
    st.code(final_accuracy_test_code)
    st.markdown("Here's the average of the accuracies:")
    st.markdown("0.9377192982456138")
    st.markdown("Note: I didn't run the final accuracy test code for you since that would take too long to run. I ran the code myself and printed the result.")
    
st.header('Interactive Prediction of Breast Cancer')
st.markdown("You can enter different values in the form and press submit to get a prediction of malignant or benign.")

#The distance function will calculate the euclidean distance between 2 arrays 
def distance(features1,features2):
    return np.sqrt(sum((features1-features2)**2))

#The distance_for_dataframe function will calculate and return an array 
#of the euclidean distance between one given row and the rest of the dataset
def distance_for_dataframe(tested_row,train_rows):
    distances = []
    for i in np.arange(train_rows.shape[0]):
        distances.append(distance(tested_row,train_rows.iloc[i]))
    return distances

#The classify function takes a tested row, some train rows, 
#an array of classifications, and a k value. 
#It will classify whether the patient is malignant of benign.
def classify(tested_row,train_rows,train_classes,k):
    #This chunk of code creates a dataframe where:
    #the "diagnosis" column is the classification for the other patients in the training rows
    #the "distance" column is the distance between the other patients and the given patient
    distances = pd.DataFrame()
    distances['diagnosis'] = train_classes
    distances['distance'] = distance_for_dataframe(tested_row,train_rows)
    
    #This chunk of code finds the k amount of patients most similar 
    #to the inputted patient and returns the most common classification of the nearest patients.
    top_distances = distances.sort_values('distance').head(k)

    number_of_malignant = sum(top_distances['diagnosis'] == 'M')
    number_of_benign = sum(top_distances['diagnosis'] == 'B')

    if number_of_malignant >= number_of_benign:
        return 'M'
    else:
        return 'B'

def make_number_input(factor_name):
    return st.number_input(factor_name,value=np.average(cancer[factor_name]) ,min_value=min(cancer[factor_name]), max_value=max(cancer[factor_name]))


inputs = []

def standardize_input(series):
    output = []
    for i in np.arange(len(series)):
        new_column = np.append(cancer[factors[i]],series[i])
        standardized_new_column = standardize(new_column)
        output.append(standardized_new_column[-1])
    return output

with st.form("form 1"):
        col1,col2,col3=st.columns(3)
        with col1:
            for i in np.arange(4):
                inputted_value = st.number_input(factors[i],value=np.round(np.average(cancer[factors[i]]),2) ,min_value=min(cancer[factors[i]]), max_value=max(cancer[factors[i]]))
                inputs.append(inputted_value)
        with col2:
            for i in np.arange(4,7,1):
                inputted_value2 = st.number_input(factors[i],value=np.average(cancer[factors[i]]) ,min_value=min(cancer[factors[i]]), max_value=max(cancer[factors[i]]))
                inputs.append(inputted_value2)
        with col3:
            for i in np.arange(7,10,1):
                inputted_value3 = st.number_input(factors[i],value=np.average(cancer[factors[i]]) ,min_value=min(cancer[factors[i]]), max_value=max(cancer[factors[i]]))
                inputs.append(inputted_value3)
                
                
        submitted = st.form_submit_button("Submit")
if submitted:
    result = classify(standardize_input(inputs),train_set.drop('id',axis=1).drop('diagnosis',axis=1),train_set['diagnosis'],28)
    if result == "M":
        st.markdown("Malignant")
    else:
        st.markdown("Benign")
