
import random
import sys
import csv
import math
import copy
import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.linear_model import LogisticRegression

def graph(x, y, title):
    #Graphing
    plt.plot(x, y)

    #TODO: fix all the titles
    plt.title(title)
    plt.xlabel('C value')
    plt.ylabel('error rate')
    plt.show()

def logistic_regression(train_data, train_labels, test_data, test_labels, cVal):

    #Fit the model
    print("What is cVal", cVal)
    clf = LogisticRegression(C=cVal).fit(train_data, train_labels)
    predicted_labels = clf.predict(test_data)
    
    #Calulate mean predicted value
    total_error_rate = 0
    for i in range(len(predicted_labels)):
        error = abs(test_labels[i] - predicted_labels[i])
        total_error_rate = total_error_rate + error

    mean_error_rate = total_error_rate / len(predicted_labels)
    print("cVal for the following error", cVal)
    print("What is the mean error rate", mean_error_rate)

    #Calculate standard deviation
    total_stand_dev = 0


    mean_stand_dev = total_stand_dev / len(predicted_labels)

    #return mean error rate and standard deviation

    return mean_error_rate, mean_stand_dev
    

def perceptron():
    penalty = 12
    alpha = [.00001, .0001, .001, .01, .1, 1, 10, 100, 1000]

    print("Perceptron")

def svm():
    c = [.00001, .0001, .001, .01, .1, 1, 10, 100, 1000]
    print("svm")

def knn():
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    print("knn")

def main():
    print("Main method")

    #Load all four datasets
    breast_cancer = datasets.load_breast_cancer()
    iris = datasets.load_iris()
    digits = datasets.load_digits()
    wine = datasets.load_wine()

    data = [breast_cancer]
    #data = [breast_cancer, iris, digits, wine]

    #c_log_regress = [.00001, .0001, .001, .01, .1, 1, 10, 100, 1000]
    c_log_regress = [.00001]


    #LOGISTIC REGRESSION
    #Main loop to go through all datasets
    for i in range(len(data)):

        dataset = data[i]

        #Create window for cross validation
        total_len_data = len(dataset.data)
        print("What is total_len_data", total_len_data)
        print("Actual window value", total_len_data / 5)
        window = round(total_len_data / 5)
        print("And what is window", window)
        initial_window = window
        
        error_rates = []
        stand_devs = []

        #TODO loop through c values
        for c in range(len(c_log_regress)):

            error_rate_log_regress = 0
            stand_dev_log_regress = 0

            #5 fold cross validation
            for j in range(5):
                #print(dataset.data)
                test_data = np.array(dataset.data[window - initial_window: window])
                test_labels = np.array(dataset.target[window - initial_window: window])
                #print("What is test", test)
                print("Length of test", len(test_data))

                part1train_data = np.array(dataset.data[0 :window - initial_window])
                part2train_data = np.array(dataset.data[window:])

                part1train_labels = np.array(dataset.target[0 :window - initial_window])
                part2train_labels = np.array(dataset.target[window:])

                train_data = np.concatenate((part1train_data, part2train_data))
                train_labels = np.concatenate((part1train_labels, part2train_labels))
                #train = dataset.data[0 :window - initial_window] + dataset.data[window:]
                
                

                print("Length of train", len(train_data))

                #Call logistic regression
                error_rate, stand_dev = logistic_regression(train_data, train_labels, test_data, test_labels, c)

                error_rate_log_regress = error_rate_log_regress + error_rate
                stand_dev_log_regress = stand_dev_log_regress + stand_dev

                window += initial_window
            
            #Calculate mean error rate and stand dev per c value
            mean_error_rate = error_rate_log_regress / 5
            mean_stand_dev = stand_dev_log_regress / 5

            #Add to the arrays that will be used to graph
            error_rates.append(mean_error_rate)
            stand_devs.append(mean_stand_dev)
                

        #plot mean classiﬁcation error rate and standard deviation (as error bars) across the 5 folds
        #points on the plots are hyperparameters



    #PERCEPTRON

    #SVM

    #KNN


    print(len(breast_cancer.data))
    

main()