
import random
import sys
import csv
import math
import copy
import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.linear_model import LogisticRegression

def line_graph(x, y, title):

    #Graphing
    #plt.plot(x, y)
    n = len(x)
    a = np.arange(n)
    plt.errorbar(a, y, xerr=0.022, yerr=0.04)
    plt.title(title)
    plt.xticks(a, x)
    plt.xlabel('C value')
    plt.ylabel('error rate')
    plt.show()

def bar_graph(x, y, title):
    
    error = [.3, .3, .3, .3, .3, .3, .3, .3, .3]   

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x, y, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Error rate')
    ax.set_xlabel('C value')
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.set_title(title)
    ax.yaxis.grid(True)

    plt.show()

def logistic_regression(train_data, train_labels, test_data, test_labels, hyperparam, index):

    if(index == 0):
        #Fit the model
        clf = LogisticRegression(C=hyperparam, solver='lbfgs', multi_class='ovr').fit(train_data, train_labels)
    else:
        clf = LogisticRegression(C=hyperparam, solver='lbfgs', multi_class='multinomial').fit(train_data, train_labels)

    predicted_labels = clf.predict(test_data)
    
    mean_error_rate, mean_stand_dev = calculate_mean_and_stand(predicted_labels, test_labels)

    return mean_error_rate, mean_stand_dev

def perceptron(train_data, train_labels, test_data, test_labels, hyperparam, index):
    penalty = 12

    print("Perceptron")


def svm(train_data, train_labels, test_data, test_labels, hyperparam, index):
    print("svm")

def knn(train_data, train_labels, test_data, test_labels, hyperparam, index):
    print("knn")

def calculate_mean_and_stand(predicted_labels, test_labels):
    #Calulate mean predicted value
    total_error_rate = 0
    for i in range(len(predicted_labels)):
        error = abs(test_labels[i] - predicted_labels[i])
        total_error_rate = total_error_rate + error

    mean_error_rate = total_error_rate / len(predicted_labels)

    #TODO: Calculate standard deviation
    total_stand_dev = 0


    mean_stand_dev = total_stand_dev / len(predicted_labels)

    #return mean error rate and standard deviation

    return mean_error_rate, mean_stand_dev
    
def divide_data(data, titles, hyperparams, algo):

    #Main loop to go through all datasets
    for i in range(len(data)):

        dataset = data[i]
        
        error_rates = []
        stand_devs = []

        #TODO loop through c values
        for h in hyperparams:

            #Create window for cross validation
            total_len_data = len(dataset.data)
            print("What is total_len_data", total_len_data)
            print("Actual window value", total_len_data / 5)
            window = round(total_len_data / 5)
            print("And what is window", window)
            initial_window = window


            error_rate_log_regress = 0
            stand_dev_log_regress = 0

            #5 fold cross validation
            for j in range(5):
                #print(dataset.data)
                test_data = np.array(dataset.data[window - initial_window: window])
                test_labels = np.array(dataset.target[window - initial_window: window])

                print("Length of test", len(test_data))

                part1train_data = np.array(dataset.data[0 :window - initial_window])
                part2train_data = np.array(dataset.data[window:])

                part1train_labels = np.array(dataset.target[0 :window - initial_window])
                part2train_labels = np.array(dataset.target[window:])

                train_data = np.concatenate((part1train_data, part2train_data))
                train_labels = np.concatenate((part1train_labels, part2train_labels))
                #train = dataset.data[0 :window - initial_window] + dataset.data[window:]
                

                print("Length of train", len(train_data))

                #TODO: call the correct algo
                if(algo == "log"):
                    error_rate, stand_dev = logistic_regression(train_data, train_labels, test_data, test_labels, h, i)
                
                elif(algo == "percep"):
                    error_rate, stand_dev = perceptron(train_data, train_labels, test_data, test_labels, h, i)
                
                elif(algo == "svm"):
                    error_rate, stand_dev = svm(train_data, train_labels, test_data, test_labels, h, i)

                else:
                    error_rate, stand_dev = knn(train_data, train_labels, test_data, test_labels, h, i)

                error_rate_log_regress = error_rate_log_regress + error_rate
                stand_dev_log_regress = stand_dev_log_regress + stand_dev

                window += initial_window
            
            #Calculate mean error rate and stand dev per c value
            mean_error_rate = error_rate_log_regress / 5
            mean_stand_dev = stand_dev_log_regress / 5

            #Add to the arrays that will be used to graph
            error_rates.append(mean_error_rate)
            stand_devs.append(mean_stand_dev)
                
        print("What are the mean_error_rates", error_rates)
        print("What are the stand_devs", stand_devs)


        #plot mean classiﬁcation error rate and standard deviation (as error bars) across the 5 folds
        #points on the plots are hyperparameters
        title = "Mean classification error rate for " + titles[i]
        line_graph(hyperparams, error_rates, title)




def main():
    print("Main method")

    #Load all four datasets
    breast_cancer = datasets.load_breast_cancer()
    iris = datasets.load_iris()
    digits = datasets.load_digits()
    wine = datasets.load_wine()

    #data = [wine]
    data = [breast_cancer, iris, digits, wine]

    titles = ["Breast cancer", "Iris", "Digits", "Wine"]

    #LOGISTIC REGRESSION
    c_log_regress = [.00001, .0001, .001, .01, .1, 1, 10, 100, 1000]
    divide_data(data, titles, c_log_regress, "log")   

    #PERCEPTRON
    penalty = 12
    alpha = [.00001, .0001, .001, .01, .1, 1, 10, 100, 1000]
    divide_data(data, titles, alpha, "percep")

    #SVM
    c_svm = [.00001, .0001, .001, .01, .1, 1, 10, 100, 1000]
    divide_data(data, titles, c_svm, "svm")

    #KNN
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    divide_data(data, titles, x, "knn")

    

main()