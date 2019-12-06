
import random
import sys
import csv
import math
import copy
import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

def line_graph(x, y, stand_devs, title):

    #Graphing
    #plt.plot(x, y)
    n = len(x)
    a = np.arange(n)
    plt.errorbar(a, y, yerr=stand_devs)
    plt.title(title)
    plt.xticks(a, x)
    plt.xlabel('Hyperparameter')
    plt.ylabel('Error rate')
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
        clf = LogisticRegression(max_iter=1000, C=hyperparam, solver='lbfgs', multi_class='ovr').fit(train_data, train_labels)
    else:
        clf = LogisticRegression(C=hyperparam, solver='lbfgs', multi_class='multinomial').fit(train_data, train_labels)

    predicted_labels = clf.predict(test_data)

    mean_error_rate = 1 - clf.score(test_data, test_labels)
    


    return mean_error_rate

def perceptron(train_data, train_labels, test_data, test_labels, hyperparam, index):

    clf = Perceptron(alpha=hyperparam, penalty="l2").fit(train_data, train_labels)
    
    predicted_labels = clf.predict(test_data)
    mean_error_rate = 1 - clf.score(test_data, test_labels)



    return mean_error_rate



def svm(train_data, train_labels, test_data, test_labels, hyperparam, index):

    clf = LinearSVC(C=hyperparam).fit(train_data, train_labels)

    predicted_labels = clf.predict(test_data)
    
    mean_error_rate = 1 - clf.score(test_data, test_labels)


    return mean_error_rate

def knn(train_data, train_labels, test_data, test_labels, hyperparam, index):
    k = 6 * hyperparam + 1

    clf = KNeighborsClassifier(n_neighbors=k).fit(train_data, train_labels)

    predicted_labels = clf.predict(test_data)

    mean_error_rate = 1 - clf.score(test_data, test_labels)
    
    return mean_error_rate

    
def divide_data(data, titles, hyperparams, algo):

    #Main loop to go through all datasets
    for i in range(len(data)):

        dataset = data[i]
        
        error_rates = []
        stand_devs = []

        #loop through hyperparameters values
        for h in hyperparams:

            #Create window for cross validation
            total_len_data = len(dataset.data)
            print("What is total_len_data", total_len_data)
            print("Actual window value", total_len_data / 5)
            window = round(total_len_data / 5)
            print("And what is window", window)
            initial_window = window


            error_rate_log_regress = 0
            error_rate_stand_dev = []

            #5 fold cross validation
            for j in range(5):
                #print(dataset.data)
                test_data = np.array(dataset.data[int(window - initial_window): int(window)])
                test_labels = np.array(dataset.target[int(window - initial_window): int(window)])

                print("Length of test", len(test_data))

                part1train_data = np.array(dataset.data[0 :int(window - initial_window)])
                part2train_data = np.array(dataset.data[int(window):])

                part1train_labels = np.array(dataset.target[0 :int(window - initial_window)])
                part2train_labels = np.array(dataset.target[int(window):])

                train_data = np.concatenate((part1train_data, part2train_data))
                train_labels = np.concatenate((part1train_labels, part2train_labels))
                #train = dataset.data[0 :window - initial_window] + dataset.data[window:]
                

                print("Length of train", len(train_data))

                algo_type = ""
                #Call the correct algo
                if(algo == "log"):
                    algo_type = "Logistic Regression"
                    error_rate = logistic_regression(train_data, train_labels, test_data, test_labels, h, i)
                
                elif(algo == "percep"):
                    algo_type = "Perceptron"
                    error_rate = perceptron(train_data, train_labels, test_data, test_labels, h, i)
                
                elif(algo == "svm"):
                    algo_type = "SVM"
                    error_rate = svm(train_data, train_labels, test_data, test_labels, h, i)

                else:
                    algo_type = "KNN"
                    error_rate = knn(train_data, train_labels, test_data, test_labels, h, i)

                error_rate_log_regress = error_rate_log_regress + error_rate
                error_rate_stand_dev.append(error_rate)

                window += initial_window
            
            #Calculate mean error rate 
            mean_error_rate = error_rate_log_regress / 5

            #Calculate stand dev per c value
            squared_diffs = []
            for k in range(len(error_rate_stand_dev)):
                value = (error_rate_stand_dev[k] - mean_error_rate) * (error_rate_stand_dev[k] - mean_error_rate)
                squared_diffs.append(value)
            
            stand_dev = math.sqrt(sum(squared_diffs) / len(squared_diffs))

            #Add to the arrays that will be used to graph
            error_rates.append(mean_error_rate)
            stand_devs.append(stand_dev)
                
        print("What are the mean_error_rates", error_rates)
        print("What are the stand_devs", stand_devs)


        #plot mean classiﬁcation error rate and standard deviation (as error bars) across the 5 folds
        #points on the plots are hyperparameters
        title = "Mean classification error rate for " + titles[i] + " for " + algo_type
        line_graph(hyperparams, error_rates, stand_devs, title)




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
    #alpha = [.00001, .0001, .001, .01, .1, 1, 10, 100, 1000]
    #divide_data(data, titles, alpha, "percep")

    #SVM
    #c_svm = [.00001, .0001, .001, .01, .1, 1, 10, 100, 1000]
    #divide_data(data, titles, c_svm, "svm")

    #KNN
    #x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    #divide_data(data, titles, x, "knn")

    

main()