# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 14:45:24 2021

@author: jajua

Homework 1: Group Homework
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import statistics

 
def get_result(k):
     
     X, y = load_iris(return_X_y=True)
     X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.7)
     knn = KNeighborsClassifier(n_neighbors=k)
     knn.fit(X_train,y_train)
     score = knn.score(X_test,y_test)
     
     return score
     
def get_avg_score(k):
    
    scores = []
    
    for _ in range(5):
        
        result = get_result(k)
        scores.append(result)
    
    return statistics.mean(scores)

def main():
    
    x = []
    y = []
    for k in range(1,21):
        
             result =get_avg_score(k)
             x.append(k)
             y.append(result)
    
    plt.title('KNN')
    plt.plot(x,y)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.show()
    


if __name__ == "__main__":
    
    main()
    