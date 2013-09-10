""" Writing my first randomforest code.
Author : AstroDave
Date : 23rd September, 2012
please see packages.python.org/milk/randomforests.html for more

""" 

import numpy as np
import csv as csv

from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint

from sklearn import preprocessing

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import RandomizedSearchCV

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def main():
    csv_file_object = csv.reader(open('Data/train.csv', 'rb')) #Load in the training csv file
    header = csv_file_object.next() #Skip the fist line as it is a header
    train_data=[] #Creat a variable called 'train_data'
    for row in csv_file_object: #Skip through each row in the csv file
        train_data.append(row[1:]) #adding each row to the data variable
    train_data = np.array(train_data) #Then convert from a list to an array
    
    #I need to convert all strings to integer classifiers:
    #Male = 1, female = 0:
    train_data[train_data[0::,3]=='male',3] = -1
    train_data[train_data[0::,3]=='female',3] = 1
    #embark c=0, s=1, q=2
    train_data[train_data[0::,10] =='C',10] = -1
    train_data[train_data[0::,10] =='S',10] = 0
    train_data[train_data[0::,10] =='Q',10] = 1
    #Survived
    train_data[train_data[0::,3]==1,0] = 1
    train_data[train_data[0::,3]==0,0] = -1
    
    #I need to fill in the gaps of the data and make it complete.
    #So where there is no price, I will assume price on median of that class
    #Where there is no age I will give median of all ages
    
    imp = preprocessing.Imputer(missing_values=0, strategy='median', axis=0)
    
    #All the ages with no data make the median of the data
    #train_data[train_data[0::,4] == '',4] = np.median(train_data[train_data[0::,4]\
    #                                          != '',4].astype(np.float))
    #All missing ebmbarks just make them embark from most common place
    #train_data[train_data[0::,10] == '',10] = np.round(np.mean(train_data[train_data[0::,10]\
    #                                                   != '',10].astype(np.float)))
    
    train_data = np.delete(train_data,[2,7,9,10],1) #remove the name data, cabin and ticket
    train_data[train_data=='']='0'
    imp.fit_transform(train_data)
    #I need to do the same with the test data now so that the columns are in the same
    #as the training data
    
    
    
    #We finally spit the data between train set and valiation set
    x_train, x_test, y_train, y_test=train_test_split(
        train_data[0::,1::],train_data[0::,0], test_size=0.2, random_state=0)
    
    #Standardise data
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train_std=scaler.transform(x_train)
    x_test_std=scaler.transform(x_test)
    
    
    test_file_object = csv.reader(open('Data/test.csv', 'rb')) #Load in the test csv file
    header = test_file_object.next() #Skip the fist line as it is a header
    test_data=[] #Creat a variable called 'test_data'
    ids = []
    for row in test_file_object: #Skip through each row in the csv file
        ids.append(row[0])
        test_data.append(row[1:]) #adding each row to the data variable
    test_data = np.array(test_data) #Then convert from a list to an array
    
    #I need to convert all strings to integer classifiers:
    #Male = 1, female = 0:
    test_data[test_data[0::,2]=='male',2] = 1
    test_data[test_data[0::,2]=='female',2] = -1
    #ebark c=0, s=1, q=2
    test_data[test_data[0::,9] =='C',9] = -1 #Note this is not ideal, in more complex 3 is not 3 tmes better than 1 than 2 is 2 times better than 1
    test_data[test_data[0::,9] =='S',9] = 0
    test_data[test_data[0::,9] =='Q',9] = 1
    
    #All the ages with no data make the median of the data
    #test_data[test_data[0::,3] == '',3] = np.median(test_data[test_data[0::,3]\
    #                                           != '',3].astype(np.float))
    #All missing ebmbarks just make them embark from most common place
    #test_data[test_data[0::,9] == '',9] = np.round(np.mean(test_data[test_data[0::,9]\
    #                                                   != '',9].astype(np.float)))
    #All the missing prices assume median of their respectice class
    #for i in xrange(np.size(test_data[0::,0])):
    #    if test_data[i,7] == '':
    #        test_data[i,7] = np.median(test_data[(test_data[0::,7] != '') &\
    #                                             (test_data[0::,0] == test_data[i,0])\
    #            ,7].astype(np.float))
    
    test_data = np.delete(test_data,[1,6,8,9],1) #remove the name data, cabin and ticket
    test_data[test_data=='']='0'
    #Impute mising values
    imp.fit_transform(test_data)
    
    #Standarize
    scaler_test = preprocessing.StandardScaler().fit(test_data)
    test_data_std=scaler_test.transform(test_data)
    #The data is now ready to go. So lets train then test!
    
    start = time()
    print 'Training estimators'
    estimators = [('linearsvc', LinearSVC()), ('KNeighborsClassifier', KNeighborsClassifier())]
    clf = Pipeline(estimators)
    # specify parameters and distributions to sample from
    param_dist = {"linearsvc__C": sp_randint(1, 1000),
                  "linearsvc__loss": ["l1", "l2"],
                  "linearsvc__dual": [True],
                  "KNeighborsClassifier__n_neighbors": sp_randint(5, 100),
                  "KNeighborsClassifier__weights": ["uniform", "distance"],
                  "KNeighborsClassifier__algorithm": ["ball_tree", "kd_tree", "brute"],
                  "KNeighborsClassifier__leaf_size": sp_randint(3, 100),
                  
                  }
    
    # run randomized search
    n_iter_search = 2000
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search,n_jobs=4, verbose=1)
    random_search.fit(x_train_std,y_train)
    
    print 'Reporting'
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.grid_scores_)
    score=random_search.score(x_test_std,y_test)
    print 'Test score'
    print score
    print 'Predicting'
    output = random_search.predict(test_data_std)
    
    open_file_object = csv.writer(open("pipelinearsvcknn.csv", "wb"))
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(ids, output))

if __name__ == "__main__":
    main()
