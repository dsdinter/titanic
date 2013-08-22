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

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
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

csv_file_object = csv.reader(open('Data/train.csv', 'rb')) #Load in the training csv file
header = csv_file_object.next() #Skip the fist line as it is a header
train_data=[] #Creat a variable called 'train_data'
for row in csv_file_object: #Skip through each row in the csv file
    train_data.append(row[1:]) #adding each row to the data variable
train_data = np.array(train_data) #Then convert from a list to an array

#I need to convert all strings to integer classifiers:
#Male = 1, female = 0:
train_data[train_data[0::,3]=='male',3] = 1
train_data[train_data[0::,3]=='female',3] = 0
#embark c=0, s=1, q=2
train_data[train_data[0::,10] =='C',10] = 0
train_data[train_data[0::,10] =='S',10] = 1
train_data[train_data[0::,10] =='Q',10] = 2

#I need to fill in the gaps of the data and make it complete.
#So where there is no price, I will assume price on median of that class
#Where there is no age I will give median of all ages

#All the ages with no data make the median of the data
train_data[train_data[0::,4] == '',4] = np.median(train_data[train_data[0::,4]\
                                           != '',4].astype(np.float))
#All missing ebmbarks just make them embark from most common place
train_data[train_data[0::,10] == '',10] = np.round(np.mean(train_data[train_data[0::,10]\
                                                   != '',10].astype(np.float)))

train_data = np.delete(train_data,[2,7,9],1) #remove the name data, cabin and ticket
#I need to do the same with the test data now so that the columns are in the same
#as the training data

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
test_data[test_data[0::,2]=='female',2] = 0
#ebark c=0, s=1, q=2
test_data[test_data[0::,9] =='C',9] = 0 #Note this is not ideal, in more complex 3 is not 3 tmes better than 1 than 2 is 2 times better than 1
test_data[test_data[0::,9] =='S',9] = 1
test_data[test_data[0::,9] =='Q',9] = 2

#All the ages with no data make the median of the data
test_data[test_data[0::,3] == '',3] = np.median(test_data[test_data[0::,3]\
                                           != '',3].astype(np.float))
#All missing ebmbarks just make them embark from most common place
test_data[test_data[0::,9] == '',9] = np.round(np.mean(test_data[test_data[0::,9]\
                                                   != '',9].astype(np.float)))
#All the missing prices assume median of their respectice class
for i in xrange(np.size(test_data[0::,0])):
    if test_data[i,7] == '':
        test_data[i,7] = np.median(test_data[(test_data[0::,7] != '') &\
                                             (test_data[0::,0] == test_data[i,0])\
            ,7].astype(np.float))

test_data = np.delete(test_data,[1,6,8],1) #remove the name data, cabin and ticket

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
              "KNeighborsClassifier__algorithm": ["ball_tree", "kd_tree", "brute",],
              "KNeighborsClassifier__leaf_size": sp_randint(1, 100),
              
              }

# run randomized search
n_iter_search = 20000000
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
random_search.fit(train_data[0::,1::],train_data[0::,0])

print 'Reporting'
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)

print 'Predicting'
output = random_search.predict(test_data)

open_file_object = csv.writer(open("pipelinearsvcknn.csv", "wb"))
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
