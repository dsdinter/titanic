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

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
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
train_data[train_data[0::,3]=='male',3] = -1
train_data[train_data[0::,3]=='female',3] = 1
#embark c=0, s=1, q=2
train_data[train_data[0::,10] =='C',10] = -1
train_data[train_data[0::,10] =='S',10] = 0
train_data[train_data[0::,10] =='Q',10] = 1

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

train_data = np.delete(train_data,[2,5,6,7,9,10],1) #remove the name data, cabin and ticket
#Split data between rows containing age or not    
train_data_noage=train_data[train_data[0::,3] =='']
train_data=train_data[train_data[0::,3] !='']
train_data_noage = np.delete(train_data_noage,[3],1)
train_data[train_data=='']='0'
imp.fit_transform(train_data)

train_data_noage[train_data_noage=='']='0'
imp.fit_transform(train_data_noage)
#I need to do the same with the test data now so that the columns are in the same
#as the training data



#We finally spit the data between train set and valiation set
x_train, x_test, y_train, y_test=train_test_split(
    train_data[0::,1::],train_data[0::,0], test_size=0.2, random_state=0)
    
x_train_noage, x_test_noage, y_train_noage, y_test_noage=train_test_split(
    train_data_noage[0::,1::],train_data_noage[0::,0], test_size=0.2, random_state=0)

#Standardise data
scaler = preprocessing.StandardScaler().fit(x_train)
x_train_std=scaler.transform(x_train)
x_test_std=scaler.transform(x_test)

scaler_noage = preprocessing.StandardScaler().fit(x_train_noage)
x_train_std_noage=scaler_noage.transform(x_train_noage)
x_test_std_noage=scaler_noage.transform(x_test_noage)

test_file_object = csv.reader(open('Data/test.csv', 'rb')) #Load in the test csv file
header = test_file_object.next() #Skip the fist line as it is a header
test_data=[] #Creat a variable called 'test_data'
ids = []
for row in test_file_object: #Skip through each row in the csv file
    test_data.append(row[0:]) #adding each row to the data variable
test_data = np.array(test_data) #Then convert from a list to an array

#I need to convert all strings to integer classifiers:
#Male = 1, female = 0:
test_data[test_data[0::,3]=='male',3] = -1
test_data[test_data[0::,3]=='female',3] = 1
#ebark c=0, s=1, q=2
test_data[test_data[0::,10] =='C',10] = -1 #Note this is not ideal, in more complex 3 is not 3 tmes better than 1 than 2 is 2 times better than 1
test_data[test_data[0::,10] =='S',10] = 0
test_data[test_data[0::,10] =='Q',10] = 1

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

test_data = np.delete(test_data,[2,5,6,7,9,10],1) #remove the name data, cabin and ticket

#Split data between rows containing age or not    
test_data_noage=test_data[test_data[0::,3] =='',1::]
ids_noage=test_data[test_data[0::,3] =='',0]

ids=test_data[test_data[0::,3] !='',0]
test_data=test_data[test_data[0::,3] !='',1::]

test_data_noage = np.delete(test_data_noage,[3],1)
test_data[test_data=='']='0'
#Impute mising values    
imp.fit_transform(test_data)

test_data_noage[test_data_noage=='']='0'
imp.fit_transform(test_data_noage)    

#Standarize
scaler_test = preprocessing.StandardScaler().fit(test_data)
test_data_std=scaler_test.transform(test_data)

#Standarize
scaler_test_noage = preprocessing.StandardScaler().fit(test_data_noage)
test_data_std_noage=scaler_test_noage.transform(test_data_noage)

#The data is now ready to go. So lets train then test!

start = time()
print 'Training estimators'
estimators = [('svc', SVC())]
clf = Pipeline(estimators)
# specify parameters and distributions to sample from
param_dist = {"svc__C": sp_randint(1, 10),
              "svc__kernel": ["linear", "rbf","poly","sigmoid"],
              "svc__gamma": sp_randint(1, 10),
              "svc__coef0": sp_randint(1, 10),
              "svc__shrinking": [True, False]
              }
#Start with data with age
# run randomized search
n_iter_search = 20
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


#Finally with data without age
# run randomized search
n_iter_search = 2000
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search,n_jobs=4, verbose=1)
random_search.fit(x_train_std_noage,y_train_noage)

print 'Reporting noage'
print("RandomizedSearchCV noage took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)
score=random_search.score(x_test_std_noage,y_test_noage)
print 'Test score noage'
print score
print 'Predicting noage'
outputnoage = random_search.predict(test_data_std_noage)

print 'Output both prediction'
open_file_object = csv.writer(open("pipesvcknn.csv", "wb"))
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
open_file_object.writerows(zip(ids_noage, outputnoage))

