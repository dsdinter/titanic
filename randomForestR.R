library(randomForest)

#now I load in both the "train" and "test" data frames

train <- read.csv("Data/train.csv")
test <- read.csv("Data/test.csv")

#Now, randomForest can either model a regression (continuous number) or a classification (yes/no). 
#This competition is to predict survival. Either the person survived or they didn't: yes/no. 
#So, I want to make sure randomForest models a classification. To do this, I tell the program 
#that the "Survived" column (our dependent variable) is a factor, not a number. 

train$Survived <- as.factor(train$Survived)

#Function to impute missing values in train and test datasets
rfImpute(train, train$Survived, iter=5, ntree=300)
rfImpute(test, train$Survived, iter=5, ntree=300)
#Next, I decide what predictors (independant variables) I want to train my model with. 
#I decided to use Pclass, Sex, Age, SibSp, Parch, and Embarked as predictors.

#Additionally, I decide to include the relationships between Pclass:Sex and Pclass:Age and Age:Sex 
#just for kicks. I think women are more likely to survive than men ("Women and children first!") 
#but I think it's possible that a rich woman (Pclass = 1) might be more likely to survive than a 
#poor woman. So we'll model things like this in and see what we get. 

#Before I do anything else, I want to set a "seed" so that others can reproduce my results. 
#randomForest is RANDOM. If I don't set a seed value, then each time I run my model I might 
#get a slightly different result.

set.seed(107)

#Now I build my model

model <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Embarked + Pclass:Sex + Pclass:Age + Age:Sex, data=train, ntree=20000)

#Now I use the model to predict against test data. This enters the prediction as a new column 
#called “Survived” in the test data frame. 

test$Survived <- predict(model, newdata=test, type='class')

#Finally I save my updated test data frame as a .csv file

write.csv(test[,c("PassengerId", "Survived")], file="predictions.csv", row.names=FALSE, quote=FALSE)