# [Xixi (Jialin), Liu]
# [20196588]
# [MMA]
# [2021W]
# [869]
# [2020-08]

# Submission to Question [7], PART [2]
if("pacman" %in% rownames(installed.packages()) == FALSE) {install.packages("pacman")} 
pacman::p_load("caret","ROCR","lift","glmnet","MASS","e1071","mice","gdata") 
library(pROC)
library(dplyr)
library(caret)
library(mice)
library(gdata)


OJ<-read.csv("C:\\Users\\liuxi\\OneDrive\\Desktop\\2020 MMA 869\\Assignment 1\\OJ.csv",na.strings=c(""," ","NA"), header=TRUE,stringsAsFactors = TRUE)
str(OJ)
dim(OJ)
#Check missing values, no missing value
md.pattern(OJ)
#Remove first col X
OJ<-OJ[,-1]
#Convert Purchase and Store7 to Binary varibles
OJ$Purchase<-ifelse(OJ$Purchase=='CH',0,1)
OJ$Purchase<-as.factor(OJ$Purchase)

#Creating Training and Testing set
set.seed(456)
inTrain <- createDataPartition(y = OJ$Purchase,
                               p = 0.7, list = FALSE)
training <- OJ[ inTrain,]
testing <- OJ[ -inTrain,]

#Model 1: Random Forest Model
control <- trainControl(method = 'repeatedcv',
                        number = 10,
                        repeats = 1,
                        search = 'grid',
                        classProbs = TRUE)

tunegrid <- expand.grid(.mtry = c(5,10))
modellist <- list()

# Add one more element to ntree and node size - tuning

for (ntree in c(20,50,150)){
  for (nodesize in c(3,5,7)){
    set.seed(123)
    fit <- train(make.names(Purchase)~.,
                 data = training,
                 method = 'rf',
                 metric = 'Accuracy',
                 tuneGrid = tunegrid,
                 trControl = control,
                 ntree = ntree,
                 nodesize = nodesize)
    key <- toString(ntree+nodesize)
    modellist[[key]] <- fit
  }
}

#Results for model
results <- resamples(modellist)
summary(results)
#ntree= 50; nodesize=5 best tuned 
fit$bestTune
#mtry=5 best tuned
#confusion Matrix for the training dataset
confusionMatrix(fit)
#Variable Importance
varImp(fit)


#Confusion Matrix 
forest_probabilities<-predict(fit,newdata=testing,type="prob") 
forest_classification<-rep("1",321)
forest_classification[forest_probabilities[,2]<0.5]="0" 
forest_classification<-as.factor(forest_classification)
 

#ROC Curve
forest_ROC_prediction <- prediction(forest_probabilities[,2], testing$Purchase) 
forest_ROC <- performance(forest_ROC_prediction,"tpr","fpr")
plot(forest_ROC) 

#AUC Calculation
AUC.tmp <- performance(forest_ROC_prediction,"auc")
forest_AUC <- as.numeric(AUC.tmp@y.values)
forest_AUC

#Model 2: XGBoost

OJ_matrix<-model.matrix(Purchase~., data = OJ)
dim(OJ_matrix)

x_train <- OJ_matrix[ inTrain,]
x_test <- OJ_matrix[ -inTrain,]

#splitting for label columns
training<-OJ[inTrain,]
testing<-OJ[-inTrain,]

y_train <-training$Purchase
y_test <-testing$Purchase
#hyper parameter tuning
xgb_grid = expand.grid(
  nrounds = c(100, 200, 300), 
  max_depth = c(2, 5, 10), 
  eta = c(0.0025, 0.05,0.1), 
  gamma = 0, 
  colsample_bytree = 1, 
  min_child_weight = 1,
  subsample = 1
)

# pack the training control parameters

xgb_trcontrol = trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 1,
  search = 'grid',
  classProbs = TRUE
)

#Building xgboost model
model_XGboost = train(
  x = x_train,
  y = make.names(y_train),
  trControl = xgb_trcontrol,
  tuneGrid = xgb_grid,
  method = "xgbTree",
  metric = "Accuracy",
)


#confusion Matrix - Training Dataset
confusionMatrix(model_XGboost)
#Model Results
model_XGboost$results
#Variable Importance Plot
varImp(model_XGboost)
#Best Tuned model out of the cross validation
model_XGboost$bestTune


#predicting on testing set
XGboost_prediction<-predict(model_XGboost,newdata=x_test, type="prob")
xgb_classification<-rep("1",321)
xgb_classification[XGboost_prediction[,2]<0.5]="0" 
xgb_classification<-as.factor(xgb_classification)


xgb_ROC_prediction <- prediction(XGboost_prediction[,2], y_test) 
xgb_ROC <- performance(xgb_ROC_prediction,"tpr","fpr")
plot(xgb_ROC) 

#AUC Calculation 
AUC.tmp <- performance(xgb_ROC_prediction,"auc")
xgb_AUC <- as.numeric(AUC.tmp@y.values)
xgb_AUC

#Model3: Logistic Model
model_logistic<-glm(Purchase ~ ., data=training, family="binomial"(link="logit"),control = list(maxit = 50))


summary(model_logistic) 

model_logistic_stepwiseAIC<-stepAIC(model_logistic,direction = c("both"),trace = 1)
summary(model_logistic_stepwiseAIC) 

logistic_probabilities<-predict(model_logistic_stepwiseAIC,newdata=testing,type="response")
logistic_classification<-rep("1",321)
logistic_classification[logistic_probabilities<0.5]="0" 
logistic_classification<-as.factor(logistic_classification)


####ROC Curve
logistic_ROC_prediction <- prediction(logistic_probabilities, testing$Purchase)
logistic_ROC <- performance(logistic_ROC_prediction,"tpr","fpr") 
plot(logistic_ROC) #Plot ROC curve

####AUC (area under curve)
auc.tmp <- performance(logistic_ROC_prediction,"auc")
logistic_auc_testing <- as.numeric(auc.tmp@y.values) 
logistic_auc_testing 
