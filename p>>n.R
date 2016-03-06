```{r}
#read the data
library(readr)
base_path <- getwd()
data_alix <- read_csv(paste0(base_path, "/data.csv"))


#separate train data with test.

training <- subset(data_alix, data_alix$train=="1")
sub<-as.numeric(rownames(training)) # because rownames() returns character
test <- data_alix[-sub,]

#remove 'train' column, we don't need it anymore.
training$train <- NULL
training$id <- NULL
#for test,
test$train <- NULL
testid <- test$id

test$target_eval <- NULL
test$id <- NULL

#load the library
library(glmnet)
set.seed(123)

#convert to factor/categorical
target <- as.factor(training$target_eval)

#applying elastic net regularization model to include both type of penalties(lasso and ridge regression).

VarSelection <- cv.glmnet(x = as.matrix(training[,c(2:301)]),
                                   y = target,
                                   family = "binomial",
                                   alpha = 0.8, #elastic net to include both l1 and l2 penalty for regularization.
                                   #dfmax = 150, 
                                   nfolds = 5,
                                   type.logistic = "modified.Newton",
                                   type.measure="auc")

#plotting auc, it gives fair result.
plot(VarSelection)


#Now the main part.

################Variable Selection##################

#we can select lambda.min or lambda.1se for it, lambda.min provides the coefficients for best model,
#which could be little complex as well as lead to over fitting sometimes, while lambda.1se provides coefficient for 
# a simpler model from best model, with added little uncertainity.


### I decided to used both lambda techniques to select the final set of features.


### First using lambda.1se parameter.
c1 <- coef.cv.glmnet(VarSelection,s="lambda.1se",exact=T)

#only those coefficient values that are non-zero.
inds <- which(c1!=0)
variables <-  row.names(c1)[inds]
`%ni%` <- Negate(`%in%`)

#final variables.
variables1 <- variables[variables %ni% '(Intercept)']
variables1


###now using lamba.min for another variable set.
c2 <- coef.cv.glmnet(VarSelection,s="lambda.min",exact=T)

#only those coefficient values that are non-zero.
inds <- which(c2!=0)
variables <-  row.names(c2)[inds]
`%ni%` <- Negate(`%in%`)

#final variables.
variables2 <- variables[variables %ni% '(Intercept)']
variables2


## finally combining both variable sets and getting unique features from them.

#combining both the sets of variables.
p <- c(variables1, variables2)
p<- unique(p)


#now using the above obtained list of variables to subset the training set predictors to smaller number but retained predicting power. 
newtrain <- training[,p]
newtrain <- as.data.frame(cbind(target, newtrain))
newtrain$target <- as.factor(newtrain$target)
```                        
                          
                          
```{r}
library(caret)
newtrain$target <- as.factor(ifelse(newtrain$target == 1, "X1", "X0"))
FL <- as.formula(paste("target ~ ", paste(p, collapse = "+")))
MyTrainControl = trainControl(method = "repeatedCV", number = 10, repeats = 50, 
    returnResamp = "all", classProbs = TRUE, summaryFunction = twoClassSummary)
model <- train(FL, data = newtrain, method = "glmnet", metric = "ROC", 
               tuneGrid = expand.grid(.alpha = c(0, 1), .lambda = seq(0, 0.25, by = 0.005)), 
               trControl = MyTrainControl)
plot(model, metric = "ROC")


##### making predictions #####

pred <- predict(model, newdata = test, type = "prob") #probabilities of prediction.
predresponse <- predict(model, newdata = test, type = "raw") #response classes.


predictions1 <- as.data.frame(pred)
predictions2 <- as.data.frame(predresponse)
testID <- as.data.frame(testid)


####### first submitting the response class file ######
submit_response = as.data.frame(c(testID, predictions2))
colnames(submit_response) <- c("Id", "target_eval")
submit_response$target_eval <- as.factor(ifelse(submit_response$target_eval == "X1", "1", "0"))
write.csv(submit_response, "problem2_responsefile.csv", row.names = F)


####### now submitting the predicted probability file ######
probresponse <- pmax(predictions1[,1], predictions1[,2], na.rm = TRUE) 
probresponse <- as.data.frame(probresponse)
submit_prob <- as.data.frame(c(testID,probresponse))
colnames(submit_prob) <- c("Id", "target_prob")
write.csv(submit_prob, "problem2_probabilityfile.csv", row.names = F)
```
