```{r}
#read the data
library(readr)
base_path <- getwd()
data_alix <- read_csv(paste0(base_path, "/data.csv"))

#remove id
data_alix$id <- NULL

#separate train data with test.

training <- subset(data_alix, data_alix$train=="1")
sub<-as.numeric(rownames(training)) # because rownames() returns character
test <- data_alix[-sub,]

#remove 'train' column, we don't need it anymore.
training$train <- NULL

#for test,
test$train <- NULL
test$target_eval <- NULL


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
plot(glmFitForVarSelection)


#Now the main part.

################Variable Selection##################

#we can select lambda.min or lambda.1se for it, lambda.min provides the coefficients for best model,
#which could be little complex as well as lead to over fitting sometimes, while lambda.1se provides coefficient for 
# a simpler model from best model, with added little uncertainity, which serves our purpose.


### I used both lambda techniques to check for comparison, and ultimately chose to settle with lambda.1se

c <- coef.cv.glmnet(VarSelection,s="lambda.1se",exact=T)

#only those coefficient values that are non-zero.
inds <- which(c!=0)
variables <-  row.names(c)[inds]
`%ni%` <- Negate(`%in%`)

#final variables.
variables <- variables[variables %ni% '(Intercept)']
variables

#now using the above obtained list of variables to subset the training set predictors to smaller number but reatined predicting power. 
newtrain <- training[,variables]
newtrain <- as.data.frame(cbind(target, newtrain))
newtrain$target <- as.factor(newtrain$target)


library(h2o)
h2o.init()

training$target_eval <- as.factor(training$target_eval)
train.hex <- as.h2o(training)
test.hex <- as.h2o(test)
model <- h2o.deeplearning(x= #2:301, y=1, training_frame = train.hex,
                          buildModel 'deeplearning', 
                          {"model_id":"deeplearning-8de7978b-22ed-4ff0-9a65-45d20beff737",
                            "training_frame":"frame_0.750","validation_frame":"frame_0.250",
                            "nfolds":"5","response_column":"target","ignored_columns":[],
                            "ignore_const_cols":true,"activation":"TanhWithDropout",
                            "hidden":[100,100],"epochs":"18","variable_importances":true,
                            "fold_assignment":"Modulo","balance_classes":false,"checkpoint":"",
                            "use_all_factor_levels":true,"train_samples_per_iteration":-2,
                            "adaptive_rate":true,"input_dropout_ratio":"0.2","hidden_dropout_ratios":[0.4,0.8],
                            "l1":0,"l2":0,"loss":"Automatic","distribution":"AUTO","score_interval":5,
                            "score_training_samples":"0","score_validation_samples":0,"score_duty_cycle":"0.001",
                            "stopping_rounds":5,"stopping_metric":"AUC","stopping_tolerance":0,"autoencoder":false,
                            "keep_cross_validation_predictions":false,"target_ratio_comm_to_comp":0.05,
                            "seed":3795404593086760400,"rho":0.99,"epsilon":1e-8,"max_w2":"Infinity",
                            "initial_weight_distribution":"UniformAdaptive","classification_stop":0,
                            "score_validation_sampling":"Uniform","diagnostics":true,"fast_mode":true,
                            "force_load_balance":true,"single_node_mode":true,"shuffle_training_data":false,
                            "missing_values_handling":"MeanImputation","quiet_mode":false,"sparse":false,
                            "col_major":false,"average_activation":0,"sparsity_beta":0,
                            "max_categorical_features":2147483647,"reproducible":false,
                            "export_weights_and_biases":false,"elastic_averaging":false}
  
```                        
                          
