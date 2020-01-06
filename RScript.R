#import libraries

library(ggplot2)
library(ggthemes)
library(ggmosaic)
library(readr)
library(tidyverse)
library(dplyr)
library(scales)
library(xgboost)
library(e1071)
library(tictoc)

train_identity <- read_csv("train_identity.csv")
train_transaction <- read_csv("train_transaction.csv")
test_identity <- read_csv("test_identity.csv")
test_transaction <- read_csv("test_transaction.csv")

dim(train_identity)
dim(train_transaction)
dim(test_identity)
dim(test_transaction)


#Join both transaction and identity tables 

colnames(test_identity)[2:39] <- c("id_01", "id_02","id_03","id_04", "id_05","id_06","id_07", "id_08", "id_09","id_10","id_11", "id_12","id_13","id_14", "id_15","id_16","id_17", "id_18","id_19","id_20", "id_21","id_22","id_23", "id_24","id_25","id_26", "id_27","id_28","id_29", "id_30","id_31","id_32", "id_33","id_34","id_35", "id_36","id_37","id_38")

train <- left_join(train_transaction,train_identity)
test <- left_join(test_transaction,test_identity)
rm(train_transaction, train_identity, test_transaction, test_identity)

## Graphs of relevant details 

#isFraud

target_df <- data.frame(table(factor(train$isFraud)))
colnames(target_df) <- c("isFraud", "Freq")

ggplot(target_df, aes(x = isFraud, y = Freq , fill = isFraud)) + geom_bar(stat = "identity")+ geom_text(aes(label=paste0(round(Freq/sum(Freq)*100, 3), "%")) , vjust = -.5) + ggtitle("Target Variable : isFraud") + labs(x = "isFraud") + theme_bw()

#The number of fradulant cases take up only around 3.5% of the total number of cases. 


#transactionDT

ggplot(train, aes(x= TransactionDT, fill = factor(isFraud))) + geom_histogram(alpha=0.7, bins=50) + scale_fill_tableau() + theme_bw()


#transactionAmt 

ggplot(train,aes(x= TransactionAmt, fill = factor(isFraud)))+geom_density()

skewness(train$TransactionAmt) #highly skewed, can consider to log()

ggplot(train,aes(x= log(TransactionAmt), fill = factor(isFraud)))+geom_density(alpha = 0.1)

#cards 1 to 6

ggplot(train, aes(card1, fill = factor(isFraud))) + geom_histogram(alpha = 0.7, bins = 50) + ggtitle("Train Card 1")

ggplot(train, aes(card2, fill = factor(isFraud))) + geom_histogram(alpha = 0.7, bins = 50) + ggtitle("Train Card 2")

ggplot(train, aes(card3, fill = factor(isFraud))) + geom_histogram(alpha = 0.7, bins = 50) + ggtitle("Train Card 3")

ggplot(train, aes(card4, fill = factor(isFraud))) + geom_bar(alpha = 0.7, bins = 50, stat = "count") + ggtitle("Train Card 4") + theme(axis.text = element_text(size = 6))
#users of 'discover' card are more likely to fraudulant 

ggplot(train) + geom_mosaic(aes(x=product(card4), fill=factor(isFraud))) + scale_fill_tableau() + theme_bw() + labs(fill='isFraud') + ggtitle("Mosaic Plot of Card 4") + coord_flip()

ggplot(train, aes(card5, fill = factor(isFraud))) + geom_histogram(alpha = 0.7, bins = 50) + ggtitle("Train Card 5")

ggplot(train, aes(factor(card6), fill = factor(isFraud))) + geom_bar(alpha = 0.7, bins = 50) + ggtitle("Train Card 6")

#users of credit card are more likely to fraudulant 


#email

ggplot(train,aes(x = factor(P_emaildomain),fill=isFraud))+geom_bar()+coord_flip()

#a large number of users gave gmail.com for their email for P_emaildomain

ggplot(train,aes(x = factor(R_emaildomain),fill=isFraud))+geom_bar()+coord_flip()
#returns a high percentage of NA, as there are lots of blanks under this variable 


## Data Preparation

#Variables with large missing data can act as noise in the model. Therefore, removing these variables may lead to a more accurate model. 


missing_train <- sort(colSums(is.na(train))[colSums(is.na(train)) > 0], decreasing=TRUE)

missing_test <- sort(colSums(is.na(test))[colSums(is.na(test)) > 0], decreasing=TRUE)

## Ratio of missing variables

# Ratio of missing values

missing_train_pct <- round(missing_train/nrow(train), 2)
missing_test_pct <- round(missing_test/nrow(test), 2)

# drop variable with more than 50% missing values
drop_col_train <- names(missing_train_pct[missing_train_pct > 0.5])

drop_col_test <- names(missing_test_pct[missing_test_pct > 0.5])

all(drop_col_test %in% drop_col_train) #TRUE

drop_col <- intersect(drop_col_test,drop_col_train)

train[,drop_col] <- NULL
test[,drop_col] <- NULL



## Modelling 
# I have chosen to use XGBoost as it has high predictive power after tuning. 

x_train <-  train %>% select(-isFraud, -TransactionID)
y_train <- train$isFraud
x_test <- test %>% select(-TransactionID)


#converting catergorical variables to numerical 
cat_vars <- names(x_train)[sapply(x_train, is.character)]

x_train[, cat_vars] <- lapply(x_train[, cat_vars], as.factor)
x_test[, cat_vars] <- lapply(x_test[, cat_vars], as.factor)

# Label encoding
x_train[, cat_vars] <- lapply(x_train[, cat_vars], as.integer)
x_test[, cat_vars] <- lapply(x_test[, cat_vars], as.integer)


#xgboost model
dtrain <- xgb.DMatrix(data=as.matrix(x_train), label=as.matrix(y_train))

#parameter tuning 

objective <- "binary:logistic"
cv.fold <- 10

# parameter ranges
max_depths <- c(4, 6, 8)  # candidates for d
etas <- c(0.05, 0.03, 0.01)  # candidates for lambda
subsamples <- c(0.5, 0.75, 1)
colsamples <- c(0.6, 0.8, 1)


set.seed(420)
tune.out <- data.frame()
for (max_depth in max_depths) {
  for (eta in etas) {
    for (subsample in subsamples) {
      for (colsample in colsamples) {
        # **calculate max n.trees by my secret formula**
        n.max <- round(100 / (eta * sqrt(max_depth)))
        xgb.cv.fit <- xgb.cv(data = dtrain, objective=objective, nfold=cv.fold, early_stopping_rounds=100, verbose=0,
                             nrounds=n.max, max_depth=max_depth, eta=eta, subsample=subsample, colsample_bytree=colsample)
        n.best <- xgb.cv.fit$best_ntreelimit
        if (objective == "reg:linear") {
          cv.err <- xgb.cv.fit$evaluation_log$test_rmse_mean[n.best]
        } else if (objective == "binary:logistic") {
          cv.err <- xgb.cv.fit$evaluation_log$test_error_mean[n.best]
        }
        out <- data.frame(max_depth=max_depth, eta=eta, subsample=subsample, colsample=colsample, n.max=n.max, nrounds=n.best, cv.err=cv.err)
        print(out)
        tune.out <- rbind(tune.out, out)
      }
    }
  }
}

tune.out

opt <- which.min(tune.out$cv.err)
max_depth.opt <- tune.out$max_depth[opt]
eta.opt <- tune.out$eta[opt]
subsample.opt <- tune.out$subsample[opt]
colsample.opt <- tune.out$colsample[opt]
nrounds.opt <- tune.out$nrounds[opt]

set.seed(420)

xgb.best <- xgboost(data=dtrain, objective="binary:logistic", nround=nrounds.opt, max.depth=max_depth.opt, eta=eta.opt, subsample=subsample.opt, colsample_bytree=colsample.opt,min_child_weight = min_child_weight.opt)

importance_matrix <- xgb.importance(model = xgb.best, feature_names = colnames(x.train))
xgb.plot.importance(importance_matrix=importance_matrix, top_n = 15)

set.seed(420)

pred <- predict(xgb.best, as.matrix(x_test))

submission <- read_csv("sample_submission.csv")
submission$isFraud <- pred
write.csv(submission, file = "submission.csv", row.names = F)
