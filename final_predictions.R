# loading libraries
.libPaths(c('C:/Users/Acer/Documents/R/win-library/4.1'))
library(caTools)
library(fastDummies)
library(tidyverse)
library(ROCR) # for ROC curv
library(randomForest)
library(ggplot2)
library(ggthemes)

# loading the dataset
df <- read.csv("dataset/credit_risk_dataset_training.csv",header=T)
head(df)

hist(df$loan_amnt)

table(df$loan_status)
# bar plot of default and paid
g <- ggplot(df, aes(loan_status)) +
  geom_bar(fill = "#4EB25A") +
  theme(axis.title.x=element_blank()) + 
  theme(axis.title.y=element_blank()) +
  scale_y_continuous(breaks=seq(0,18000,1000)) +
  scale_x_discrete(labels = c("Paid","Default")) +
  ggtitle("Count of Paid and Defaulted Loans") + 
  theme_minimal()

g
summary(df)

df$person_home_ownership <- as.factor(df$person_home_ownership)
df$loan_intent <- as.factor(df$loan_intent)
df$loan_grade <- as.factor(df$loan_grade)
df$cb_person_default_on_file <- as.factor(df$cb_person_default_on_file)
df$loan_status <- as.factor(df$loan_status)

summary(df)

boxplot(df$person_age)

plot(df$person_age, df$person_income, main="Age vs Income")

# drop rows where age is greater than 80
age_idx <- which(df$person_age > 80)
df[age_idx,]$person_age <- 80

summary(df)

# Imputing Missing values
# columns with missing values: person_emp_length, loan_int_rate

# impute using median of columns
emp_length_median <- median(df$person_emp_length,na.rm=T)
emp_length_median
df$person_emp_length[is.na(df$person_emp_length)] <- emp_length_median

int_rate_median <- median(df$loan_int_rate, na.rm=T)
int_rate_median
df$loan_int_rate[is.na(df$loan_int_rate)] <- int_rate_median

# Handling Outliers

# Use capping 

# person_income
qnt <- quantile(df$person_income,c(0.25,0.75),na.rm=T)
cap <- quantile(df$person_income,c(0.05,0.95,na.rm=T))
threshold <- 1.5 * IQR(df$person_income,na.rm=T)

# replace outliers using capping method
df$person_income[df$person_income > (qnt[[2]]+threshold)] <- cap[[2]]
summary(df$person_income)

# employment length
plot(df$person_age, df$person_emp_length, main="Age vs Employment length")

df$person_emp_length[1] <- median(df$person_emp_length)


# EDA

# histogram of loan amount
hist(df$loan_amnt, breaks = 50, xlab = "Loan Amount", 
     main = "Histogram of the Loan Amount",col = "red")

barplot(table(df$loan_status))

# histogram of interest rate
hist(df$loan_int_rate, breaks = 20, xlab = "Loan Interest Rate", 
     main = "Histogram of the Loan Interest rate",col = "red")

g1 <- df %>% filter(loan_status==1) %>% group_by(loan_grade) %>% summarise(default_counts=n())
g2 <- df %>% group_by(loan_grade) %>% summarise(count = n())
g3 <- g2 %>% left_join(g1) %>% mutate(default_rate = (default_counts/count * 100))

ggplot(g3, aes(x=loan_grade, y=default_rate, fill=loan_grade)) + geom_bar(stat="identity") + ggtitle("Default Rate (%) per Loan Grade") + theme_stata()

# relationship between loan grade vs interest rate

x1 <- df %>% filter(loan_status == 1) %>% group_by(loan_grade) %>% summarise(int_rate = mean(loan_int_rate))

ggplot(x1, aes(x=loan_grade, y=int_rate, fill=loan_grade)) + geom_bar(stat="identity",position="dodge") + ggtitle("Interest rates for different loan grades") + theme_minimal()

# default rate for each intent
g1 <- df %>% filter(loan_status == 1) %>% group_by(loan_intent) %>% summarise(default_counts = n())
g2 <- df %>% group_by(loan_intent) %>% summarise(count = n())
g3 <- g2 %>% left_join(g1) %>% mutate(default_rate = default_counts/count*100)
ggplot(g3.sort, aes(x=reorder(loan_intent,-default_rate), y=default_rate,fill=loan_intent)) + geom_bar(stat="identity") + 
  theme(axis.text.x = element_text(angle=45, vjust=1, hjust=1)) + 
  ggtitle("Default Rate per Loan Intent") +
  labs(x="Loan Intent", y="Default Rate (%)")
# create dummy variables

df <- dummy_cols(df,
                 select_columns = c('person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file'),
                 remove_first_dummy=TRUE,
                 remove_selected_columns=TRUE)
df <- df %>% select(-loan_status, everything())


# Data Splitting
set.seed(30) 
train <- sample.split(df$loan_status, SplitRatio = 0.75)

# get number of rows and columns for train set
train_dim <- dim(df[train,])
paste("Number of rows and columns for training set: ",train_dim)

# get number of rows and columns for test set
test_dim <- dim(df[!train,])
paste("Number of rows and columns for test set: ",test_dim)

# create a separate dataframe for training set
train_set <- df[train,]

# create a separate df for testing
test_set <- df[!train,]

# logistic regression with more than one predictors
glm.mult <- glm(loan_status~., data=train_set, family="binomial")
summary(glm.mult)

# predict on test set
preds <- predict(glm.mult, newdata = test_set, type="response")

pred.results <- ifelse(preds>0.5,1,0)
pred.results

pred_20.results <- ifelse(preds>0.25,1,0)
# model accuracy
mean(pred.results == test_set$loan_status)

mean(pred_20.results==test_set$loan_status)
confusionMatrix <- table(pred.results, test_set$loan_status)
confusionMatrix

sum(diag(confusionMatrix))/sum(confusionMatrix)

# ROC curve
pr <- prediction(pred.results,test_set$loan_status)
prf <- performance(pr, measure="tpr", x.measure="fpr")
plot(prf)

# get area under the ROC curve
auc <- performance(pr, measure='auc')
auc <- auc@y.values[[1]]
auc # 0.74

# Random Forest model
set.seed(333)
rfModel <- randomForest(loan_status~.,data=train_set)
rfFit <- predict(rfModel, test_set ,type="prob")[,2]
rfPred <- prediction(rfFit, test_set$loan_status)
plot(performance(rfPred,'tpr','fpr'))


performance(rfPred, measure = 'auc')@y.values[[1]] # 0.9279

par(mfrow=c(1,1))
varImpPlot(rfModel, pch=1, main="Random Forest Model Variables Importance")

rf_confusionMatrix <- table(test_set$loan_status,predict(rfModel, test_set, type="class"))
rf_accuracy <- sum(diag(rf_confusionMatrix))/sum(rf_confusionMatrix)

evaluation_df <- read.csv("dataset/credit_risk_dataset_test.csv", header=T)
eval_df_orig <- read.csv("dataset/credit_risk_dataset_test.csv", header=T)
head(evaluation_df)

# drop the target variable
evaluation_df <- evaluation_df %>% select(-loan_status)


evaluation_df$person_home_ownership <- as.factor(evaluation_df$person_home_ownership)
evaluation_df$loan_intent <- as.factor(evaluation_df$loan_intent)
evaluation_df$loan_grade <- as.factor(evaluation_df$loan_grade)
evaluation_df$cb_person_default_on_file <- as.factor(evaluation_df$cb_person_default_on_file)

summary(evaluation_df)

# handling NA values

# median impute for person_emp_length, loan_int_rate
emp_length_median <- median(evaluation_df$person_emp_length,na.rm=T)
emp_length_median
evaluation_df$person_emp_length[is.na(evaluation_df$person_emp_length)] <- emp_length_median

int_rate_median <- median(evaluation_df$loan_int_rate, na.rm=T)
int_rate_median
evaluation_df$loan_int_rate[is.na(evaluation_df$loan_int_rate)] <- int_rate_median

summary(evaluation_df)

# person_income
qnt <- quantile(evaluation_df$person_income,c(0.25,0.75),na.rm=T)
cap <- quantile(evaluation_df$person_income,c(0.05,0.95,na.rm=T))
threshold <- 1.5 * IQR(evaluation_df$person_income,na.rm=T)

# replace outliers using capping method
evaluation_df$person_income[evaluation_df$person_income > (qnt[[2]]+threshold)] <- cap[[2]]
summary(evaluation_df$person_income)

# make maximum age 80
max_ages_idx <- which(evaluation_df$person_age > 80)
evaluation_df[max_ages_idx,]$person_age <- 80

# create dummy variables
evaluation_df <- dummy_cols(evaluation_df,
           select_columns = c('person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file'),
           remove_first_dummy=TRUE,
           remove_selected_columns=TRUE)

eval_df_orig$loan_status <- predict(rfModel, evaluation_df, type="class")
write.csv(eval_df_orig,"./dataset/submission.csv",row.names=F)