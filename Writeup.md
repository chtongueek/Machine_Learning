---
title: "Machine Learning Writeup"
author: "Andrew"
date: "Thursday, June 18, 2015"
output: html_document
---

## Data Processing
The training data are completely free from NAs. This is great; however, the test data are not. In fact, there are several columns in the test data for which there are no data at all. Even though these variables might be useful predictors for the training data, the choices made using those variables can't be used for the test cases. Therefore any variables for which no data exist in the test set are elimiated in the training set.

The variables "X", "user_name", "raw_timestamp_part_1", and "raw_timestep_part_2" will not be useful predictors and are discarded from both sets.

```r
# Read in training and test data
train = read.csv("~/Data Science/coursera/ML/pml-training.csv", na.strings = "\"\"")
test = read.csv("~/Data Science/coursera/ML/pml-testing.csv", na.strings = "\"\"")

# The first 4 columns are variables which are not useful predictors. They are removed.
train = train[,-(1:4)]

# Remove any variables from the test set which aren't in the training set
test = test[, -which(!(names(test) %in% names(train)))]

# Get indicies for which variables have no data in the test set
ISNA = function(x){all(is.na(x)) | all(x == "NA")}
remove = which(sapply(test, ISNA))
names = c(names(test[,remove]))

# Remove those variables from both training and test sets
train = train[, -remove]
test = test[,-remove]

# Two variables have different types between the training and test sets. They are set to be the same
test$cvtd_timestamp = as.numeric(test$cvtd_timestamp)
test$new_window = as.numeric(test$new_window)
```
Since the data set is so large, only a portion of the data set is used to fit the model initially. If the model seems to predict well with cross-validation on only a portion of the data, then there's no need to consume more time simply to supply the model with more data.

A specified number of samples are selected randomly from the data set. These samples are used to build the model.

```r
size = 5000
ind = sample(1:dim(train)[1], size)
trn = train[ind,]
```
## Model
This model uses a backpropagating neural network to predict the outcomes. Because neural networks predict based on calculations, it is necessary to get the data into completely numerical form. So all factors are converted to numerics.

```r
# Change factors in training set to numerics for neural network
facts = which(sapply(train, class) == "factor")
for (i in 1:length(facts)){
  train[,facts[i]] = as.numeric(train[,facts[i]])
}

# Change factors in test set to numerics for neural network
facts = which(sapply(test, class) == "factor")
for (i in 1:length(facts)){
  test[,facts[i]] = as.numeric(test[,facts[i]])
}
```
Neural networks are also very sensitive to data which are of different magnitudes. If one variable is in the range 0 - 1, and another is in the range of 1000-2000, the second variable will dominate the calculation, even if the first is a better predictor. To mitigate this, the data are scaled.

```r
scaled = scale(train1, center = T, scale = T)
scaled2 = scale(test, T, scale = T)
```
The output from a neural network is a value between 0 and 1. This means that the output variable must be transformed into a form with only 0s and 1s. This is accomplished by using a vector to represent each outcome possibility. For example A --> 1 0 0 0 0; B --> 0 1 0 0 0, and so on.

```r
out2 = matrix(0, length(train$classe), length(unique(train$class)))
for (i in 1:length(train$classe)){
  out2[i, train$class[i]] = 1
}
```
## Cross Validation
In order to test the model's inherent accuracy, the selected sample is futher split into a training set and a validation set. The training set is 75% of the sample. The model is built using the training data, and then the validation data is run through the model and the predicted outputs are compared to the actual outputs. From this, the number of correct predictions divided by the total number of predicitons is calculated to give a percentage. For this particular setup, with sample size being 5000 split 75/25 training and validation, the in-sample accuracy was 99.7% and the out-of-sample prediction accuracy was 95%

```r
# Do a 75/25 split on these 5000 observations for training and validation
part = sample(1:size, round(.75*size, 0))
trn2 = trn[part,]
tst = trn[-part,]
tstout = tst$classe
tst = tst[,-which(names(tst) == "classe")]
```
## Running the model
Neural networks generally require several test run-throughs to tweak some of the imput parameters to useful values. For this data, a learning rate of .3 with 1000 backpropagation iterations seemed to get it to settle well. There are as many input neurons as input variables, and as many output neurons as unique output values. It was decided to do one hidden layer with twice as many hidden neurons as input neurons. A bias neuron was included, and the data were precessed in batches of 10 oberservations.

```r
# set numbers of neurons
IN = dim(trn)[2]
hide = 2*IN
OUT = dim(unique(outcome))[1]

BIAS = 1
LR = .03
steps = 1000
BS = 10

source("./neural.R")

listy = neural(IN, hide, OUT, trn, trnout, LR, steps, BIAS, scaled2, BS, tst)
```
## Processing Output
Since the resulting predictions are a vectors of numbers between 0 and 1, a bit of post-processing needs to take place before the predictions can be compared. This is accomplished by getting the placement of the max value for each vector. This will correspond to a number 1-5, which can then easily be transformed to A-E. The validation predictions are compared to the validation correct outcomes to get the expected out-of-sample error.

```r
coerce = function(x){
which(x == max(x))[1]
}
output = data.frame(t(listy[[1]]))
valout = data.frame(t(listy[[3]]))
prediction = data.frame(t(listy[[2]]))

outclass = sapply(output, coerce)
valclass = sapply(valout, coerce)
predictclass = sapply(prediction, coerce)

comp1 = (train$classe[ind])[part]
comp2 = (train$classe[ind])[-part]

insamp = sum(outclass == comp1)/length(outclass)
percV = sum(valclass == comp2)/length(valclass)
}
```
