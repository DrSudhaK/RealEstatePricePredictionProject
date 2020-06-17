#----------------------------------------------------------------------------------------
## Sudha Kankipati
## Real estate price prediction Project 
## HarvardX: PH125.9x Data Science: Capstone 
## Running on  Windows 10, RAM 32GB 
#----------------------------------------------------------------------------------------
#Introduction
#----------------------------------------------------------------------------------------
#Real estate price prediction: it solves the problem of predicting house prices 
#for house buyers and house sellers.

#A house value is more than location and square footage. An educated party would want to 
#know all aspects that give a house its value. Like age of the house, distance to the
#nearest MRT station, number of near by convenience stores, its location

#We will be applying machine learning techniques that go beyond standard linear regression.

#----------------------------------------------------------------------------------------
#Data Preparation
#----------------------------------------------------------------------------------------
###Install Packages
if(!require(caret))install.packages("caret", repos ="http://cran.us.r-project.org")
if(!require(data.table))install.packages("data.table", repos ="http://cran.us.r-project.org")
if(!require(rattle))install.packages("rattle", repos ="http://cran.us.r-project.org")
if(!require(magrittr))install.packages("magrittr", repos ="http://cran.us.r-project.org")
if(!require(Hmisc))install.packages("Hmisc", repos ="http://cran.us.r-project.org")
library(Hmisc, quietly=TRUE)
if(!require(reshape))install.packages("reshape", repos ="http://cran.us.r-project.org")
if(!require(arules))install.packages("arules", repos ="http://cran.us.r-project.org")
if(!require(rpart))install.packages("rpart", repos ="http://cran.us.r-project.org")
if(!require(dataMaid)) install.packages("dataMaid", repos = "http://cran.us.r-project.org")
library(dataMaid)

building <- TRUE
scoring  <- ! building

# A pre-defined value is used to reset the random seed 
# so that results are repeatable.
crv$seed <- 42 

#=======================================================================
# Load a dataset from file.
# dataset is present at https://www.kaggle.com/quantbruce/real-estate-price-prediction/datasets_88705_204267_Real estate.csv

fname         <- "file:///C:/DataScience/datasets_88705_204267_Real estate.csv" 

#Creating Dataset from file
crs$dataset <- read.csv(fname,
                        na.strings=c(".", "NA", "", "?"),
                        strip.white=TRUE, encoding="UTF-8")

#=======================================================================
# Build the train/validate/test datasets.
# nobs=414 train=290 validate=62 test=62

set.seed(crv$seed)

#Number of observations
crs$nobs <- nrow(crs$dataset)
crs$nobs

#Creating training set
crs$train <- sample(crs$nobs, 0.7*crs$nobs)

#Creating validation set
crs$nobs %>%
  seq_len() %>%
  setdiff(crs$train) %>%
  sample(0.15*crs$nobs) ->
  crs$validate

#Creating testing set
crs$nobs %>%
  seq_len() %>%
  setdiff(crs$train) %>%
  setdiff(crs$validate) ->
  crs$test

#----------------------------------------------------------------------------------------
##  Data Cleaning
#----------------------------------------------------------------------------------------
#The following variable selections have been noted.
# We ignore transaction date
# we use No as identity
crs$input     <- c("X2.house.age",
                   "X3.distance.to.the.nearest.MRT.station",
                   "X4.number.of.convenience.stores", "X5.latitude",
                   "X6.longitude")

crs$numeric   <- c("X2.house.age",
                   "X3.distance.to.the.nearest.MRT.station",
                   "X4.number.of.convenience.stores", "X5.latitude",
                   "X6.longitude")

crs$categoric <- NULL

crs$target    <- "Y.house.price.of.unit.area"
crs$risk      <- NULL
crs$ident     <- "No"
crs$ignore    <- "X1.transaction.date"
crs$weights   <- NULL


#=======================================================================
## Data Exploration
# The 'Hmisc' package provides the 'contents ,describe' function.
#Summary of the dataset.
contents(crs$dataset[crs$train, c(crs$input, crs$risk, crs$target)])
summary(crs$dataset[crs$train, c(crs$input, crs$risk, crs$target)])

#Generating a description of the dataset.
describe(crs$dataset[crs$train, c(crs$input, crs$risk, crs$target)])

#=======================================================================
## Data Visualization
# Displaying histogram plots for the selected variables. 
# Generating histogram plot for X2.house.age
p01 <- crs %>%
  with(dataset[train,]) %>%
  dplyr::select(X2.house.age) %>%
  ggplot2::ggplot(ggplot2::aes(x=X2.house.age)) +
  ggplot2::geom_density(lty=3) +
  ggplot2::xlab("X2.house.age\n\nRattle 2020-Jun-15 19:47:14 drsud") +
  ggplot2::ggtitle("Distribution of X2.house.age (sample)") +
  ggplot2::labs(y="Density")

#Generating histogram plot for X3.distance.to.the.nearest.MRT.station
p02 <- crs %>%
  with(dataset[train,]) %>%
  dplyr::select(X3.distance.to.the.nearest.MRT.station) %>%
  ggplot2::ggplot(ggplot2::aes(x=X3.distance.to.the.nearest.MRT.station)) +
  ggplot2::geom_density(lty=3) +
  ggplot2::xlab("X3.distance.to.the.nearest.MRT.station\n\nRattle 2020-Jun-15 19:47:14 drsud") +
  ggplot2::ggtitle("Distribution of X3.distance.to.the.nearest.MRT.station (sample)") +
  ggplot2::labs(y="Density")

#Generating histogram plot for X4.number.of.convenience.stores
p03 <- crs %>%
  with(dataset[train,]) %>%
  dplyr::select(X4.number.of.convenience.stores) %>%
  ggplot2::ggplot(ggplot2::aes(x=X4.number.of.convenience.stores)) +
  ggplot2::geom_density(lty=3) +
  ggplot2::xlab("X4.number.of.convenience.stores\n\nRattle 2020-Jun-15 19:47:14 drsud") +
  ggplot2::ggtitle("Distribution of X4.number.of.convenience.stores (sample)") +
  ggplot2::labs(y="Density")

#Displaying the plots.
gridExtra::grid.arrange(p01, p02, p03)

#=======================================================================
# KMeans 
# Reseting the random number seed to obtain the same results each time.

set.seed(crv$seed)

#Generating a kmeans cluster of size 10.
crs$kmeans <- kmeans(sapply(na.omit(crs$dataset[crs$train, crs$numeric]), rescaler, "range"), 10)

#=======================================================================
# Report on the cluster characteristics. 
# Cluster sizes:
paste(crs$kmeans$size, collapse=' ')

# Data means:
colMeans(sapply(na.omit(crs$dataset[crs$train, crs$numeric]), rescaler, "range"))

# Cluster centers:
crs$kmeans$centers

# Within cluster sum of squares:
crs$kmeans$withinss

#=======================================================================
# Hierarchical Cluster 
# Generating a hierarchical cluster from the numeric data.

crs$dataset[crs$train, crs$numeric] %>%
  amap::hclusterpar(method="euclidean", link="ward", nbproc=1) ->
  crs$hclust

#=======================================================================
#Association Rules 
# The 'arules' package provides the 'arules' function.
#Generating a transactions dataset.
crs$transactions <- as(split(crs$dataset[crs$train, crs$target],
                             crs$dataset[crs$train, crs$ident]),
                       "transactions")

#Generating the association rules.
crs$apriori <- apriori(crs$transactions, parameter = list(support=0.100, confidence=0.100, minlen=2))

#Summary the resulting rule set.
generateAprioriSummary(crs$apriori)

#Building Models
#=======================================================================
## Decision Tree 
# The 'rpart' package provides the 'rpart' function.
#Reseting the random number seed to obtain the same results each time.
set.seed(crv$seed)

#Building the Decision Tree model.
crs$rpart <- rpart(Y.house.price.of.unit.area ~ .,
                   data=crs$dataset[crs$train, c(crs$input, crs$target)],
                   method="anova",
                   parms=list(split="information"),
                   control=rpart.control(usesurrogate=0, 
                                         maxsurrogate=0),
                   model=TRUE)

#Generating a textual view of the Decision Tree model.
print(crs$rpart)
printcp(crs$rpart)
cat("\n")

#=======================================================================
##Building a Random Forest model using the traditional approach.

set.seed(crv$seed)

crs$rf <- randomForest::randomForest(Y.house.price.of.unit.area ~ .,
                                     data=crs$dataset[crs$train, c(crs$input, crs$target)], 
                                     ntree=500,
                                     mtry=2,
                                     importance=TRUE,
                                     na.action=randomForest::na.roughfix,
                                     replace=FALSE)

#Generating textual output of the 'Random Forest' model.
crs$rf

# Listing the importance of the variables.
rn <- crs$rf %>%
  randomForest::importance() %>%
  round(2)
rn[order(rn[,1], decreasing=TRUE),]

#=======================================================================
# Regression model 
# Building a Regression model.

crs$glm <- lm(Y.house.price.of.unit.area ~ ., data=crs$dataset[crs$train,c(crs$input, crs$target)])

# Generating a textual view of the Linear model.
print(summary(crs$glm))
cat('==== ANOVA ====')
print(anova(crs$glm))

#Plot the model evaluation.
ttl <- genPlotTitleCmd("Linear Model",crs$dataname,vector=TRUE)
plot(crs$glm, main=ttl[1])

#=======================================================================
## Regression model 
# Building a Regression model.

crs$glm <- glm(Y.house.price.of.unit.area ~ ., data=crs$dataset[crs$train,c(crs$input, crs$target)], family=gaussian(identity))

# Generate a textual view of the Linear model.
print(summary(crs$glm))
cat('==== ANOVA ====')
print(anova(crs$glm))


# Evaluate model performance on the validation dataset. 
# RPART: Generate a Predicted v Observed plot for rpart model on datasets_88705_204267_Real estate.csv [validate].
crs$pr <- predict(crs$rpart, newdata=crs$dataset[crs$validate, c(crs$input, crs$target)])

# Obtain the observed output for the dataset.
obs <- subset(crs$dataset[crs$validate, c(crs$input, crs$target)], select=crs$target)

# Handle in case categoric target treated as numeric.
obs.rownames <- rownames(obs)
obs <- as.numeric(obs[[1]])
obs <- data.frame(Y.house.price.of.unit.area=obs)
rownames(obs) <- obs.rownames

# Combine the observed values with the predicted.
fitpoints <- na.omit(cbind(obs, Predicted=crs$pr))

# Obtain the pseudo R2 - a correlation.
fitcorr <- format(cor(fitpoints[,1], fitpoints[,2])^2, digits=4)
dtRsqr <- fitcorr

# Plot settings for the true points and best fit.
op <- par(c(lty="solid", col="green"))

# Display the observed (X) versus predicted (Y) points.
plot(fitpoints[[1]], fitpoints[[2]], asp=1, xlab="Y.house.price.of.unit.area", ylab="Predicted")

# Generate a simple linear fit between predicted and observed.
prline <- lm(fitpoints[,2] ~ fitpoints[,1])

# Add the linear fit to the plot.
abline(prline)

# Add a diagonal representing perfect correlation.
par(c(lty="dashed", col="brown"))
abline(0, 1)

# Include a pseudo R-square on the plot
legend("bottomright",  sprintf(" Pseudo R-square=%s ", fitcorr),  bty="n")

# Add a title and grid to the plot.
title(main="Predicted vs. Observed Decision Tree Model
 datasets_88705_204267_Real estate.csv [validate]")
grid()

# RF: Generate a Predicted v Observed plot for rf model on datasets_88705_204267_Real estate.csv [validate].
crs$pr <- predict(crs$rf, newdata=na.omit(crs$dataset[crs$validate, c(crs$input, crs$target)]))

# Obtain the observed output for the dataset.
obs <- subset(na.omit(crs$dataset[crs$validate, c(crs$input, crs$target)]), select=crs$target)

# Handle in case categoric target treated as numeric.
obs.rownames <- rownames(obs)
obs <- as.numeric(obs[[1]])
obs <- data.frame(Y.house.price.of.unit.area=obs)
rownames(obs) <- obs.rownames

# Combine the observed values with the predicted.
fitpoints <- na.omit(cbind(obs, Predicted=crs$pr))

# Obtain the pseudo R2 - a correlation.
fitcorr <- format(cor(fitpoints[,1], fitpoints[,2])^2, digits=4)
rfRsqr <- fitcorr

# Plot settings for the true points and best fit.
op <- par(c(lty="solid", col="green"))

# Display the observed (X) versus predicted (Y) points.
plot(fitpoints[[1]], fitpoints[[2]], asp=1, xlab="Y.house.price.of.unit.area", ylab="Predicted")

# Generate a simple linear fit between predicted and observed.
prline <- lm(fitpoints[,2] ~ fitpoints[,1])

# Add the linear fit to the plot.
abline(prline)

# Add a diagonal representing perfect correlation.
par(c(lty="dashed", col="brown"))
abline(0, 1)

# Include a pseudo R-square on the plot
legend("bottomright",  sprintf(" Pseudo R-square=%s ", fitcorr),  bty="n")

# Add a title and grid to the plot.
title(main="Predicted vs. Observed  Random Forest Model
 datasets_88705_204267_Real estate.csv [validate]")
grid()

# GLM: Generate a Predicted v Observed plot for glm model on datasets_88705_204267_Real estate.csv [validate].
crs$pr <- predict(crs$glm, 
                  type    = "response",
                  newdata = crs$dataset[crs$validate, c(crs$input, crs$target)])

# Obtain the observed output for the dataset.
obs <- subset(crs$dataset[crs$validate, c(crs$input, crs$target)], select=crs$target)

# Handle in case categoric target treated as numeric.
obs.rownames <- rownames(obs)
obs <- as.numeric(obs[[1]])
obs <- data.frame(Y.house.price.of.unit.area=obs)
rownames(obs) <- obs.rownames

# Combine the observed values with the predicted.
fitpoints <- na.omit(cbind(obs, Predicted=crs$pr))

# Obtain the pseudo R2 - a correlation.
fitcorr <- format(cor(fitpoints[,1], fitpoints[,2])^2, digits=4)
lmRsqr <- fitcorr

# Plot settings for the true points and best fit.
op <- par(c(lty="solid", col="green"))

# Display the observed (X) versus predicted (Y) points.
plot(fitpoints[[1]], fitpoints[[2]], asp=1, xlab="Y.house.price.of.unit.area", ylab="Predicted")

# Generate a simple linear fit between predicted and observed.
prline <- lm(fitpoints[,2] ~ fitpoints[,1])

# Add the linear fit to the plot.
abline(prline)

# Add a diagonal representing perfect correlation.
par(c(lty="dashed", col="brown"))
abline(0, 1)

# Include a pseudo R-square on the plot
legend("bottomright",  sprintf(" Pseudo R-square=%s ", fitcorr),  bty="n")

# Add a title and grid to the plot.
title(main="Predicted vs. Observed Linear Model
 datasets_88705_204267_Real estate.csv [validate]")
grid()

# Conclusion 
# Based on the pseudo R-square results of the 3 models used
lmRsqr # Linear Model 
rfRsqr # Random Forest Model 
dtRsqr #Decision Tree Model

results <- c(lmRsqr,rfRsqr,dtRsqr)
minRsqr <- min(results)
if (minRsqr == lmRsqr) {
  a <-  paste("Linear Model has least pseudo R-square of ", lmRsqr, sep = "")
  a <- paste(a, " over the other 2 models", sep = "")
  a
}
if (minRsqr == rfRsqr) {
  a <-  paste("Random Forest Model has least pseudo R-square of ", rfRsqr, sep = "")
  a <- paste(a, " over the other 2 models", sep = "")
  a
}
if (minRsqr == dtRsqr) {
  a <-  paste("Decision Tree Model has least pseudo R-square of ", dtRsqr, sep = "")
  a <- paste(a, " over the other 2 models", sep = "")
  a
}
maxRsqr <- max(results)
if (maxRsqr == lmRsqr) {
  a <-  paste("Linear Model has higher pseudo R-square of ", lmRsqr, sep = "")
  a <- paste(a, " over the other 2 models", sep = "")
  a
}
if (maxRsqr == rfRsqr) {
  a <-  paste("Random Forest Model has higher pseudo R-square of ", rfRsqr, sep = "")
  a <- paste(a, " over the other 2 models", sep = "")
  a
}
if (maxRsqr == dtRsqr) {
  a <-  paste("Decision Tree Model has higher pseudo R-square of ", dtRsqr, sep = "")
  a <- paste(a, " over the other 2 models", sep = "")
  a
}

# Report creation date: Wed Jun 17 2020 
# R version 4.0.0 (2020-04-24).
# Platform: x86_64-w64-mingw32/x64 (64-bit)(Windows 10 x64 (build 18363)).
# Placed files for this project in https://github.com/DrSudhaK/RealEstatePricePredictionProject.git
# Dataset is located at 
# https://www.kaggle.com/quantbruce/real-estate-price-prediction?select=Real+estate.csv
# click on the download button "datasets_88705_204267_Real estate.csv" will be downloaded.

