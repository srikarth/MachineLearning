
# Loading appropiate libraries
library(caret)
library(kernlab)
library(AppliedPredictiveModeling)
library(caret)
library(rattle)
library(rpart.plot)
library(randomForest)

# Load datasets
training <- read.csv("/Users/Srinivas/Desktop/pml-training.csv", header=TRUE, sep=",",
                     stringsAsFactors=FALSE)
testing  <- read.csv("/Users/Srinivas/Desktop/pml-training.csv", header=TRUE, sep=",",
                     stringsAsFactors=FALSE)

#Summary of Data
str(training)
summary(training)


table(training$classe)
table(testing$classe)
summary(training$classe)
str(training$classe)

#Histogran and Feature Plot
hist(as.numeric(as.factor(training$classe)))
boxplot(as.numeric(as.factor(training$classe)))
featurePlot(x=training[,c("user_name","new_window","num_window", "X")],
            y = training$classe,
            plot="pairs")

names(training)

# Print class of all variables in dataset
sapply(training[1,], class)
classes1 <- sapply(training[1,], class)
table(classes1)
classes2 <- sapply(testing[1,], class)
table(classes2)

# Change some classes
training$classe <- as.factor(training$classe)
training$user_name<- as.factor(training$user_name)
training$new_window <- factor(training$new_window, labels=c("no", "yes"), 
                                  levels=c("no", "yes"))

testing$user_name <- as.factor(testing$user_name)
testing$new_window <- factor(testing$new_window, labels=c("no", "yes"), 
                                 levels=c("no", "yes"))

classes1 <- sapply(training[1,], class)
table(classes1)
classes2 <- sapply(testing[1,], class)
table(classes2)

classes1[classes1=="character"]
names(classes1[classes1=="character"])
classes_character <- names(classes1[classes1=="character"])
summary(training[, classes_character])

for (i in 1:34) {
  print(classes_character[i])
  print(table(training[, classes_character[i]]))
}

#Handling NA and Inf Values
for (i in 2:34) {
  training[, classes_character[i]][training[, classes_character[i]]==""] <- NA
  training[, classes_character[i]][training[, classes_character[i]]=="#DIV/0!"] <- Inf
  training[, classes_character[i]] <- as.numeric(as.character(training[, classes_character[i]]))
  print(classes_character[i])
  print(table(training[, classes_character[i]]))
  print(class(training[, classes_character[i]]))
}

sapply(training[, classes_character], as.numeric)
classes1 <- sapply(training[1,], class)
table(classes1)

table(training$classe)
table(training$user_name)
table(training$new_window)
table(training$classe, training$new_window)
table(training$classe, training$num_window)
plot(training$classe, training$num_window)

table(training$min_yaw_forearm)
table(training$max_yaw_forearm)
table(training$cvtd_timestamp)
table(training$new_window)

summary(training)
sapply(training[1,], class)

classes2[classes2=="logical"]
names(classes2[classes2=="logical"])
classes2_logical <- names(classes2[classes2=="logical"])
summary(testing[, classes2_logical])

for (i in 1:100) {
  print(classes2_logical[i])
  print(table(testing[, classes2_logical[i]]))
}

for (i in 1:100) {
  # testing[, classes2_logical[i]] <- as.numeric(as.character(testing[, classes2_logical[i]]))
  testing[, classes2_logical[i]] <- as.numeric(testing[, classes2_logical[i]])
  print(classes2_logical[i])
  print(table(testing[, classes2_logical[i]]))
  print(class(testing[, classes2_logical[i]]))
}

summary(testing)
sapply(testing[1,], class)

classes2 <- sapply(testing[1,], class)
table(classes2)
table(classes1)

table(classes1, classes2)

save(training, file="training.RData")
save(testing,  file="testing.RData")

# Extract variables with values not NA in a new DF
mean0 <- sapply(training, mean)
training_mod <- training[, na.omit(names(mean0)[as.numeric(!(mean0=="NA"))==1])]
testing_mod <-  testing[, names(training_mod)]

# Include user_name and classe
training_mod$user_name <- training$user_name
testing_mod$user_name  <- testing$user_name
training_mod$classe    <- training$classe

# Complete DF with training and testing data. We change classe variable to numeric
temp <- training
temp$classe <- as.numeric(temp$classe)
temp2 <- testing
temp2$problem_id <- NULL
temp2$classe <- 6

all <- rbind(temp, temp2)

# Delete temp DF
rm(temp)
rm(temp2)

names(all)

minimum <- all[, c(2, 5, 6, 160)]

minimum$cvtd_timestamp <- as.POSIXct(strptime(minimum$cvtd_timestamp, "%d/%m/%Y %H:%M"))
minimum <- minimum[order(minimum$user_name, minimum$cvtd_timestamp),]

# Some plots to study data
plot(minimum$cvtd_timestamp, minimum$classe, col=minimum$user_name, pch=19)
qplot(cvtd_timestamp, classe, data=all, geom="jitter", colour=factor(user_name), 
      main="Classe by time", ylab="Classe", xlab="Time")

qplot(cvtd_timestamp, classe, data=all, geom="jitter", colour=factor(user_name), 
      main="Classe by time", ylab="Classe", xlab="Time") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Principal components analysis
pca.out <- prcomp(training_mod[, -c(57, 58)], scale.=TRUE)
pca.out
summary(pca.out)
biplot(pca.out, scale=0)
biplot(pca.out, scale=0, cex=.6)

plot(pca.out$x[,1], pca.out$x[,2], xlab="PC1", ylab="PC2")

str(pca.out)
pca.out$rotation
names(pca.out)

# Correlation Only numeric variables
M_cor <- abs(cor(training[,-c(2, 5, 6, 160)]))
M_cor
diag(M_cor) <- 0
which(M_cor>0.8, arr.ind=T)

# variables with correlation > 0.8
new_names <- row.names(which(M_cor>0.8, arr.ind=T))
new_names

# New DF with variables with cor > 0.8. We also include timestamp and classe
training_mod3 <- training[, new_names]
training_mod3$cvtd_timestamp <- training$cvtd_timestamp
training_mod3$classe         <- training$classe

# Same for testing 
testing_mod3 <- testing[, new_names]
testing_mod3$cvtd_timestamp <- testing$cvtd_timestamp
testing_mod3$classe         <- testing$classe

# Models. We use the training_mod3 dataset with relevant feature.

# Random Forest
model30 <- train(classe ~., method="rf", preProcess=c("center", "scale"),
                 trControl=trainControl(method="cv", number=4),data=training_mod3 )
pred30_train <- predict(model30, training_mod3)
table(pred30_train, training_mod3$classe)
pred30 <- predict(model30, testing_mod3)
print(model30)
pred30
confusionMatrix(pred30_train, training_mod3$classe)

# C5.0
model32 <- train(classe ~., method = "C5.0", preProcess=c("center", "scale"),
                 trControl=trainControl(method="cv", number=4),data=training_mod3 )
pred32_train <- predict(model32, training_mod3)
table(pred32_train, training_mod3$classe)
pred32 <- predict(model32, testing_mod3)
print(model32)
pred32
confusionMatrix(pred32_train, training_mod3$classe)

# Linear SVM
model33 <- train(classe ~., method = "svmLinear", preProcess=c("center", "scale"),
                 trControl=trainControl(method="cv", number=4),data=training_mod3 )
pred33_train <- predict(model33, training_mod3)
table(pred33_train, training_mod3$classe)
pred33 <- predict(model33, testing_mod3)
print(model33)
pred33
confusionMatrix(pred33_train, training_mod3$classe)

# Radial SVM
model34 <- train(classe ~., method = "svmRadial", preProcess=c("center", "scale"),
                 trControl=trainControl(method="cv", number=4),data=training_mod3 )
pred34_train <- predict(model34, training_mod3)
table(pred34_train, training_mod3$classe)
pred34 <- predict(model34, testing_mod3)
print(model34)
pred34
confusionMatrix(pred34_train, training_mod3$classe)

##################

# Divide each of these 4 sets into training (60%) and test (40%) sets.
set.seed(666)
inTrain <- createDataPartition(y=training_mod3$classe, p=0.6, list=FALSE)
df_small_training1 <- training_mod3[inTrain,]
df_small_testing1 <- training_mod3[-inTrain,]
View(df_small_training1)

#Random Forest
model30 <- train(classe ~., method="rf", preProcess=c("center", "scale"),
                 trControl=trainControl(method="cv", number=4),data=df_small_training1 )
pred30_train <- predict(model30, df_small_training1)
table(pred30_train, df_small_training1$classe)
pred30 <- predict(model30, df_small_testing1)
print(model30)
pred30
confusionMatrix(pred30_train, df_small_training1$classe)

# C5.0
model31 <- train(classe ~., method = "C5.0", preProcess=c("center", "scale"),
                 trControl=trainControl(method="cv", number=4),data=df_small_training1 )
pred31_train <- predict(model31, df_small_training1)
table(pred31_train, df_small_training1$classe)
pred31 <- predict(model31, df_small_testing1)
print(model31)
pred31
confusionMatrix(pred31_train, df_small_training1$classe)

# Linear SVM
model33 <- train(classe ~., method = "svmLinear", preProcess=c("center", "scale"),
                 trControl=trainControl(method="cv", number=4),data=df_small_training1 )
pred33_train <- predict(model33, df_small_training1)
table(pred33_train, df_small_training1$classe)
pred33 <- predict(model33, df_small_testing1)
print(model33)
pred33
confusionMatrix(pred33_train, df_small_training1$classe)

# Radial SVM
model34 <- train(classe ~., method = "svmRadial", preProcess=c("center", "scale"),
                 trControl=trainControl(method="cv", number=4),data=df_small_training1 )
pred34_train <- predict(model34, df_small_training1)
table(pred34_train, df_small_training1$classe)
pred34 <- predict(model34, df_small_testing1)
print(model34)
pred34
confusionMatrix(pred34_train, df_small_training1$classe)


#Decision Tree Graph
library(rpart)
fit <- rpart(classe ~., method="class",data=training_mod3 )

# plot tree 
plot(fit, uniform=TRUE, main="Classification Tree ")
text(fit, use.n=TRUE, all=TRUE, cex=.8)

#FancyPlot
fancyRpartPlot(fit)


