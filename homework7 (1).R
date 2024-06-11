---
  title: "R Notebook"
output: html_notebook
---
  
  ```{r}
#install.packages('randomForest')
#install.packages('stringr')
```

. 
```{r}
require(caret)
require(e1071)
require(rpart)
require(dplyr)
require(stringr)
require(randomForest)
library(rsample)
```

```{r}
options(java.parameters = "-Xmx2560m")
require(rJava)

library("RWeka")
trainset <- read.csv("/Users/gozi/Downloads/digit-recognizer/train.csv")
testset <- read.csv("/Users/gozi/Downloads/digit-recognizer/test.csv")
trainset$label=factor(trainset$label)
```




```{r}
search_grid = expand.grid(k = c(5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25))

# set up 3-fold cross validation procedure
train_control <- trainControl(
  method = "cv", 
  number = 3
)

# more advanced option, run 5 fold cross validation 10 times
train_control_adv <- trainControl(
  method = "repeatedcv", 
  number = 3,
  repeats = 10
)

# train model
knn <- train(label ~ .,
             data = trainset,
             method = "knn",
             trControl = train_control_adv,
             tuneGrid = search_grid
)

```

```{r}
# top 5 modesl
knn$results %>% 
  top_n(5, wt = Accuracy) %>%
  arrange(desc(Accuracy))

# results for best model
confusionMatrix(knn)

pred <- predict(knn, newdata = testset)
```



```{r}
myids=c("labels")
id_col=testset[myids]
combined_pred=cbind(id_col, pred)
head(combined_pred)

colnames(combined_pred)=c("labels")

write.csv(combined_pred, file="digit-KNN-pred.csv", row.names=FALSE)
```



```{r}
# Introduction to SVM algorithm

```{r}
library(tidyverse)    # data manipulation and visualization
library(kernlab)      # SVM methodology
library(e1071)        # SVM methodology
library(RColorBrewer) # customized coloring of plots
```

## Review the SVM Classifier

```{r}


# Plot data
ggplot(data = trainset, aes(x = x.2, y = x.1, color = y, shape = y)) + 
  geom_point(size = 2) +
  scale_color_manual(values=c("#000000", "#FF0000")) +
  theme(legend.position = "none")
```


```{r}
svm <- make_Weka_classifier("weka/classifiers/functions/SMO")
svm_model=svm(label~., data=trainset)
e4 <- evaluate_Weka_classifier(svm_model, numFolds = 3, seed = 1, class = TRUE)
e4
```

```{r}
WOW("weka/classifiers/trees/RandomForest")
randomforest <- make_Weka_classifier("weka/classifiers/trees/RandomForest")
# build default model with 100 trees
rf_model=randomforest(label~., data=trainset)
# build a model with 10 trees instead
rf_model=randomforest(label~., data=trainset, control=Weka_control(I=10))
e5 <- evaluate_Weka_classifier(rf_model, numFolds = 3, seed = 1, class = TRUE)
e5
```

I am unable to get the code to run, but wanted to turn in something to get points 
