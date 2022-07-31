######################### Machine Learning Project #############################
############################ By Tamir Karamani #################################
######################### Heart Disease Prediction #############################

### Libraries Needed ###

library(ISLR)
library(boot)
library(InformationValue)
library(randomForest)
library(class)
library(tree)
library(bestglm)
library(corrplot)
library(PerformanceAnalytics)
library(fitdistrplus)


# Uploading Dataset 

heart = read.csv("C:/Users/sensa/Desktop/data science project/heart.csv",stringsAsFactors = TRUE)
#heart = read.csv("C:/Users/tamir/Desktop/Data Scientist Projects/Subjects/Statistic Learning With R/Machine Learning Project/Data/heart.csv")
head(heart)

# comparing positives and negatives of the dataset

sum(heart$HeartDisease == 0)
sum(heart$HeartDisease == 1)
hist(heart$HeartDisease)

# checking if the dataset contains NA values

sum(is.na(heart))

# Checking the independent variables before subset selection

NumericHeart = matrix(data = 0, nrow = nrow(heart) , ncol = ncol(heart)-1)
for (j in c(1:11)){
  NumericHeart[,j] = as.numeric(heart[,j]) 
}
colnames(NumericHeart) = colnames(heart[,c(1:11)])
heart.cor = cor(NumericHeart)
corrplot(heart.cor, type = "full")
chart.Correlation(NumericHeart,histogram = T , method = "pearson")
descdist(heart$HeartDisease, discrete = F)
boxplot(heart) #The boxplot shows that there are out of range observation at 3 independent variable.
#we will see that those 3 are removed by best subset.

# Factorizing and level configurations of the relevant features

heart$FastingBS = factor(heart$FastingBS)
heart$HeartDisease = factor(heart$HeartDisease)

levels(heart$FastingBS) = c(0,1)
levels(heart$ExerciseAngina) = c(0,1)
levels(heart$HeartDisease) = c(0,1)
levels(heart$Sex) = c(0,1)
levels(heart$ChestPainType) = c(0,1,2,3)
levels(heart$ST_Slope) = c(0,1,2)

# Best Subset calculation

BestSubset = bestglm(heart,IC = "AIC",family=binomial)
Best.Features = colnames(BestSubset$BestModel$model[-1])
heart = heart[,-c(4,7,8)]


# selection of train and test datasets

set1_index = sample(length(heart$HeartDisease),0.8*length(heart$HeartDisease))
train.set = heart[set1_index,]
test.set = heart[-set1_index,]

# Normalization function for KNN usage

normalize = function(x,NormVec)
{
  for (d in NormVec){
      x[,d] = (x[,d]-min(x[,d]))/(max(x[,d])-min(x[,d]))
  }
  return(x)
}

# Finding optimal trees for later usage

rfcs = randomForest(formula = HeartDisease ~., data = train.set)
opt_tree = which.min(rfcs$err.rate)

# Normal tree calculation to check random forest efficiency

#Tree.Heart = tree(HeartDisease ~. -RestingBP - Cholesterol - RestingECG, data = train.set)
#plot(Tree.Heart)
#cv.tree = cv.tree(Tree.Heart, K = 10)
#plot(cv.tree)
#prune.Tree=prune.tree(Tree.Heart,best=5)
#plot(prune.Tree)
#text(prune.Tree)

# Creating cross validation sets 

LogOptThr = c()
LogCVAcc = c()
RFTable = list()
LogAcc = c()
RFAcc = c()
KNNCVAcc = c()

# Calculation of several K fold cases. run time might extend as k is higher.

for (k in 2:20){
  
  # Random Forest
  
  rfcs_cv = rfcv(train.set,train.set$HeartDisease, 
                 step = 0.8, cv.fold = k )
  mtry_min = rfcs_cv$n.var[rfcs_cv$error.cv==min(rfcs_cv$error.cv)]
  
  rfcs = randomForest(formula = HeartDisease ~., data = train.set, 
                      ntree=opt_tree, mtry = mtry_min )
  RFpred = predict(rfcs, train.set)
  RFTable[[k-1]] = table(train.set$HeartDisease,RFpred)
  RFAcc[k-1] = ( RFTable[[k-1]][1] + RFTable[[k-1]][4]) / sum(RFTable[[k-1]])
  
  # Declaring Parameters
  
  LogConfMat = list()
  LogAcc = c()
  KNNTable = list()
  KNNAcc = c()
  train_set=train.set
  fold_size = floor(nrow(train_set)/k)
  k_fold_sets = list()
  index = list()
  
  for(i in 1:k)
  {
    if(nrow(train_set)>=fold_size)
    {
      index[[i]] = sample(1:nrow(train_set),fold_size, replace = F)
      k_fold_sets[[i]] = train.set[index[[i]],]
      train_set = train_set[-index[[i]],]
    } else {k_fold_sets[[i]]=train_set}
  }
    
    # Parameters deceleration for optimal cutoff
    
  LogYauden = c()

  for(i in 1:length(k_fold_sets))
  {
    # Logistic Regression
    
    train_kf_set = train.set[-index[[i]],]
    test_kf_set = train.set[index[[i]],]
    log_reg = glm(HeartDisease ~., data = train_kf_set, family = binomial)
    LogPred = predict(log_reg, test_kf_set, type = "response")
    LogYauden[i] = optimalCutoff(test_kf_set$HeartDisease, LogPred, optimiseFor = "Both")
    LogConfMat[[i]] = confusionMatrix(test_kf_set$HeartDisease, LogPred, threshold = LogYauden[i])
    LogAcc[i] = ( LogConfMat[[i]]$`0`[1] + LogConfMat[[i]]$`1`[2] ) / sum(LogConfMat[[i]])
    
    # KNN
    
    train_kf_set_n = normalize(train_kf_set,c(1,4,7))
    test_kf_set_n = normalize(test_kf_set,c(1,4,7))
    for (j in c(1,4,7)){
      train_kf_set_n[,j] = as.numeric(train_kf_set_n[,j]) 
      test_kf_set_n[,j] = as.numeric(test_kf_set_n[,j]) 
    }
    
    KNN = knn(train = train_kf_set_n, test = test_kf_set_n, cl = train_kf_set_n$HeartDisease, k = 3)
    KNNTable[[i]] = table(test_kf_set_n$HeartDisease,KNN)
    KNNAcc[i] = (KNNTable[[i]][1]+KNNTable[[i]][4])/sum(KNNTable[[i]])
  }

  # Calculation of CV accuracy of LOG and KNN
  
  #LogOptThr[k-1] = mean(LogYauden, na.rm = T) # for optimal cutoff calculation unmark this
  LogCVAcc[k-1] = mean(LogAcc)
  KNNCVAcc[k-1] = mean(KNNAcc)
}

# Calculation of the best K value for RF , LOG , KNN

RFBestK = (which(RFAcc == max(RFAcc))+1)[1]
LogBestK = (which(LogCVAcc == max(LogCVAcc))+1)[1]
KNNBestK = (which(KNNCVAcc == max(KNNCVAcc))+1)[1]

par(mfrow = c(4, 1))
plot(x = 2:k , y = LogCVAcc ,main = "Logistic Regression ~ Accuracy VS K", type = 'b' , xlab = 'k value' , ylab = 'Accuracy')
plot(x = 2:k , y = KNNCVAcc ,main = "KNN ~ Accuracy VS K", type = 'b' , xlab = 'k value' , ylab = 'Accuracy')
plot(x = 2:k , y = RFAcc ,main = "Random Forest ~ Accuracy VS K", type = 'b' , xlab = 'k value' , ylab = 'Accuracy')

# Visualizing the best method by accuracy

LogCVAcc
RFAcc
KNNCVAcc

Logg.Mean = mean(LogCVAcc)
RF.Mean = mean(RFAcc)
KNN.Mean = mean(KNNCVAcc)
Mean.Vec = c(Logg.Mean,RF.Mean,KNN.Mean)
Method.Vec = as.factor(c("Logistic Reg","RF","KNN"))

plot(x=Method.Vec,y = Mean.Vec ,main = "Method VS Mean Accuracy", type = 'b' , xlab = 'Method' , ylab = 'Accuracy')

print("The best accuracy is achieved by KNN method")

############## Testing ###############

# KNN

train.set_n = normalize(train.set,c(1,4,7))
test.set_n = normalize(test.set,c(1,4,7))

for (j in c(1,4,7)){
  train.set_n[,j] = as.numeric(train.set_n[,j]) 
  test.set_n[,j] = as.numeric(test.set_n[,j]) 
}

KNN = knn(train = train.set_n, test = test.set_n, cl = train.set_n$HeartDisease, k = 3)
KNNTable = table(test.set_n$HeartDisease,KNN)
KNNAcc = (KNNTable[1]+KNNTable[4])/(sum(KNNTable))








