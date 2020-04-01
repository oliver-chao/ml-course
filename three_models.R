
#packages
library(cvTools)
library(FNN)
source("setup.R")


#load data
Data = read.table("http://www-stat.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data",sep=",",head=T,row.names=1)

#one out of k coding and create a new column 'famhistory'
Data$famhistory[Data$famhist=="Present"]=1
Data$famhistory[Data$famhist=="Absent"]=0

#delete the column 'famhist'
Data=Data[-5]
#extract all the features
X = Data[-9]
N = dim(X)[1]
attributeNames = attributes(X)
attributeNames <- as.vector(unlist(attributeNames$names))
attributeNames

#extract the output 'chd'
y = Data[9]



#k-fold crossvalidation   
K = 10
KK = 10

set.seed(1234)

CV <- cvFolds(N, K=K);
CV$TrainSize <- c()
CV$TestSize <- c()

# K-nearest neighbors parameters
L = 20; # Maximum number of neighbors
knn_parameter = c(1:L)
test_error_knn = array(rep(NA, times=K*L), dim=c(KK,L))

paramter_knn = c(rep(NA,K))
E_test_knn = c(rep(NA,K))

#basline
test_error_baseline = array(rep(NA,K * 3),c(K,3))

#logistic regression parameters
N_lambdas = 10 # the number of lambda
lambda_tmp <- 10^(seq(from=-2, to=0, length=N_lambdas)) # the range of lambda
test_error_logistic = array(rep(NA,N_lambdas*K),c(KK,N_lambdas))
colnames(test_error_logistic) <- lambda_tmp

paramter_logistic = c(rep(NA,K)) 
E_test_logistic = c(rep(NA,K))

num = c(rep(NA,K))


for(k in 1:K){
  
  paste('Crossvalidation fold ', k, '/', K, sep='')
  
  # Extract the training and test set
  X_train <- X[CV$subsets[CV$which!=k],];
  y_train <- y[CV$subsets[CV$which!=k],];
  X_test <- X[CV$subsets[CV$which==k],];
  y_test <- y[CV$subsets[CV$which==k],];
  
  #store the trainsize and testsize
  CV$TrainSize[k] <- length(y_train)
  CV$TestSize[k] <- length(y_test)
  
  #baseline
  m = names(which.max(table(y_train))) # the majority class
  m = as.integer(m)
  y_predict_baseline = c(rep(m,length(y_test)))
  e_test = sum(y_predict_baseline != y_test)/(length(y_test))
  
  test_error_baseline[k,1] = k
  test_error_baseline[k,2] = m
  test_error_baseline[k,3] = e_test
  
  #logistic regression regularization
  mu <- colMeans(X_train)
  sigma <- apply(X_train, 2, sd)
  X_train_logistic <- scale(X_train, mu, sigma)
  X_test_logistic <- scale(X_test, mu, sigma)
  
  
  # Use 10-fold crossvalidation to estimate optimal value of lambda and K
  KK <- 10
  CV2 <- cvFolds( dim(X_train)[1], K=KK)
  CV2$TrainSize <- c()
  CV2$TestSize <- c()
  
  for(kk in 1:KK){ # For each crossvalidation fold
    
    # Extract training and test set
    X_train2 <- X_train[CV2$subsets[CV2$which!=kk],];
    y_train2 <- y_train[CV2$subsets[CV2$which!=kk]];
    X_test2 <- X_train[CV2$subsets[CV2$which==kk],];
    y_test2 <- y_train[CV2$subsets[CV2$which==kk]];
    CV2$TrainSize[kk] <- length(y_train2)
    CV2$TestSize[kk] <- length(y_test2)
    
    #logistic regression regularization
    mu2 <- colMeans(X_train2)
    sigma2 <- apply(X_train2, 2, sd)
    X_train_logistic2 <- scale(X_train2, mu2, sigma2)
    X_test_logistic2 <- scale(X_test2, mu2, sigma2)
    mdl <- glmnet(X_train_logistic2, y_train2, family="binomial", alpha=0, lambda=lambda_tmp)
    
    # knn
    for(l in 1:L){ # For each number of neighbors
      # Evaluate test performance 
      y_test_est <- knn(X_train2, X_test2, cl=y_train2, k = l, prob = FALSE, algorithm="kd_tree")
      test_error_knn[kk,l] = sum(y_test2!=y_test_est)/length(y_test_est);
    }
    
    # logistic regression
  
    for(m in 1:N_lambdas){
      # Evaluate test performance 
      y_test_est <- predict(mdl, X_test_logistic2, type="class", s=lambda_tmp[m])
      
      test_error_logistic[kk,m] = sum(y_test_est != y_test2)/length(y_test2)
    }
    
  }
  
  
  # Evaluate generation error for knn and logistic regression 
  generation_error_knn = (CV2$TestSize/CV$TestSize[k]) %*% test_error_knn
  generation_error_logistic = (CV2$TestSize/CV$TestSize[k]) %*% test_error_logistic
  
  # get the optimal k and train knn to evaluate the performance
  optimal_knn = apply(generation_error_knn,1,which.min)
  paramter_knn[k] = knn_parameter[optimal_knn]
  y_test_est_knn = knn(X_train, X_test, cl=y_train, k = paramter_knn[k], prob = FALSE, algorithm="kd_tree")
  E_test_knn[k] =  sum(y_test_est_knn != y_test)/length(y_test)
  
  # get the optimal k and train logistic regression to evaluate the performance
  optimal_logistic = apply(generation_error_logistic,1,which.min)
  paramter_logistic[k] = lambda_tmp[optimal_logistic]
  logistic = glmnet(X_train_logistic, y_train, family="binomial", alpha=0, lambda=paramter_logistic[k])
  y_test_est_logistic =predict(logistic, X_test_logistic, type="class", s=paramter_logistic[k])
  E_test_logistic[k] = sum(y_test_est_logistic != y_test)/length(y_test)
}

E_test_knn
paramter_knn
# get the optimal k which has the minimum error
min_indx_knn = which.min(E_test_knn)
optimal_k = paramter_knn[min_indx_knn]

paramter_logistic
E_test_logistic
# get the optimal lambda which has the minimum error
min_indx_logistic = which.min(E_test_logistic)
optimal_lambda = paramter_logistic[min_indx_logistic]




# models comparison

#k-fold crossvalidation
p = 6
CV3 <- cvFolds(N, K=p);
CV3$TrainSize <- c()
CV3$TestSize <- c()

# matrix to store the true values and predicted values 
y_true = array(rep(NA,N),c(N/6,6))
y_est_k = array(rep(NA,N),c(N/6,6))
y_est_baseline = array(rep(NA,N),c(N/6,6))
y_est_logistic = array(rep(NA,N),c(N/6,6))



for(k in 1:p){ # For each crossvalidation fold
  print(paste('Crossvalidation fold ', k, '/', CV$NumTestSets, sep=''))
  
  # Extract training and test set
  X_train3 <- X[CV3$subsets[CV3$which!=k],];
  y_train3 <- y[CV3$subsets[CV3$which!=k],];
  X_test3 <- X[CV3$subsets[CV3$which==k], ];
  y_test3 <- y[CV3$subsets[CV3$which==k],];
  CV3$TrainSize[k] <- length(y_train)
  CV3$TestSize[k] <- length(y_test)
  
  y_true[,k] = y_test3
  
  
  #logistic regresson regularization
  mu3 <- colMeans(X_train3)
  sigma3 <- apply(X_train3, 2, sd)
  
  X_train_logistic3 <- scale(X_train3, mu3, sigma3)
  X_test_logistic3 <- scale(X_test3, mu3, sigma3)
  model_logistic = glmnet(X_train_logistic3, y_train3, family="binomial", alpha=0, lambda=optimal_lambda)
  y_est_logistic[,k] <- as.numeric(predict(model_logistic, X_test_logistic3, type="class", s=optimal_lambda))
  
  
  #baseline
  m = names(which.max(table(y_train3)))
  m = as.integer(m)
  y_predict_baseline = c(rep(m,length(y_test3)))
  y_est_baseline[,k] = y_predict_baseline
  
  
  
  #knn
  pre_knn <- knn(X_train3, X_test3, cl= y_train3, k = optimal_k, prob = FALSE, algorithm="kd_tree")
  y_est_k[,k]= as.numeric(as.character(pre_knn[1:(N/6)]))
}

y_est_k
y_true
y_est_logistic
y_est_baseline



# knn vs logistic regression
alpha = 0.05
rt1 <- mcnemar(y_true, y_est_k, y_est_logistic, alpha=alpha)
rt1$CI
rt1$p 
rt1$thetahat 

# baseline vs logistic regression
rt2 <- mcnemar(y_true, y_est_baseline, y_est_logistic, alpha=alpha)
rt2$CI
rt2$p 
rt2$thetahat 

# baseline vs knn
rt3 <- mcnemar(y_true, y_est_baseline, y_est_k, alpha=alpha)
rt3$CI
rt3$p 
rt2$thetahat 

