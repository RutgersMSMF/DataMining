## NAME: Steven Semeraro
## NETID: ss3403

################
## Code for HW 2 problem 2 part (a)
##
## INSTRUCTIONS:
## The following file implements kernel SVM with
## polynomial kernel. 
##
## It also trains kernel SVM classifier on a 
## dataset that contains sociodemographic information of all counties 
## in the United States as well as how these counties voted in the 2016
## presidential election. 
##
## The kernel SVM classifier is use to predict whether a county prefers
## Trump over Clinton based on sociodemographic features.
## 
## Fill in the code in parts labeled "FILL IN". There are FOUR parts to fill in.
##
################

## Trains kernel SVM via Projected Gradient Descent
## and with polynomial kernel
##
## INPUT: "X" n--by--p matrix, 
##        "Y" n--by--1 vector of labels
##        "C" scalar, weight placed on violation reduction
##        "d" degree of polynomial kernel
## OUTPUT:  


# For Matrix Power
library(expm) 

kernelSVM = function(X, Y, C, d){
  
  p = ncol(X)
  n = nrow(X)
  
  vars = rep(0, n)
  
  alpha = .2
  gamma = .9
  stepsize = 1
  
  ## FILL IN: compute the kernel between all pairs of training samples
  
  K = (X %*% t(X)) %^% d # Square Matrix (X * X)
  K2 = diag(Y) %*% K %*% diag(Y) # Kernel Matrix (Y * X * X * Y)
  
  it = 1
  while(TRUE){
    
    cur_obj = .5* t(vars) %*% K2 %*% vars - sum(vars)
      
    gradient = K2 %*% vars - rep(1, n)
    vars_new = dualproject(vars - stepsize * t(gradient), Y, C)
    
    ## backtracking line search
    while (.5* t(vars_new) %*% K2 %*% vars_new - sum(vars_new) > cur_obj + alpha * sum(gradient * (vars_new - vars))){
      
      stepsize = stepsize * gamma
      vars_new = dualproject(vars - stepsize * t(gradient), Y, C)
      
    }
    
    new_obj = .5 * t(vars_new) %*% K2 %*% vars_new - sum(vars_new)
    
    if (it %% 1000 == 0) {
      
      print(sprintf("Iteration: %d  objective: %.3f  conv threshold (log10) %.3f  stepsize %.4f",
                    it, new_obj, 
                    log10(mean((vars - vars_new)^2 / mean(vars^2))),
                    stepsize
      ))
      
    }
    
    it = it+1
    
    if (mean((vars - vars_new)^2)/mean(vars^2) < 1e-7) {
      break
    }
    else {
      vars = vars_new
    }
    
  }
  
  return(vars)
  
}

## Runs Dykstra's algorithm
##
## INPUT:  u is length-n vector
##         Y is length-n vector
##         C is a positive scalar
##  

dualproject <- function(u, Y, C) {
  
  n = length(u)
  
  v = u
  p = rep(0, n)
  q = rep(0, n)
  
  T = 1000
  for (it in 1:T){
    
    w = project2(v + p, Y)
    p = v + p - w
    v = project1(w + q, C)
    q = w + q - v

    if (mean((v-w)^2) < 1e-20) {
      break
    }
      
  }
  
  return(v)

}

project1 <- function(u, C){
  return(pmin(C, pmax(u, 0)))
}

project2 <- function(u, Y){
  return(u - sum(u * Y / sum(Y^2))*Y)
}

##################
##
##  Using kernel SVM to predict voting behavior in the 2016 US presidential election.
##
##

set.seed(1)

#votes = read.csv("votes.csv")
votes = read.csv("C:/Users/steve/Documents/GitHub/DataMining/Project2/votes.csv")

votes$prefer_trump = votes$trump > votes$clinton
features = c("white", 
             "black", 
             "poverty", 
             "density", 
             "bachelor", 
             "highschool", 
             "age65plus",
             "income", 
             "age18under", 
             "population2014")

## Modify this to 2, 3, 4, 5, and 6 for part (b)
deg = c(1, 2, 3, 4, 5, 6)

for (i in deg) {
  
  X = votes[, features]
  X = scale(X)
  Y = votes[, "prefer_trump"]
  
  ntrain = 400 
  test_ix = sample(nrow(votes), nrow(votes) - ntrain)
  
  Y[Y==0] = -1
  
  X1 = X[-test_ix, ]
  Y1 = Y[-test_ix]
  
  X2 = X[test_ix, ]
  Y2 = Y[test_ix]
  
  p = ncol(X)
  
  ## Run SVM
  C = .5
  
  print(i)
  alpha = kernelSVM(X1, Y1, C, i)
  ixs = which(alpha > 0 & alpha < C)
  
  ## FILL IN: complete the computation of the intercept b in kernel SVM
  summation1 = alpha %*% t(Y1) %*% X1 %*% t(X1)
  tempSummation = rowSums(summation1)[ixs]
  b = mean(Y1[ixs] - tempSummation) # From Class Notes
  
  ## FILL IN: compute the labels predicted by kernel SVM
  Y2_svm = sign(sum(summation1) + b) # From Class Notes
  
  svm_error = mean(abs(Y2_svm - Y2))/2 
  # NOTE: we divide by 2 because Y2 and Y2_svm are +1/-1
  
  ## FILL IN: compute the number of support vectors
  num_supp_vec = length(ixs)
  
  ## Run logistic regression in comparison
  Y1[Y1==-1] = 0
  Y2[Y2==-1] = 0
  
  X1 = cbind(X1, rep(1, nrow(X1)))
  X2 = cbind(X2, rep(1, nrow(X2)))
  
  w_lr = glm.fit(X1, Y1, family=binomial(link="logit"))$coefficients
  
  Y2_lr = (X2 %*% w_lr >= 0)
  lr_error = mean(abs(Y2_lr - Y2))
  
  baseline_error = ifelse(mean(Y1) > 0.5, mean(1 - Y2), mean(Y2))
  
  print(sprintf("Baseline Error: %.3f  kernelSVM Error: %.3f  LR Error: %.3f  #Support vec: %d", 
                baseline_error,
                svm_error, lr_error, num_supp_vec))
  
}








