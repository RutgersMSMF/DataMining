
################
## Code for HW 3 problem 3
##
## INSTRUCTIONS:
## The following file implements adaBoost with decision stump as the weak
## learner.
##
## It uses adaBoost to predict whether a movie has an above average rating or
## not according to various features. 
## 
## Fill in the code in parts labeled "FILL IN". In part (a), you will complete
## the implementation of a decision stump classifier. In part (b), you will
## complete the implementation of adaBoost. 
##
################


################
## Part (b)
##
## adaBoost
## INPUT: X (n--by--p) matrix, Y (n--by--1) vector of {-1, +1}, M integer
## OUTPUT: a list of 4 elements
##          "feat_to_split" (M--by--1) vector describing which feature to 
##                          split for the decision stump created at iteration m
##          "thresh_to_split" (M--by--1) vector describing where to split
##                          the feature
##          "alphas" (M--by--1) the alpha coefficient as described in the 
##                           adaboost algorithm
##          "errs" (M--by--1) the WEIGHTED predictive error of the m-th decision
##                            stump
##
##

sumError <- function(predicted, true, wts) {
  
  binaryError = rep(0, length(predicted))
  
  index = 1
  for (p in predicted) {
    
    if (p == true[index]) {
      binaryError[index] = 0
    } else {
      binaryError[index] = 1 * wts[index]
    }
    
    index = index + 1
    
  }
  
  return(binaryError)
  
}

IndicatorFunction <- function(predicted, true) {
  
  binaryError = rep(0, length(predicted))
  
  index = 1
  for (p in predicted) {
    
    if (p == true[index]) {
      binaryError[index] = 0
    } else {
      binaryError[index] = 1
    }
    
    index = index + 1
    
  }
  
  return(binaryError)
  
}


adaBoost <- function(X, Y, M) {
  
  n = nrow(X)
  
  ## FILL IN
  wts = rep((1 / n), n)
  
  feat_to_split = rep(0, M)
  thresh_to_split = rep(0, M)
  errs = rep(0, M)
  alphas = rep(0, M)
  
  for (m in 1:M){
    
    stump = decisionStumpClassifier(X, Y, wts)
    feat_to_split[m] = stump[1]
    thresh_to_split[m] = stump[2]
    
    Ypred = ifelse(X[, feat_to_split[m]] > thresh_to_split[m], 1, -1)
    
    errorVector = sumError(Ypred, Y, wts)

    ## FILL IN
    errs[m] = sum(errorVector) / sum(wts)

    ## FILL IN
    alphas[m] = log((1 - errs[m]) / errs[m])
    
    ## FILL IN
    new_vector = IndicatorFunction(Ypred, Y)
    wts = wts * exp(alphas[m] * new_vector)

  }
  
  return(list(feat_to_split, thresh_to_split, alphas, errs))
  
}


################
## Part (a)
## 
## decisionStumpClassifier
## INPUT:  X  (n--by--p) matrix, Y length n vector of {-1, +1}, 
##         wts length n vector of non-negative numbers.
## OUTPUT: a length three vector "result"
##         result[1]: the feature on which to split
##         result[2]: the value on which to split
##         result[3]: the weighted predictive error
##
##  NOTE: for adaBoost, we do not need to specify 
##        whether we should predict -1 on left and +1 on right or vice versa.
##        We only need to specify which feature and which value to split.
## 

decisionStumpClassifier <- function(X, Y, wts) {
  
  p = ncol(X)
  n = nrow(X)
  thresh = rep(0, p)
  best_errs = rep(0, p)
  
  for (j in 1:p){
    
    sorted_ixs = order(X[, j])
    feat = X[sorted_ixs, j]
    
    ## We need the following since the
    ## values of j-th feature may not be unique
    breaks = which(feat[2:n] - feat[1:(n-1)] > 0)

    Yord = Y[sorted_ixs]
    wts_o = wts[sorted_ixs]
    
    Yrev = rev(Yord)
    rwts_o = rev(wts_o)
    
    ## minus_left: a length (n-1) vector such that
    ##      minus_left[i] = sum_{j=1}^i wts_j * Indicator(Y_j == -1) 
    ## This is the error accrued by predicting data point
    ## from 1, 2, ... i as -1
    minus_left = cumsum(wts_o * (1 + Yord) / 2)[1:(n-1)]
    
    ## plus_right: a length (n-1) vector such that
    ##      plus_right[i] = sum_{j=i+1}^n wts_j * Indicator(Y_j == +1) 
    ## This is the error accrued by predicting data point
    ## from i+1, i+2, ..., n as +1
    plus_right = rev(cumsum(rwts_o * (1 - Yrev) / 2)[1:(n-1)])
    
    ## plus_left: a length (n-1) vector such that
    ##      plus_left[i] = sum_{j=1}^i wts_j * Indicator(Y_j == +1) 
    ## This is the error accrued by predicting data point
    ## from 1, 2, ... i as +1  
    ## FILL IN
    plus_left = cumsum(wts_o * (1 - Yord) / 2)[1:(n-1)]

    ## minus_right: a length (n-1) vector such that
    ##      minus_right[i] = sum_{j=1}^i wts_j * Indicator(Y_j == -1) 
    ## This is the error accrued by predicting data point
    ## from i+1, i+2, ..., n as -1  
    ## FILL IN
    minus_right =  rev(cumsum(rwts_o * (1 + Yrev) / 2)[1:(n-1)])
    
    predict_err_minus_left_plus_right = minus_left + plus_right
    predict_err_plus_left_minus_right = plus_left + minus_right
      
    errs = pmin(predict_err_minus_left_plus_right, predict_err_plus_left_minus_right)
    
    ix = which(errs == min(errs[breaks]))
    
    thresh[j] = X[sorted_ixs[ix[1]], j]
    best_errs[j] = min(errs[breaks])
    
  }
  
  ## FILL IN
  best_j = which.min(best_errs)
  result = c(best_j, thresh[best_j], min(best_errs))
  return(result)
  
}


# Test decision stump:
# X1 = c(1,1,1,1,2,2,2,3,4,4,4,4,4)
# X2 = runif(13)
# X = cbind(X1, X2)
# Y = c(rep(1,6), rep(-1,7))
# test_result = decisionStumpClassifier(X, Y, rep(1/13, 13))

####################
## Running adaBoost on the movies data set. 
##
####################

set.seed(1)

# Initial Path
# movies = read.csv("movies_hw3.csv")

# Local Machine Path
movies = read.csv("C:/Users/steve/Documents/GitHub/DataMining/Project3/movies_hw3.csv")
# movies = read.csv("C:/Users/18627/Documents/GitHub/DataMining/Project3/movies_hw3.csv")
movies = na.omit(movies)

Y = ifelse(movies[, "vote_average"] > mean(movies[, "vote_average"]), 1, -1)

feats = c("runtime", "log_budget", "age", "Action", "Adventure", "Romance", "Comedy", "Drama",
          "Horror", "Thriller")
X = movies[, feats]

ntrain = 600 
test_ix = sample(nrow(movies), nrow(movies) - ntrain)

X1 = X[-test_ix, ]
Y1 = Y[-test_ix]

X2 = X[test_ix, ]
Y2 = Y[test_ix]

p = ncol(X)

M = 100
ada_res = adaBoost(X1, Y1, M)
alphas = ada_res[[3]]
feat_to_split = ada_res[[1]]
thresh_to_split = ada_res[[2]]

Hval1 = rep(0, length(Y1))
Hval2 = rep(0, length(Y2))

train_errs = rep(0, M)
test_errs = rep(0, M)

for (m in 1:M){
  
  Hval1 = Hval1 + alphas[m]*ifelse(X1[, feat_to_split[m]] > thresh_to_split[m], 1, -1)
  Hval2 = Hval2 + alphas[m]*ifelse(X2[, feat_to_split[m]] > thresh_to_split[m], 1, -1)
  
  Y1pred = sign(Hval1)
  Y2pred = sign(Hval2)
  
  train_errs[m] = mean(abs(Y1 - Y1pred))/2
  test_errs[m] = mean(abs(Y2 - Y2pred))/2
  
}

plot(1:M, train_errs, type='l', col="blue", ylim=c(.1,.5), xlab="predictive error", ylab="number of iterations")
lines(1:M, test_errs, col="red")

baseline_error = ifelse(mean(Y1) > 0, mean(abs(1 - Y2))/2, mean(abs(1 + Y2))/2)

print(sprintf("Baseline Error: %.3f  AdaBoost Error: %.3f", baseline_error, test_errs[M]))






