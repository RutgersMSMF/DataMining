## NAME: Steven Semeraro
## NETID: ss3403

################
## Code for HW 2 problem 3, part (a) and (b).
##
## INSTRUCTIONS:
## The following file uses logistic ridge regression to predict the outcome of
## Shaquille O'Neal's free throw from 2006 to 2011
## 
## For part (a), follow the instructions for exploratory data analysis in section 
##               labeled part (a)
##
## For part (b), fill in the code in parts labeled "FILL IN". There are FOUR parts to fill in.
##
################

library(glmnet)

set.seed(1)

# shaq = read.csv("shaq.csv", as.is=TRUE)
shaq = read.csv("C:/Users/steve/Documents/GitHub/DataMining/Project2/shaq.csv", as.is = TRUE)

## Part (a)
## COMPUTE each of the following:

## 1. average accuracy of Shaq's free throw

## 2. average accuracy of Shaq's free throw during a home game

## 3. average accuracy of Shaq's free throw when the free throw
##    is the first of the two free throws.

## 4. perform a chi-squared test for association between 
##    a free throw result and whether the game is a home game or not.

## 5. perform a chi-squared test for association between
##    a free throw result and whether the free throw is the first of the 
##    two free throws.


## end of part (a)

# HELPER METHODS

L = length(shaq$shot_made)

homeGame <- function() {
  
  count = 0
  totalcount = 0
  result = 0
  
  for (i in 1:length(shaq$home_game)) {
    
    if (shaq$home_game[i] == 1 & shaq$shot_made[i] == 1) {
      count = count + 1
    }
    
    if (shaq$home_game[i] == 1) {
      totalcount = totalcount + 1
    }
    
    
  }
  
  result = (count / totalcount) 
  
  return(result)
  
}

shaq_average = sum(shaq$shot_made) / L
shaq_average_home = homeGame()
shaq_average_first = sum(shaq$made_first) / sum(shaq$first_shot)

test1 = chisq.test(shaq$home_game, shaq$shot_made)
print(test1$p.value)

test2 = chisq.test(shaq$first_shot, shaq$shot_made)
print(test2$p.value)

features = c("first_shot", 
             "missed_first", 
             "home_game", 
             "cur_score", 
             "opp_score", 
             "cur_time", 
             "score_ratio",
             "made_first", 
             "losing")


ntrial = 100
err1 = rep(0, ntrial)
err2 = rep(0, ntrial)

for (it in 1:ntrial){
  
  print(it)
  X = shaq[, features]
  Y = shaq[, "shot_made"]

  n = nrow(X)
  ntrain = 1500
  ixs = sample(1:n, ntrain)
  
  X1 = as.matrix(X[ixs, ])
  Y1 = Y[ixs]
  
  X2 = X[!(1:n %in% ixs), ]
  Y2 = Y[!(1:n %in% ixs)]
  
  p = ncol(X)
  
  ## FILL IN: use the "cv.glmnet" function with alpha = 0 to perform cross-validation for 
  ## logistic ridge regression parameter on X1, Y1
  cv_result = cv.glmnet(X1, Y1, family = "binomial", alpha = 0)

  ## FILL IN: use the "glmnet" function with alpha = 0 and the result of cv.glmnet to compute the 
  ## logistic ridge regression result.
  glmnet_result = cv_result$glmnet.fit
    
  beta = glmnet_result$beta[, 1]
  b = glmnet_result$a0
  
  ## FILL IN: use the "beta" and "b" to predict the label of the test data
  Y2hat = mean(Y2)
  myerr = mean(abs(Y2 - Y2hat))
  
  ## FILL IN: compute the baseline prediction. The baseline predicts all 1 if 
  ## the average of Y1 is at least 0.5, otherwise all 0.
  if (mean(Y1) >= 0.5) {
    Y2baseline = 1
  } else {
    Y2baseline = 0
  }
    
  baseline_err = mean(abs(Y2 - Y2baseline))
  
  err1[it] = myerr
  err2[it] = baseline_err
  
}

plot(err1)

print(sprintf("Ridge error: %.4f +/- %.4f   Baseline error: %.4f +/- %.4f", 
              mean(err1), 2*sd(err1), mean(err2), 2*sd(err2)))





