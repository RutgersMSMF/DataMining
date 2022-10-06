
## NAME: Steven Semeraro
## NETID: ss3403


################
## Code for HW 1 problem 2
##
## INSTRUCTIONS:
## The following file performs ridge regression with cross-validation
## 
## For Part (a): Fill in the code in parts labeled "FILL IN". 
## For Part (b): Construct the features in the section labeled "part (b)"
##
################

# Old Path 
# movies = read.csv("movies_hw1.csv")

# Local Machine Path
set.seed(1)
movies = read.csv("C:/Users/steve/Documents/GitHub/DataMining/Project1/movies_hw1.csv")

#####
## For part (b): construct your features here. 

# Feature construction
# Example. movies$log_age = log(movies$age + 1)

movies$log_budget_sq = movies$log_budget**2
movies$log_revenue_sq = movies$log_revenue**2
movies$log_vote_count = log(movies$vote_count + 1)

# Helper Method
checkGenre <- function(arg1, arg2) {
  
  myData = rep(0, length(arg1))
  
  for (i in 1:length(arg1)) {
    
    if (arg1[i] == 1 & arg2[i] == 1) {
      myData[i] = 1
    } else {
      myData[i] = 0
    }
    
  }
  
  return(myData)
  
}

movies$action_adventure = checkGenre(movies$Action, movies$Adventure)
movies$rom_com = checkGenre(movies$Romance, movies$Comedy)
movies$vote_budget = movies$log_vote_count * movies$log_budget

# Helper Method
checkRuntime <- function(arg1) {
  
  myData = rep(0, length(arg1))
  
  for (i in 1:length(arg1)) {
    
    if (arg1[i] > 120) {
      myData[i] = 1
    } else {
      myData[i] = 0
    }
    
  }
  
  return(myData)
  
}

movies$long = checkRuntime(movies$runtime)

#####

n = 300 # Training Set
test_ix = sample(nrow(movies), nrow(movies) - n)

## Exclude title and vote_average
X = cbind(as.matrix(movies[, !(names(movies) %in% c("vote_average", "title"))]), rep(1, nrow(movies)))
Y = movies[, "vote_average"]

# Training Set
# Length : 300
X1 = X[-test_ix, ]
Y1 = Y[-test_ix] 

# Test Set
# Length: 2651
X2 = X[test_ix, ]
Y2 = Y[test_ix]

p = ncol(X)
K = 10

## FILL IN: randomly permute the n samples (rows of X1 and entries of Y1).
rand = sample(nrow(X1), size = n, replace = TRUE)
X1 = X1[rand, ]
Yperm = sample(Y1, size = n, replace = TRUE)

lambda_ls = c(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3)
errs = rep(0, length(lambda_ls))

index_matrix = matrix(1:n, ncol=K)

for (k in 1:K){
  
  valid_ix = index_matrix[, k]
  
  ## FILL IN:
  # Create variables Xtrain, Ytrain, Xvalid, Yvalid
  Xtrain = X1[valid_ix, 1:p]
  Ytrain = Y1[valid_ix]
  Xvalid = X2
  Yvalid = Y2

  for (il in 1:length(lambda_ls)){
    
    lambda = lambda_ls[il]
    beta_ridge = solve(t(Xtrain) %*% Xtrain + lambda*diag(p), t(Xtrain) %*% Ytrain, tol = 1e-20)
    errs[il] = mean((Xvalid %*% beta_ridge - Yvalid)^2)
      
  }
  
  print(errs)
  
  plot(
    1:length(lambda_ls), 
    errs, 
    xlab = 'Lambda', 
    ylab = 'Error', 
    main = 'Ordinary Least Squares',
    col = 'blue'
    )

}

# Print Results
min_index = which.min(errs)
lambda_star = lambda_ls[min_index]
beta_ridge_final = solve(t(X2) %*% X2 + lambda_star*diag(p), t(X2) %*% Y2, tol = 1e-20)

test_error = mean((X2 %*% beta_ridge_final - Y2)^2)
baseline = mean((mean(Y1) - Y2)^2)

print(sprintf("Test error: %.3f  Baseline: %.3f   lambda_star: %f", test_error, baseline, lambda_star))




