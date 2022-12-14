################
## Code for HW 4 problem 2
##
## INSTRUCTIONS:
##
## In this problem, we compare Linear Discriminant Analysis and PCA to data set of 
## 29,000 wine reviews.
## Each row is associated with a single wine. The feature "description" contains the 
## review. We also added binary features associated with a list of words that indicate
## whether that specific word was used in the review. 
## Another important feature is "variety", which indicates the type of wine. 
##
## In this problem, we consider only two varieties of white wines:
## "Chardonnay" and "Riesling". 
## 
##
## For part (a) of the problem, fill in the code in parts labeled "FILL IN". 
##
################

# wines = read.csv("wines_hw4.csv")
wines = read.csv("C:/Users/steve/Documents/GitHub/DataMining/Project4/wines_hw4.csv")
wines = wines[ wines$variety %in% c("Chardonnay", "Riesling"), ]


words = c("sweet", "acid", "earthy", "fruit", "tannin", "herb",
          "tart", "spice", "smooth", "full", "intense",
          "wood", "soft", "dry", "apple", "pear", "cherry",
          "berry", "aroma", "citrus", "lemon", "lime", "peach", "blossom",
          "sugar", "simple", "cinnamon", "ripe",
          "crisp", "honey", "brisk", "fresh", "sour", "floral",
          "dark", "complex", "oak", "balance", "caramel", "plum", "mint",
          "apricot", "cream", "vanilla", "butter", "sharp")

## NOTE: wines[, "apple"] is either 0/1. wines[i, "apple"] == 1 if the review of wine i uses
##       the word "apple" and 0 else.

X = wines[, words]
X = as.matrix(X)
X = scale(X)

p = ncol(X)
n = nrow(X)

## Perform dimensionality reduction with PCA. FILL IN.
svdMatrix = svd(X)
U = svdMatrix$u
D = diag(svdMatrix$d)

Z = (U %*% D)[, 1] ## FILL IN; Z should be n--by--1 vector

## We now perform Linear Discriminant Analysis
## Complete the code below

X1 = X[wines$variety=="Chardonnay", ]
X2 = X[wines$variety=="Riesling", ]

mu1 = colMeans(X1)
mu2 = colMeans(X2)

n1 = nrow(X1)
n2 = nrow(X2)

X1c = X1 - matrix(1, n1, 1) %*% t(mu1) 
X2c = X2 - matrix(1, n2, 1) %*% t(mu2)

S1 = (1 / n1) * sum((X1c**2)) ## FILL IN
S2 = (1 / n2) * sum((X2c**2)) ## FILL IN

top = solve(n1 / n * S1 + n2 / 2 * S2) * (mu1 - mu2)
bottom = abs((solve(n1 / n * S1 + n2 / 2 * S2) * (mu1 - mu2)))
w1 = top / bottom

## We plot LDA projection on the x-axis and 
## the first principal PCA projection on the y-axis to see 
## which projection better separates the data

plot_points = cbind(X %*% w1, Z)
plot(plot_points[, 1:2], cex=0.01, xlab="LDA Projection", ylab="PCA Projection")
points(plot_points[wines$variety=="Chardonnay", c(1,2)], col="blue", cex=0.04)
points(plot_points[wines$variety=="Riesling", c(1,2)], col="green", cex=0.04)

