
################
## Code for HW 4 problem 1
##
## INSTRUCTIONS:
##
## In this problem, we apply PCA and K-means to a data set of 29,000 wine reviews.
## Each row is associated with a single wine. The feature "description" contains the 
## review. We also added binary features associated with a list of words that indicate
## whether that specific word was used in the review. 
## Another important feature is "variety", which indicates the type of wine. 
## This dataset contains only three varieties: Pinot Noir, Chardonnay, and Riesling.
## 
##
## For part (a) of the problem, fill in the code in parts labeled "FILL IN". 
##
################

# wines = read.csv("wines_hw4.csv")
wines = read.csv("C:/Users/steve/Documents/GitHub/DataMining/Project4/wines_hw4.csv")

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

#####################
## For part (a):

## Perform dimensionality reduction with PCA. FILL IN.
svdMatrix = svd(X)
U = svdMatrix$u
D = diag(svdMatrix$d)

Z = (U %*% D)[, 1:3]
plot(Z[, c(1,2)], cex=0.01)
points(Z[wines$variety == "Chardonnay", c(1,2)], col="blue", cex=0.04)
points(Z[wines$variety == "Pinot Noir", c(1,2)], col="red", cex=0.04)
points(Z[wines$variety == "Riesling", c(1,2)], col="green", cex=0.04)

###################
## For part (b):

V = svdMatrix$v
plot(V[, 1:2], cex=0.2, xlim=c(min(V[, 1])-0.1, max(V[, 1])+0.1), 
     ylim=c(min(V[, 2])-0.1, max(V[, 2])+0.1))
ixs = which(words %in% c("apple", "pear", "peach", 
                         "cherry", "berry", "citrus", 
                         "lime", "lemon", "caramel", "cream",
                         "sweet", "dry", "tannin", "acid"))
points(V[ixs, 1:2], col="red", cex=1)
text(V[ixs, 1:2], words[ixs], pos=1, cex=0.8)


###################
## For part (c):
##
## We cluster the wines using the k-means algorithm. We 
## will run k-means with K=3 clusters. 
##
set.seed(1) # DO NOT CHANGE
##
## In the first part, cluster the wines by using all the features,
## that is, using the entire X matrix. 


## FILL IN: perform k-means on X with the "kmeans" function.
kObj = kmeans(X, centers = 3)

## Create vector of indices indicating cluster membership
cluster1 = rep(0, length(kObj$cluster)) #FILL IN
cluster2 = rep(0, length(kObj$cluster)) #FILL IN
cluster3 = rep(0, length(kObj$cluster)) #FILL IN

index = 1
for (c in kObj$cluster) {
  
  if (c == 1) {
    cluster1[index] = 1
    cluster2[index] = 0
    cluster3[index] = 0
  }
  
  if (c == 2) {
    cluster1[index] = 0
    cluster2[index] = 1
    cluster3[index] = 0
  }
  
  if (c == 3) {
    cluster1[index] = 0
    cluster2[index] = 0
    cluster3[index] = 1
  }
  
  index = index + 1
  
}

C1 = table(wines[cluster1, "variety"])
C2 = table(wines[cluster2, "variety"])
C3 = table(wines[cluster3, "variety"])
print(cbind(C1,C2,C3))


## In this second part, we use the reduced features. 
## Use only the top three principal components. That is, we
## use the Z matrix where Z is n--by--3.

## FILL IN: perform k-means on Z with the "kmeans" function.
kObj = kmeans(Z, centers = 3)

## Create vector of indices indicating cluster membership
cluster1 = rep(0, length(kObj$cluster)) #FILL IN
cluster2 = rep(0, length(kObj$cluster)) #FILL IN
cluster3 = rep(0, length(kObj$cluster)) #FILL IN
  
index = 1
for (c in kObj$cluster) {
  
  if (c == 1) {
    cluster1[index] = 1
    cluster2[index] = 0
    cluster3[index] = 0
  }
  
  if (c == 2) {
    cluster1[index] = 0
    cluster2[index] = 1
    cluster3[index] = 0
  }
  
  if (c == 3) {
    cluster1[index] = 0
    cluster2[index] = 0
    cluster3[index] = 1
  }
  
  index = index + 1
  
}

C1 = table(wines[cluster1, "variety"])
C2 = table(wines[cluster2, "variety"])
C3 = table(wines[cluster3, "variety"])
print(cbind(C1,C2,C3))





