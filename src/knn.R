################################################################################
### K-Nearest Neighbors (KNN)
knn = function(X, Y, myx, k, type=2, ztype="z-score", weight=NULL) {
  # X is a matrix, Y and myx are vectors
  n = nrow(X)
  p = ncol(X)
  
  # ------------------- Normalization --------------------
  if(ztype == 'z-score') {
    Xs = scale(X)
    myx_s = (myx - colMeans(X)) / apply(X, 2, sd)
  } else if(ztype == 'min-max') {
    xmin = apply(X, 2, min)
    xmax = apply(X, 2, max)
    rng = xmax - xmin
    Xs = sweep(sweep(X, 2, xmin, FUN = "-"), 2, rng, FUN = "/")
    myx_s = (myx - xmin) / rng
  } else if(ztype == 'none') {
    Xs = X
    myx_s = myx
  } else {
    stop("Unknown target space. Use 'z-score', 'min-max', or 'none'.")
  }
  
  # find the distances from the myx to all training points
  diff = Xs - matrix(myx_s, nrow = n, ncol = p, byrow=T)
  
  # ---------------- Apply Weights (optional) -----------------
  if (!is.null(weight)) {
    if (length(weight) == p) {
      # per-feature weights -> scale columns
      diff = sweep(diff, 2, weight, `*`)
    } else if (length(weight) == n) {
      # per-observation weights -> scale rows
      diff = diff * matrix(weight, nrow = n, ncol = p, byrow=F)
    }
  }
  
  # ----------------- Distance Metrics -----------------
  if(type == 1) {
    dist = rowSums(abs(diff))
  } else if(type == 2) {
    dist = sqrt(rowSums(diff^2))
  } else if(type == 'infty') {
    dist = apply(abs(diff), 1, max)
  } else {
    dist = (rowSums(abs(diff))^type)^(1/type)
  } 
  
  # find the k-nearest neighbors
  Y = factor(Y)
  idx = order(dist)[1:k]
  ylabel = Y[idx]
  ydist = dist[idx]
  
  # majority vote
  tab = table(ylabel)
  major_vote = names(tab)[tab == max(tab)][1]
  
  # distribution table
  distri_table = data.frame(class = names(tab),
                            count = as.integer(tab),
                            probs = as.integer(tab)/k)
  
  result = list('k_nearest_labels'   = ylabel,
                'k_nearest_dist'     = ydist,
                'distribution_table' = distri_table,
                'prob'               = max(as.integer(tab)/k),
                'yhat'               = major_vote)
  
  return(result)
}

################################################################################
### KNN accuracy o for test/train split
myknnacc = function(Xtrain, Ytrain, Xtest, Ytest, k, type=2, ztype="z-score", weight=NULL) {
  ntest = nrow(Xtest)
  myclass = c()
  myprobs = c()
  for (i in 1:ntest) {
    myknn = knn(Xtrain, Ytrain, Xtest[i,], k, type, ztype, weight)
    myclass = c(myclass, myknn$yhat)
    myprobs = c(myprobs, myknn$prob)
  }
  acc = mean(myclass == Ytest)
  
  result = list('yhat'     = myclass,
                'probs'    = myprobs,
                'accuracy' = acc)
  return(result)
}

