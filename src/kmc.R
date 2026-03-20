################################################################################
# K-Means Clustering (KMC)
kmc = function(X, k, init='Forgy', iter='Lloyd', seed) {
  # X is n x p matrix with n observation and p features
  # k is the number of clusters we want
  set.seed(seed)
  n = nrow(X)
  p = ncol(X)
  
  # check whether k here is a valid parameter
  if(k < 1 || k > n) {
    stop("invalid k value: k should be = or < than the sample size n")
  }
  
  # ----------------- Initialization of Centroids ------------------------
  if(init == 'Forgy') {
    # choose k random existing points as initial centers
    centroids = X[sample.int(n, k, replace=F), , drop=F]
  } else if (init == 'Bounding-Box') {
    # find the bounding box for the entire dataset and uniformly random select
    # smaple k centroids inside the box
    mins = apply(X, 2, min)
    maxs = apply(X, 2, max)
    centroids = matrix(NA, nrow = k, ncol = p)
    for(j in 1:p) {
      centroids[,j] = runif(k, mins[j], maxs[j])
    }
  } else if (init == 'Random-Partition') {
    # randomly assign labels (1, 2, ..., k) to all points, then take the mean 
    # of each label as the initial centroids
    init = sample.int(k, n, replace=T)
    centroids = matrix(NA, nrow = k, ncol = p)
    for (j in 1:k) {
      idx = (init == j)
      centroids[j,] = colMeans(X[idx, , drop=F])
    }
  } else {
    stop("Unknown initialization metric. Use 'Forgy', 'Bounding-Box', or 'Random-Partition'.")
  }
  
  initial_centroids = centroids
  
  # -------------------------- Iterations ---------------------------------
  # initialize the clusters vector, which should contains only labels
  if (iter == 'Lloyd') {
    clusters = rep(0, n)
    iter = 0
    repeat {
      iter = iter + 1
      dist = matrix(0, nrow = n, ncol = k)
  
      # compute the 2-norm distance between each points and the k centers
      for (j in 1:k) {
        diff = X - matrix(centroids[j, ], nrow = n, ncol = p, byrow=T)
        dist[, j] = sqrt(rowSums(diff^2))
      }
      
      # assign cluster label to every other points
      new_clusters = apply(dist, 1, which.min)
      change = sum(new_clusters != clusters)
      clusters = new_clusters
      
      # move centers to the mean of assigned points
      for (j in 1:k) {
        idx = (clusters == j)
        if (any(idx)) {
          centroids[j, ] = colMeans(X[idx, , drop=F])
        }
      }
      
      # convergence check: labels unchanged
      if (change == 0) break
    }
  } else if (iter == 'Mac-Queen') {
    clusters = rep(0, n)
    prev_clusters = rep(0, n)
    iter = 0
    repeat {
      iter = iter + 1
      counts = rep(0, k)
      prev_clusters = clusters
      
      for (i in 1:n) {
        # distance from Xi to each center
        diff = centroids - matrix(X[i,], nrow = k, ncol = p, byrow=T)
        dist = sqrt(rowSums(diff^2))
        
        # nearest cluster; j here is the temporary centroids
        j = which.min(dist)
        clusters[i] = j
        
        # Mac-Queen incremental updates of centroids
        counts[j] = counts[j] + 1
        centroids[j,] = centroids[j,] + (X[i,] - centroids[j,])/counts[j]
      }
      
      if (sum(prev_clusters != clusters) == 0) break
    }
  } else if (iter == 'hierarchy') {
    clusters = rep(0, n)
    iter = 1
    for(i in 1:n) {
      diff = centroids - matrix(X[i,], nrow = k, ncol = p, byrow=T)
      dist = sqrt(rowSums(diff^2))
      clusters[i] = which.min(dist)
    }
  } else {
    stop("Unknown iteration metric. Use 'Lloyd', 'Mac-Queen', or 'hierarchy'.")
  }
  
  # ------------------------- Output and Result -----------------------------
  # build a distribution table: how many points in each cluster
  tab = table(clusters)
  nj = as.numeric(tab)
  
  Xdf = data.frame(X)
  Xs = split(Xdf, clusters)
  
  SSC = numeric(k)
  for (j in 1:k) {
    idx = (clusters == j)
    if (any(idx)) {
      Xj = X[idx, , drop=F]
      diff = Xj - matrix(centroids[j, ], nrow=nrow(Xj), ncol=p, byrow=T)
      SSC[j] = sum(rowSums(diff^2))
    }
  }
  TSSC = sum(SSC)
  
  nearest_cluster = function(myx) {
    diff = centroids - matrix(myx, nrow = k, ncol = p, byrow=T)
    dist = sqrt(rowSums(diff^2))
    return(which.min(dist))
  }
  
  result = list('iteration_num'     = iter,
                'initial_centroids' = initial_centroids,
                'final_centroids'   = centroids,
                'cids'              = clusters,
                'nj'                = nj,
                'SSC'               = SSC,
                'aSSC'              = SSC/nj,
                'TSSC'              = TSSC,
                'aTSSC'             = TSSC/sum(nj),
                'nearest_cids'      = nearest_cluster)
  return(result)
}

