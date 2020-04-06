#' data - a matrix/data frame of attribute values
#' reps - number of replications of data set required
#' p_swap - probability parameter
#' local_var - local variance parameter

munge <- function(data, reps, p_swap = 0.5, local_var = 0.2){

  require(FNN)
  index <- as.numeric(get.knn(data, 1)$nn.index)
  needed <- nrow(data)*(reps + 1)
  
  d <- list()
  for(i in 1:(reps+1)){
    d[[i]] <- data
    for(j in 1:nrow(d[[i]])){
      nn <- data[index[j], ]
      for(k in 1:ncol(d[[i]])){
        tmp <- rnorm(1, nn[k], abs(nn[k] - d[[i]][j, k])/local_var)
        d[[i]][j, k] <- sample(c(tmp, d[[i]][j, k]), 1, prob = c(p_swap, (1 - p_swap)))
      }
    }
  }
  
  res <- unique(do.call(rbind, d))
  i <- 1
  while((nrow(res) < needed) & (i < 100)){
    d <- data
    for(j in 1:nrow(d)){
      nn <- data[index[j], ]
      for(k in 1:ncol(d)){
        tmp <- rnorm(1, nn[k], abs(nn[k] - d[j, k])/local_var)
        d[j, k] <- sample(c(tmp, d[j, k]), 1, prob = c(p_swap, (1 - p_swap)))
      }
    }
    i <- i + 1
    res <- unique(rbind(res, d))
  }
  
  res
}


# # maybe the result also do depend on dimensionality...
# # so try both local_var = 0.2 and local_var = 1 for PRIM
# 
# # normal data
# set.seed(11)
# data <- cbind(rnorm(200), rnorm(200))
# plot(data)
# points(munge(data, 50, local_var = 0.2), col = "red")
# points(munge(data, 50, local_var = 0.1), col = "green")
# 
# plot(data)
# points(munge(data, 50, local_var = 5), col = "orange")
# 
# 
# # donut data
# set.seed(11)
# r <- runif(1000) + 1
# g <- runif(1000, 0, 2*pi)
# data <- cbind(r*sin(g), r*cos(g))
# plot(data)
# points(munge(data, 10, local_var = 0.2), col = "red")
# points(munge(data, 10, local_var = 0.1), col = "green")
# 
# plot(data)
# points(munge(data, 10, local_var = 5), col = "orange")
# 
