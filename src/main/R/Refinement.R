# as compared to the latest version, I allow the box to be 
# expanded and made default depth equal 20

#' dx - a matrix/data frame of attribute values
#' dy - a vector with 0 and 1
#' beam.size is the parameter for the beam search
#' depth is the maximal number of restricted dimensions

beam.refine <- function(dx, dy, beam.size = 1, depth = 20){
  
  #### functions ####
  
  require(data.table)
  setDTthreads(threads = 1)
  
  # local function to assess WRAcc quality metric
  wracc <- function(n, np, N, Np){
    (n/N)*(np/n - Np/N)
  }
  
  # refine a single dimension of the box
  refine <- function(dx, dy, box, ind, start.q){
    
    N <- length(dy)
    Np <- sum(dy)
    
    ind.in.box <- rep(TRUE, N)                    
    for(i in 1:ncol(dx)){
      if(!(i == ind)){
        ind.in.box <- ind.in.box & (dx[, i] >= box[1, i] & dx[, i] <= box[2, i])
      }
    }
    in.box <- cbind(dx[ind.in.box, ind], dy[ind.in.box])
    in.box <- data.table(in.box[order(in.box[,1]),])
    colnames(in.box) <- c("x", "y")
    
    t.m <- h.m <- -1000000                        # 3-4
    l <- box[1, ind]                              # 1
    r <- box[2, ind]                              # 1
    n <- in.box[,.N]
    np <- in.box[, sum(y)]
    wracc.m = start.q                             # 2
    # start.q <- wracc.m <- wracc(n, np, N, Np)     # 2 
    
    t <- sort(unique(in.box[, x]))                # define T
    for(i in 2:length(t)){                        # 5
      tmp <- in.box[in.box[, x == t[i - 1]]]      
      n <- n - tmp[,.N]                           # 6
      np <- np - tmp[, sum(y)]                    # 6
      h <- wracc(n, np, N, Np)                    # 7
      if(h > h.m){                                # 8
        h.m <- h                                  # 9
        t.m <- t[i]                               # 10
      }
      tmp <- in.box[in.box[, x >= t.m & x <= t[i]]]
      n.i <- tmp[,.N]
      np.i <- tmp[, sum(y)]
      wracc.i <- wracc(n.i, np.i, N, Np)
      if(wracc.i > wracc.m){                      # 11
        l <- t.m                                  # 12
        r <- t[i]                                 # 12
        wracc.m <- wracc.i                        # 13
      }
    }
    
    box[, ind] <- c(l, r)
    list(box, wracc.m, ifelse(wracc.m == start.q, 0, 1)) # the last value 0 indicates that the box is a dead end
  }
  
  #### end functions ####
  

  if((min(dy) < 0) | (max(dy) > 1)){
   warning("The target variable takes values from outside [0,1]. Just making sure you are aware of it")
  }

  dim <- ncol(dx)
  if(depth > dim){
    warning("Restricting depth parameter to the number of atributes in data!")
    depth <- dim
  }
  
  box.init <- matrix(c(apply(dx, 2, min), apply(dx, 2, max)), ncol = dim, byrow = TRUE)
  dims <- 1:ncol(box.init)
  res.box <- list()
  res.tab <- as.data.frame(matrix(ncol = 3, nrow = 0))
  
  for(i in 1:ncol(box.init)){
    tmp <- refine(dx, dy, box.init, i, 0)
    res.box <- c(res.box, list(tmp[[1]]))
    res.tab <- rbind(res.tab, c(tmp[[2]], tmp[[3]], i))
  }
  
  if(depth > 1){
    for(j in 1:(depth - 1)){
      if(nrow(res.tab) > beam.size){
        retain <- which(res.tab[, 1] >= sort(res.tab[, 1], decreasing = TRUE)[beam.size])
        res.tab <- res.tab[retain, ]
        res.box <- res.box[retain]
      }
      for(k in 1:nrow(res.tab)){
        if(res.tab[k, 2] == 1){
          res.tab[k, 2] <- 0
          inds.r <- dims[!(dims %in% res.tab[k, 3])]
          for(i in inds.r){
            tmp <- refine(dx, dy, res.box[[k]], i, res.tab[k, 1])
            res.box <- c(res.box, list(tmp[[1]]))
            res.tab <- rbind(res.tab, c(tmp[[2]], tmp[[3]], i))
          }
        }
      }
    }
  }
  
  winner <- which(res.tab[, 1] == max(res.tab[, 1]))[1]
  res <- res.box[[winner]]
  #res[1, res[1, ] == box.init[1, ]] <- -10^8
  #res[2, res[2, ] == box.init[2, ]] <- 10^8
  res
}



#### TEST ####
# 
# full.data <- read.table("banknote.txt", sep = ",")
# for(i in 1:4){
# full.data[, i] <- (full.data[, i] - min(full.data[, i]))/(max(full.data[, i]) - min(full.data[, i]))
# }
# selector <- (1:nrow(full.data))%%2
# 
# tmpx <- full.data[selector == 1, 1:4]
# tmpy <- full.data[selector == 1, 5]
# 
# a <- Sys.time()
# beam.refine(tmpx, tmpy, 4, 4)
# a <- Sys.time() - a # 25 s
# 
# tmpx <- full.data[selector == 0, 1:4]
# tmpy <- full.data[selector == 0, 5]
# 
# b <- Sys.time()
# beam.refine(tmpx, tmpy, 4, 4)
# b <- Sys.time() - b
# 
# tmpx <- full.data[, 1:4]
# tmpy <- full.data[, 5]
# 
# d <- Sys.time()
# beam.refine(tmpx, tmpy, 4, 4)
# d <- Sys.time() - d
# 
# e <- Sys.time()
# beam.refine(tmpx, tmpy, 1, 4)
# e <- Sys.time() - e
# 
# d - a - b
# d
# e

# So the boxes obtained on subsamples are quite different

#### experiment with avila

# full.data <- read.table("C:\\Projects\\6_PRIM_RF_real\\Implementation\\resources\\data\\avila\\avila.txt", sep = ",")
# for(i in 1:10){
# # the row below ensures that each value is unique
#  full.data[, i] <- full.data[, i] + runif(length(full.data[, i]), min = 0, max = 10^-8)
#  full.data[, i] <- (full.data[, i] - min(full.data[, i]))/(max(full.data[, i]) - min(full.data[, i]))
#  print(length(unique(full.data[, i])))
# }
# selector <- (1:nrow(full.data))%%4
# 
# tmpx <- full.data[selector == 1, 1:10]
# tmpy <- full.data[selector == 1, 11]
# tmpy <- ifelse(tmpy == "A", 1, 0)
# 
# a <- Sys.time()
# beam.refine(tmpx, tmpy)
# Sys.time() - a

#### SAAC2 experiment

# d <- read.csv("C:\\Projects\\6_PRIM_RF_real\\gr_prim\\resources\\data\\tmp_saac2.csv", header = TRUE)
# d <- d[, -1]
# dx <- d[, -1]
# dy <- d[, 1]
# a <- Sys.time()
# beam.refine(dx, dy, 1, 10)
# Sys.time() - a 
