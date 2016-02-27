#' K-means Clustering on GPU
#' 
#' Performs k-means clustering on a data matrix using the GPU.
#'
#' @param data A numeric data matrix or compatible type with attributes in columns and data samples in rows.
#' 
#' @param k Number of clusters to compute for the data matrix.
#' 
#' @param seed A seed used to initialize the random number generators for initialization of the cluster centers.
#' 
#' @param maxiter Maximum number of iterations until the algorithm execution is aborted.
#' 
#' @param threshold If the percentage change of cluster memberships is lower than the \code{threshold}, then the algorithm
#' converges (default is 5\% of data samples). 
#' 
#' @param useSinglePrecision Algorithm can be exeucted either using single precision or double precision floating point values.
#' Especially when you are not using High Performance Computing (HPC) devices such as Nvidia Tesla series it might lead to 
#' significant performance improvements if single precision computation is enabled. Most graphic devices only have limited support 
#' for double precision floating point operations.
#' 
#' @param object Object of class "\code{kmeansgpu}"
#' 
#' @param ... Unused arguments
#' 
#' @return Returns a list of class "\code{kmeansgpu}" which has a \code{print} and a \code{fitted} method. The list 
#' contains the following information:
#' 
#' \tabular{ll}{
#'  \code{cluster} \tab Integer vector of length \code{1:nrow(data)} with cluster assignments in the range of \code{1:k}. \cr
#'  
#'  \code{centers} \tab Numeric matrix containing \code{k} cluster centers computed by the k-Means algorithm. \cr
#'  
#'  \code{size} \tab Integer vector of length \code{k} containing the number of assigned data samples for each cluster. \cr
#'  
#'  \code{betweenss} \tab Numeric value containing the between-cluster sum of squares. \cr
#'  
#'  \code{withinss} \tab Numeric vector of length \code{k} containing the within-cluster sum of squares for each cluster. \cr
#'  
#'  \code{tot.withinss} \tab Numeric value containg the sum of all \code{withinss} values. \cr
#'  
#'  \code{totss} \tab Numeric value containing the sum of \code{betweenss} and \code{tot.within}.
#' }
#' 
#' @rdname kmeansgpu
#' @export kmeansgpu
kmeansgpu <- function(data, k, seed=floor(runif(1, 0, .Machine$integer.max)), maxiter=100, threshold=0.05, useSinglePrecision=F) {
  # If data is a data.frame then try to convert it to a numeric matrix
  if(is.data.frame(data)) {
    if(all(sapply(data, is.numeric))) {
      data <- as.matrix(data)
    } else {
      stop("Data frame does contain non-numeric variables")
    }
  }
  
  # Validate values
  validateMatrix(data, is.nonneg=F)
  
  # Algorithm expects features to be rows
  transposed.data <- transpose.generic(data)
  
  # Execute algorithm
  if(useSinglePrecision) {
    result <- adapterComputeKMeansSingle(transposed.data, k, seed, maxiter, threshold)
  } else {
    result <- adapterComputeKMeansDouble(transposed.data, k, seed, maxiter, threshold)
  }
  
  if(!is.null(result)) {
    # Transpose clusters as they are in Row-Major format
    result$centers <- t(result$centers)
    
    # Membership indices are 0-based but must be converted to be 1-based
    result$cluster <- result$cluster + 1
    
    # Assign names
    colnames(result$centers) <- colnames(data)
    rownames(result$centers) <- paste("c", 1:k, sep="")
    
    # Count number of assignments per cluster
    result$size <- as.integer(table(result$cluster))
    
    # Compute between cluster sum of squares
    data.mean <- colSums.generic(data) / nrow(data)
    result$betweenss <- 0.0
    for(clusterIndex in 1:k) {
      if(result$size[clusterIndex] > 0) {
        diff <- result$centers[clusterIndex,] - data.mean
        between <- result$size[clusterIndex] * (diff %*% diff)
        result$betweenss <- result$betweenss + between
      }
    }
    
    #Compute within cluster sum of squares
    result$withinss <- numeric(length=k)
    for(clusterIndex in 1:k) {
      clusterAssignments <- result$cluster == clusterIndex
      if(any(clusterAssignments)) {
        result$withinss[clusterIndex] <- sum(sapply(which(clusterAssignments), function(index) {
          diff <- rowval.generic(data, index) - result$centers[clusterIndex,]
          return(diff %*% diff)
        }))
      }
    }
    result$tot.withinss = sum(result$withinss)
    
    
    # Compute total sum of squares
    result$totss <- result$tot.withinss + result$betweenss
    
    # Set S4 class
    class(result) <- "kmeansgpu"
  }
  
  return(result)
}

#' @rdname kmeansgpu
#' @method print kmeansgpu
#' @export
print.kmeansgpu <- function(object, ...) {
  cat("K-means clustering on GPU with", nrow(object$centers), "clusters of sizes", paste(object$size, sep=", "), "\n\n")
  cat("Cluster means:\n")
  print(object$centers)
  cat("\nClustering vector:\n")
  print(object$cluster)
  cat("\nWithin cluster sum of squared by cluster:\n")
  print(object$withinss)
  cat(" (betweenss / totss =", format(object$betweenss / object$totss * 100.0, digits=2, nsmall=2), "%)\n")
  cat("\nAvailable components:\n")
  print(names(object))
}

#' @rdname kmeansgpu
#' @method fitted kmeansgpu
#' @export
fitted.kmeansgpu <- function(object, ...) {
  fitted.centers <- object$centers[object$cluster,]
  rownames(fitted.centers) <- object$cluster
  return(fitted.centers)
}