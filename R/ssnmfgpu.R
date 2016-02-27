#' @rdname ssnmfgpu
#' @export
ssnmfgpu <- function(...) {
  UseMethod("ssnmfgpu")
}

#' Semi-supervised Non-negative Matrix Factorization (SSNMF) on GPU
#' 
#' test
#' \deqn{V_{train}=W_{train}H_{train}}{V.train=W.train*H.train}
#' \deqn{V_{test}=W_{train}H_{test}}{V.test=W.train*H.test}
#' 
#' @param data Test
#' @param r Test
#' @param ...
#' 
#' @return If the factorization process was successful then a list of class "\code{ssnmfgpu}" is returned which has a \code{predict} method. The list 
#' contains the following information:
#' 
#' \tabular{ll}{
#'  \code{basis.train} \tab Numeric matrix with \code{r} basis vectors from the training dataset. These basis vectors will
#'  be used to compute a mixing matrix to encode new data matrices. (For ease of further computations this matrix is stored in transposed form) \cr
#'  \code{encoding.train} \
#' }
#' 
#' @rdname ssnmfgpu
#' @method ssnmfgpu default
#' @export
ssnmfgpu.default <- function(data, r, ...) {
  # Compute regular NMF
  result <- nmfgpu.default(transpose.generic(data), r, ...)
  
  if(!is.null(result)) {
    # Build ssnmfgpu data structure
    result.ssnmfgpu <- list()
    result.ssnmfgpu$basis.train <- result$W
    result.ssnmfgpu$encoding.train <- t(result$H)
    result.ssnmfgpu$nmf.features <- r
    result.ssnmfgpu$nmf.args <- list(...)
    result.ssnmfgpu$input.features <- ncol(data)
    
    # Set S4 class name
    class(result.ssnmfgpu) <- "ssnmfgpu"
  }
  
  return(result.ssnmfgpu)
}

#' @rdname ssnmfgpu
#' @method ssnmfgpu formula
#' @export
ssnmfgpu.formula <- function(formula, data, ...) {
  #if(!all(rownames(data.train) == rownames(data.test))) {
  #  stop("Both data matrices must have the same attribute names!")
  #}
  
  labels <- validateFormulaAndGetLabels(formula, data)
  
  result <- ssnmfgpu.default(data[labels,], ...)
  result$feature.labels <- labels
}

#' @param object Object of class "\code{ssnmfgpu}"
#' @param newdata New data matrix compatible to the training data matrix, for computing the corresponding mixing matrix.
#' 
#' @rdname ssnmfgpu
#' @method predict ssnmfgpu
#' @export
predict.ssnmfgpu <- function(object, newdata, ...) {
  # If no new data is provided then return the encoding matrix of the training data
  if(missing(newdata)) {
    return(object$encoding.train)
  }
  
  # If a formula was provided to the training data, then restrict newdata to the same features
  if("feature.labels" %in% names(object)) {
    if(!all(object$feature.labels %in% colnames(newdata))) {
      stop("Data matrix does not contain the same features as the training data!")
    } else {
      newdata <- newdata[,object$feature.labels]
    }
  }
  
  # Check if new data has the same amount of features as the training data
  if(is.matrix(newdata) || is.matrix.SparseM(newdata) || is.matrix.Matrix(newdata)) {
    if(ncol(newdata) != object$input.features) {
      stop("New data matrix does not have the same amount of features as the training data!")
    }
  }
  
  # Adjust original arguments to support SSNMF
  args <- object$nmf.args
  args$initMethod <- "AllRandomValues"
  if("parameters" %in% names(args)) {
    args$parameters$W <- object$basis.train
  } else {
    args$parameters <- list(W=object$basis.train)
  }
  
  # Compute the NMF for the new data using the basis vectors from the training data
  args$data <- transpose.generic(newdata)
  args$r <- object$nmf.features
  args$runs <- 1
  args$ssnmf <- T
  
  result <- do.call(nmfgpu.default, args)
  
  if(!is.null(result)) {
    return(t(result$H))
  } else {
    return(NULL)
  }
}