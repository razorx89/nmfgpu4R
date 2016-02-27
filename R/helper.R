is.matrix.SparseM <- function(matrix) {
  return(isS4(matrix) && matrix@class %in% c("matrix.csr", "matrix.csc", "matrix.coo"))
}

is.matrix.Matrix <- function(matrix) {
  return(isS4(matrix) && matrix@class %in% c("dgRMatrix", "dgCMatrix", "dgTMatrix", "dgeMatrix"))
}

transpose.generic <- function(matrix) {
  if(is.matrix.SparseM(matrix)) {
    ret <- SparseM::t(matrix)
  } else if(is.matrix.Matrix(matrix)) {
    ret <- Matrix::t(matrix)
  } else {
    ret <- t(matrix)
  }
  return(ret)
}

rowval.generic <- function(matrix, index) {
  if(is.matrix.SparseM(matrix)) {
    ret <- as.vector(SparseM::as.matrix(matrix[index,]))
  } else if(is.matrix.Matrix(matrix)) {
    ret <- Matrix::as.vector(matrix[index,])
  } else {
    ret <- matrix[index,]
  }
  
  return(ret)
}

colSums.generic <- function(matrix) {
  if(is.matrix.SparseM(matrix)) {
    if(matrix@class %in% c("matrix.csr", "matrix.coo")) {
      # CSR/COO format
      ret <- sapply(1:ncol(matrix), function(column) {
        return(sum(matrix@ra[matrix@ja == column]))
      })
    } else {
      # CSC format
      ret <- sapply(1:ncol(matrix), function(column) {
        return(sum(matrix@ra[matrix@ia[column]:matrix@ia[column+1]]))
      })
    }
  } else if(is.matrix.Matrix(matrix)) {
    ret <- Matrix::colSums(matrix)
  } else {
    ret <- colSums(matrix)
  }
  
  return(ret)
}

validateMatrix <- function(matrix, is.nonneg=T) {
  # Extract numeric values if a S4 matrix object was passed
  if(isS4(matrix)) {
    if(is.matrix.SparseM(matrix)) {
      values <- matrix@ja
    } else if(is.matrix.Matrix(matrix)) {
      values <- matrix@x
    } else {
      stop("Unknown S4 class as data matrix!")
    }
  } else if(is.numeric(matrix)) {
    values <- matrix
  } else {
    stop("Unknown matrix format")
  }
  
  if(is.nonneg && any(values < 0)) {
    stop("Data matrix contains negative data, which is not allowed for a Non-negative Matrix Factorization!");
  }
  
  if(any(is.na(values))) {
    stop("Data matrix contains NA values, which cannot be processed!");
  }
}

validateFormulaAndGetLabels <- function(formula, data) {
  if(!inherits(formula, "formula")) {
    stop("A formula must be provided as first argument!")
  }
  
  termsResult <- terms(formula, data=t(data))
  response <- attr(termsResult, "response")
  
  if(response != 0) {
    stop("Invalid formula, the NMF does not support any response!")
  }
  
  labels <- attr(termsResult, "term.labels")
  
  if(length(labels) == 0) {
    stop("Formula yields to zero attributes!")
  }
  
  return(labels)
}