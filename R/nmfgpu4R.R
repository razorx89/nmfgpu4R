#' R binding for computing non-negative matrix factorizations using CUDA
#' 
#' R binding for the libary \emph{libnmfgpu} which can be used to compute non-negative matrix factorizations using CUDA hardware
#' acceleration. 
#' 
#' The main function to use is \code{\link{nmfgpu}} which can be configured using various arguments.
#' In addition to it a few helper functions are provided, but they aren't necessary for using \code{\link{nmfgpu}}.
#' 
#' @docType package
#' @name nmfgpu4R
#' @useDynLib nmfgpu4R
#' @import Rcpp
NULL

# Loads the C++ library when the package is loaded 
.onLoad <- function(lib, pkg) {
  nmfgpuRoot <- Sys.getenv("NMFGPU_ROOT")
  if(nmfgpuRoot == "") {
    if(dir.exists("/usr/local/nmfgpu")) {
      nmfgpuRoot <- "/usr/local/nmfgpu"
    } else if(dir.exists("C:/Program Files/nmfgpu")) {
      nmfgpuRoot <- "C:/Program Files/nmfgpu"
    }
    stop("[ERROR] Environment variable NMFGPU_ROOT not set!")
  }
  if(!dir.exists(nmfgpuRoot)) {
    stop("[ERROR] Environment variable NMFGPU_ROOT points to '", nmfgpuRoot, "' which is not a valid and/or existing directory!")
  }
  if(!initializeAdapters(nmfgpuRoot)) {
    stop("[ERROR] Initialization failed!");
  }
}

# Unloads the C++ library when the package is unloaded
.onUnload <- function(lib, pkg) {
  shutdownAdapters()
}

# Sets a callback which gets called when a new frobenius norm has been calculated. Information like iteration number, frobenius norm, 
# RMSD and the delta frobenius norm are provided.
# 
# @param callbackFunction Either a valid R function to process data during algorithm execution or the NULL object to unset any callback.
#
# @note The callback function need to accept the following parameters: iteration, frobenius, deltaFrobenius, rmsd.
# 
# @export
#nmfgpu.setCallback <- function(callbackFunction) {
#  return(adapterSetCallback(callbackFunction))
#}


#' Requests the currently available and total amount of device memory.
#' 
#' @param device.index If specified the memory info retrieval is restricted to the passed device indices. By default no restriction is active and
#' therefore memory information about all available CUDA devices are retrieved.
#' 
#' @export
cudaMemoryInfo <- function(device.index=NA) {
  if(is.na(device.index)) {
    device.index = 0:(getNumberOfGpu()-1)
  }
  
  result <- list()
  for(i in device.index) {
    tmp <- getInfoForGpuIndex(i)
    tmp$index = i
    result <- c(result, list(tmp))
  }
  
  class(result) <- "cudameminfo"
  
  return(result)
}

library(utils)

#' Prints the information of a 'cudameminfo' object.
#' @param x Object of class 'cudameminfo'
#' @param ... Other arguments
#' @export
print.cudameminfo <- function(x, ...) {
  for(i in 1:length(x)) {
    device.info <- x[[i]]
    
    # Try to format bytes
    if(requireNamespace("gdata", quietly=T)) {
      used <- gdata::humanReadable(device.info$total.bytes - device.info$free.bytes, digits=2)
      total <- gdata::humanReadable(device.info$total.bytes, digits=2)
    } else {
      used <- paste(device.info$total.bytes - device.info$free.bytes, "B")
      total <- paste(device.info$total.bytes, "B")
    }
    
    cat("#", device.info$index, ": ", device.info$name, " [ Allocated: ", used, " / ", total, " ]\n", sep="")
    pb <- txtProgressBar(initial=(device.info$total.bytes - device.info$free.bytes) / device.info$total.bytes, style=3)
  }
}

#' Retrieves the total number of installed CUDA devices. 
#' @export
gpuCount <- function() {
  return(getNumberOfGpu())
}

#' Selects the specified device as primary computation device. All further invocations to nmfgpu will use the specified
#' CUDA device. 
#' 
#' @param device.index Index of the CUDA device, which should be used for computation.
#'  
#' @export
chooseGpu <- function(device.index) {
  if(!is.numeric(device.index) || device.index %% 1 != 0 || device.index < 0) {
    stop("device.index must be a non-negative integer number")
  }
  
  if(!chooseGpuImpl(device.index)) {
    stop("Failed to choose specified gpu!")
  }
}