#include <memory>
#include <nmfgpu.h>
#include <Rcpp.h>
#include "Helper.h"
#include "WrapperNMFGPU.h"

// [[Rcpp::export]]
SEXP adapterComputeKMeansSingle(SEXP pData, unsigned k, unsigned seed, unsigned maxiter, double threshold) {
  // Convert data matrix to single precision
  nmfgpu::KMeansDescription<float> desc;
  
  std::unique_ptr<std::vector<float>> values;
  if(!Details::fillMatrixDescriptionFromRObject<float, std::vector<float>>(desc.inputMatrix, values, pData)) {
    return R_NilValue;
  } 
  
  auto matClusters = std::vector<float>(desc.inputMatrix.rows * k);
  desc.outputMatrixClusters.format = nmfgpu::StorageFormat::Dense;
  desc.outputMatrixClusters.rows = desc.inputMatrix.rows;
  desc.outputMatrixClusters.columns = k;
  desc.outputMatrixClusters.dense.values = &matClusters[0];
  desc.outputMatrixClusters.dense.leadingDimension = desc.inputMatrix.rows;
  
  desc.numClusters = k;
  desc.seed = seed;
  desc.numIterations = maxiter;
  desc.thresholdValue = threshold;
  auto membershipIndices = std::vector<unsigned>(desc.inputMatrix.columns);
  desc.outputMemberships = membershipIndices.data();
  
  auto error = g_funcKMeansComputeSingle(&desc);
  if(error != nmfgpu::ResultType::Success) {
    return R_NilValue;
  }
  
  auto result = Rcpp::List();
  result["cluster"] = Rcpp::wrap(membershipIndices);
  
  Rcpp::NumericVector clusters = Rcpp::wrap(matClusters);
  clusters.attr("dim") = Rcpp::Dimension(desc.inputMatrix.rows, desc.numClusters);
  result["centers"] = clusters;
  
  return result;
  return R_NilValue;
}

// [[Rcpp::export]]
SEXP adapterComputeKMeansDouble(SEXP pData, unsigned k, unsigned seed, unsigned maxiter, double threshold) {
  nmfgpu::KMeansDescription<double> desc;

  std::unique_ptr<Rcpp::NumericVector> values;
  if(!Details::fillMatrixDescriptionFromRObject<double, Rcpp::NumericVector>(desc.inputMatrix, values, pData)) {
    return R_NilValue;
  }
  
  auto matClusters = Rcpp::NumericMatrix(desc.inputMatrix.rows, k);
  desc.outputMatrixClusters.format = nmfgpu::StorageFormat::Dense;
  desc.outputMatrixClusters.rows = desc.inputMatrix.rows;
  desc.outputMatrixClusters.columns = k;
  desc.outputMatrixClusters.dense.values = &matClusters[0];
  desc.outputMatrixClusters.dense.leadingDimension = desc.inputMatrix.rows;
  
  desc.numClusters = k;
  desc.seed = seed;
  desc.numIterations = maxiter;
  desc.thresholdValue = threshold;
  
  auto membershipIndices = std::vector<unsigned>(desc.inputMatrix.columns);
  desc.outputMemberships = membershipIndices.data();
  
  auto error = g_funcKMeansComputeDouble(&desc);
  if(error != nmfgpu::ResultType::Success) {
    return R_NilValue;
  }
  
  auto result = Rcpp::List();
  result["cluster"] = Rcpp::wrap(membershipIndices);
  result["centers"] = matClusters;
  
  return result;
}