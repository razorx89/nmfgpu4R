/*
nmfgpu4R - R binding for the nmfgpu library

Copyright (C) 2015-2016  Sven Koitka (svenkoitka@fh-dortmund.de)

This file is part of nmfgpu4R.

nmfgpu4R is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

nmfgpu4R is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with nmfgpu4R.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "Helper.h"
#include "WrapperNMFGPU.h"

#include "nmfgpu.h"
#include <limits>
#include <Rcpp.h>
#include <string> 


void checkInterruptImpl(void*) {
  R_CheckUserInterrupt();
}

bool checkInterrupt() {
  return (R_ToplevelExec(checkInterruptImpl, NULL) == FALSE); 
}

// [[Rcpp::export]]
bool initializeAdapters(std::string nmfgpuRoot) {
  return initializeLibrary(std::move(nmfgpuRoot));
}

// [[Rcpp::export]]
void shutdownAdapters() {
  finalizeLibrary();
}

// [[Rcpp::export]]
bool cppChooseGpu(int index) {
  return g_funcChooseGpu(static_cast<unsigned>(index)) == nmfgpu::ResultType::Success;
}

// [[Rcpp::export]]
unsigned cppNumberOfGpu() {
  return g_funcGetNumberOfGpu();
}

// [[Rcpp::export]]
Rcpp::List cppInfoForGpuIndex(unsigned index) {
  auto result = Rcpp::List::create();
  
  nmfgpu::GpuInformation info;
  auto error = g_funcGetInfoForGpuIndex(index, &info);
  if(error == nmfgpu::ResultType::Success) {
    result["name"] = info.name;
    result["free.bytes"] = info.freeMemory;
    result["total.bytes"] = info.totalMemory;
  } else {
    Rcpp::stop("Failed to retrieve informations about cuda device");
  }
  
  return result;
}

namespace details {
  nmfgpu::ResultType compute(nmfgpu::NmfDescription<float>& context, nmfgpu::ISummary* summary) {
    return g_funcNmfComputeSingle(&context, summary);
  }
  
  nmfgpu::ResultType compute(nmfgpu::NmfDescription<double>& context, nmfgpu::ISummary* summary) {
    return g_funcNmfComputeDouble(&context, summary);
  }
}

template<typename NumericType>
bool executeAlgorithm(Rcpp::List& result, const std::string& algorithm, nmfgpu::NmfDescription<NumericType>& context, const Rcpp::List& parameters) {
  nmfgpu::ISummary* summary;
  g_funcNmfCreateSummary(&summary);
  
  std::vector<nmfgpu::Parameter> nmfgpuParams;

  if(algorithm == "mu") {
    context.algorithm = nmfgpu::NmfAlgorithm::Multiplicative;
  } else if(algorithm == "als") {
    context.algorithm = nmfgpu::NmfAlgorithm::ALS;
  } else if(algorithm == "gdcls") {
    if(parameters.containsElementNamed("lambda")) {
      auto lambda = Rcpp::as<double>(parameters["lambda"]);
      
      if(lambda < 0.0) {
        Rcpp::Rcerr << "[ERROR] Parameter 'lambda' for the algorithm GDCLS needs to be positive" << std::endl;
        return false;
      } else {
        context.algorithm = nmfgpu::NmfAlgorithm::GDCLS;
        nmfgpuParams.emplace_back(nmfgpu::Parameter{"lambda", lambda});
      }        
    } else {
      Rcpp::Rcerr << "[ERROR] Algorithm GDCLS needs the following parameters to be set: lambda" << std::endl;
      return false;
    }
  } else if(algorithm == "acls") {
    if(parameters.containsElementNamed("lambdaH") && parameters.containsElementNamed("lambdaW")) {
      auto lambdaH = Rcpp::as<double>(parameters["lambdaH"]);
      auto lambdaW = Rcpp::as<double>(parameters["lambdaW"]);
      
      if(lambdaH < 0.0 || lambdaW < 0.0) {
        Rcpp::Rcerr << "[ERROR] Parameters 'lambdaH' and 'lambdaW' for the algorithm ACLS need to be positive" << std::endl;
        return false;
      } else {
        context.algorithm = nmfgpu::NmfAlgorithm::ACLS;
        nmfgpuParams.emplace_back(nmfgpu::Parameter{"lambdaH", lambdaH});
        nmfgpuParams.emplace_back(nmfgpu::Parameter{"lambdaW", lambdaW});
      }
    } else {
      Rcpp::Rcerr << "[ERROR] Algorithm ACLS needs the following parameters to be set: lambdaH, lambdaW" << std::endl;
      return false;
    }
  } else if(algorithm == "ahcls") {
    if((parameters.containsElementNamed("lambdaH") && parameters.containsElementNamed("lambdaW")) ||
       (parameters.containsElementNamed("alphaH") && parameters.containsElementNamed("alphaW"))) {
      auto lambdaH = Rcpp::as<double>(parameters["lambdaH"]);
      auto lambdaW = Rcpp::as<double>(parameters["lambdaW"]);
      auto alphaH = Rcpp::as<double>(parameters["alphaH"]);
      auto alphaW = Rcpp::as<double>(parameters["alphaW"]);
      
      if(lambdaH < 0.0 || lambdaW < 0.0) {
        Rcpp::Rcerr << "[ERROR] Parameters 'lambdaH' and 'lambdaW' for the algorithm ACLS need to be positive" << std::endl;
        return false;
      } else if(alphaH < 0.0 || alphaH > 1.0 || alphaW < 0.0 || alphaW > 1.0) {
        Rcpp::Rcerr << "[ERROR] Parameters 'alphaH' and 'alphaW' for the algorithm ACLS need to be in the range of 0.0 and 1.0" << std::endl;
        return false;
      } else {
        context.algorithm = nmfgpu::NmfAlgorithm::AHCLS;
        nmfgpuParams.emplace_back(nmfgpu::Parameter{"lambdaH", lambdaH});
        nmfgpuParams.emplace_back(nmfgpu::Parameter{"lambdaW", lambdaW});
        nmfgpuParams.emplace_back(nmfgpu::Parameter{"alphaH", alphaH});
        nmfgpuParams.emplace_back(nmfgpu::Parameter{"alphaW", alphaW});
      }
    } else {
      Rcpp::Rcerr << "[ERROR] Algorithm AHCLS needs the following parameters to be set: lambdaH, lambdaW, alphaH, alphaW" << std::endl;
      return false;
    }
  } else if(algorithm == "nsNMF") {
    if(parameters.containsElementNamed("theta")) {
      auto theta = Rcpp::as<double>(parameters["theta"]);
      
      if(theta < 0.0 || theta > 1.0) {
        Rcpp::Rcerr << "[ERROR] Parameter 'theta' for the algorithm nsNMF needs to be in the range of 0.0 and 1.0" << std::endl;
        return false;
      }
      
      context.algorithm = nmfgpu::NmfAlgorithm::nsNMF;
      nmfgpuParams.emplace_back(nmfgpu::Parameter{"theta", theta});
    } else {
      Rcpp::Rcerr << "[ERROR] Algorithm nsNMF needs the following parameters to be set: theta" << std::endl;
    }
  } else {
    Rcpp::Rcerr << "[ERROR] Unknown algorithm specified!" << std::endl;
    return false;
  }
  
  context.parameters = nmfgpuParams.data();
  context.numParameters = nmfgpuParams.size();
  
  // Check result
  if(details::compute(context, summary) != nmfgpu::ResultType::Success) {
    Rcpp::Rcerr << "[ERROR] Failed to execute algorithm!" << std::endl;
    return false;
  }
  
  // Create result list
  nmfgpu::ExecutionRecord statistic;
  auto bestRun = summary->bestRun();
  summary->record(bestRun, statistic);
  result["Frobenius"] = statistic.frobenius;
  result["RMSD"] = statistic.rmsd;
  result["ElapsedTime"] = statistic.elapsedTime;
  result["NumIterations"] = statistic.numIterations;
  
  summary->destroy();
  
  return true;
}

template<typename NumericType>
bool parseRemainingInitializationMethods(nmfgpu::NmfDescription<NumericType>& context, const std::string& initMethod) {
    if(initMethod == "AllRandomValues") {
      context.initMethod = nmfgpu::NmfInitializationMethod::AllRandomValues;      
    } else if(initMethod == "MeanColumns") {
      context.initMethod = nmfgpu::NmfInitializationMethod::MeanColumns;
    } else if(initMethod == "K-Means/Random") {
      context.initMethod = nmfgpu::NmfInitializationMethod::KMeansAndRandomValues;
    } else if(initMethod == "K-Means/NonNegativeWTV") {
      context.initMethod = nmfgpu::NmfInitializationMethod::KMeansAndNonNegativeWTV;
    } else if(initMethod == "EIn-NMF") {
      context.initMethod = nmfgpu::NmfInitializationMethod::EInNMF;
    } else {
      Rcpp::Rcerr << "[ERROR] Unknown initialization method!" << std::endl;
      return false;
    }
    
    return true;
}

SEXP computeSinglePrecisionUnifiedDataMatrix(nmfgpu::NmfDescription<float>& context, const std::string& algorithm, const std::string& initMethod, 
                                             unsigned rows, unsigned columns, int features, int seed, double threshold, unsigned maxiter, 
                                             unsigned runs, Rcpp::List parameters, bool ssnmf) {
  context.seed = seed;
  context.numIterations = maxiter;
  context.numRuns = runs;
  context.thresholdType = nmfgpu::NmfThresholdType::Frobenius;
  context.thresholdValue = threshold;
  context.callbackUserInterrupt = &checkInterrupt;
  
  // Detect which initialization method should be applied
  std::vector<float> W;
  std::vector<float> H;
  if(initMethod == "CopyExisting") {
    if(parameters.containsElementNamed("W") && parameters.containsElementNamed("H")) {
      W = Rcpp::as<std::vector<float>>(parameters["W"]);
      H = Rcpp::as<std::vector<float>>(parameters["H"]);
      
      context.initMethod = nmfgpu::NmfInitializationMethod::CopyExisting;
    } else {
      Rcpp::Rcerr << "[ERROR] Initialization method 'CopyExisting' requires the matrices W and H to be set in the 'parameters' list!" << std::endl;
      return R_NilValue;
    }
  } else {
    W.resize(rows * features);
    H.resize(features * columns);
    if(!parseRemainingInitializationMethods(context, initMethod)) {
      return R_NilValue;
    }
  }
  
  if(ssnmf) {
    W = Rcpp::as<std::vector<float>>(parameters["W"]);
  }
  
  // Set matrices
  context.features = features;
  context.outputMatrixW.rows = rows;
  context.outputMatrixW.columns = features;
  context.outputMatrixW.format = nmfgpu::StorageFormat::Dense;
  context.outputMatrixW.dense.values = &W[0];
  context.outputMatrixW.dense.leadingDimension = rows;
  context.outputMatrixH.rows = features;
  context.outputMatrixH.columns = columns;
  context.outputMatrixH.format = nmfgpu::StorageFormat::Dense;
  context.outputMatrixH.dense.values = &H[0];
  context.outputMatrixH.dense.leadingDimension = features;
  
  // Execute the algorithm
  auto results = Rcpp::List();
  if(!executeAlgorithm<float>(results, algorithm, context, parameters))
    return R_NilValue;
  else {
    // Convert result back to double precision
    Rcpp::NumericVector Wd = Rcpp::wrap(W);
    Wd.attr("dim") = Rcpp::Dimension(rows, features);
    results["W"] = Wd;
    
    Rcpp::NumericVector Hd = Rcpp::wrap(H);
    Hd.attr("dim") = Rcpp::Dimension(features, columns);
    results["H"] = Hd;
    
    return results;
  }                                               
}
                                               
// [[Rcpp::export]]
SEXP adapterComputeSinglePrecision(const std::string& algorithm, const std::string& initMethod, Rcpp::NumericMatrix V, 
                                   int features, int seed, double threshold, unsigned maxiter, unsigned runs, Rcpp::List parameters, bool verbose, bool ssnmf) {
  g_funcNmfSetVerbosity(verbose ? nmfgpu::Verbosity::Summary : nmfgpu::Verbosity::None);
  
  nmfgpu::NmfDescription<float> context;
  
  context.useConstantBasisVectors = ssnmf;
  
  auto Vf = Rcpp::as<std::vector<float>>(V);
  context.inputMatrix.rows = V.nrow();
  context.inputMatrix.columns = V.ncol();
  context.inputMatrix.format = nmfgpu::StorageFormat::Dense;
  context.inputMatrix.dense.values = &Vf[0];
  context.inputMatrix.dense.leadingDimension = V.nrow();
  
  auto result = computeSinglePrecisionUnifiedDataMatrix(context, algorithm, initMethod, V.nrow(), V.ncol(), features, seed, threshold, maxiter, runs, parameters, ssnmf);
  
  return result;  
}

SEXP computeDoublePrecisionUnifiedDataMatrix(nmfgpu::NmfDescription<double>& context, const std::string& algorithm, const std::string& initMethod, 
                                             unsigned rows, unsigned columns, int features, int seed, double threshold, unsigned maxiter, 
                                             unsigned runs, Rcpp::List parameters, bool ssnmf) {
  
  context.seed = seed;
  context.numIterations = maxiter;
  context.numRuns = runs;
  context.thresholdType = nmfgpu::NmfThresholdType::Frobenius;
  context.thresholdValue = threshold;
  context.callbackUserInterrupt = &checkInterrupt;
  
  // Detect which initialization method should be applied
  Rcpp::NumericMatrix W;
  Rcpp::NumericMatrix H;
  if(initMethod == "CopyExisting") {
    if(parameters.containsElementNamed("W") && parameters.containsElementNamed("H")) {
      W = Rcpp::as<Rcpp::NumericMatrix>(parameters["W"]);
      H = Rcpp::as<Rcpp::NumericMatrix>(parameters["H"]);
      
      if(unsigned(W.nrow()) != rows || W.ncol() != features || H.nrow() != features || unsigned(H.ncol()) != columns) {
        Rcpp::Rcerr << "[ERROR] Initialization method 'CopyExisting' requires the matrices W and H to have the correct dimensions!" << std::endl;
        return R_NilValue;
      }
      
      context.initMethod = nmfgpu::NmfInitializationMethod::CopyExisting;
    } else {
      Rcpp::Rcerr << "[ERROR] Initialization method 'CopyExisting' requires the matrices W and H to be set in the 'parameters' list!" << std::endl;
      return R_NilValue;
    }
  } else {
    W = Rcpp::NumericMatrix(rows, features);
    H = Rcpp::NumericMatrix(features, columns);
    if(!parseRemainingInitializationMethods(context, initMethod)) {
      return R_NilValue;
    }
  }
  
  if(ssnmf) {
    W = Rcpp::as<Rcpp::NumericMatrix>(parameters["W"]);
  }
  
  // Set matrices
  context.features = features;
  context.outputMatrixW.rows = rows;
  context.outputMatrixW.columns = features;
  context.outputMatrixW.format = nmfgpu::StorageFormat::Dense;
  context.outputMatrixW.dense.values = &W[0];
  context.outputMatrixW.dense.leadingDimension = rows;
  context.outputMatrixH.rows = features;
  context.outputMatrixH.columns = columns;
  context.outputMatrixH.format = nmfgpu::StorageFormat::Dense;
  context.outputMatrixH.dense.values = &H[0];
  context.outputMatrixH.dense.leadingDimension = features;
  
  // Execute the algorithm
  auto results = Rcpp::List();
  if(!executeAlgorithm<double>(results, algorithm, context, parameters))
    return R_NilValue;
  else {
    results["W"] = W;
    results["H"] = H;
    
    return results;
  }
}

// [[Rcpp::export]]
SEXP adapterComputeDoublePrecision(const std::string& algorithm, const std::string& initMethod, Rcpp::NumericMatrix V, 
                                   int features, int seed, double threshold, unsigned maxiter, unsigned runs, Rcpp::List parameters, bool verbose, bool ssnmf) {
  g_funcNmfSetVerbosity(verbose ? nmfgpu::Verbosity::Summary : nmfgpu::Verbosity::None);
  
  nmfgpu::NmfDescription<double> context;
  
  context.useConstantBasisVectors = ssnmf;
  
  context.inputMatrix.rows = V.nrow();
  context.inputMatrix.columns = V.ncol();
  context.inputMatrix.format = nmfgpu::StorageFormat::Dense;
  context.inputMatrix.dense.values = &V[0];
  context.inputMatrix.dense.leadingDimension = V.nrow();
  
  auto result = computeDoublePrecisionUnifiedDataMatrix(context, algorithm, initMethod, V.nrow(), V.ncol(), features, seed, threshold, maxiter, runs, parameters, ssnmf);

  return result;
}

template<typename NumericType, typename StorageType>
bool convertAndSetSparseMatrix(nmfgpu::NmfDescription<NumericType>& context, const Rcpp::S4& sparseV, StorageType& values, unsigned& rows, unsigned& columns) {
  if(sparseV.is("matrix.csr")) {
    auto dim = Rcpp::IntegerVector(sparseV.slot("dimension"));
    values = Rcpp::as<StorageType>(sparseV.slot("ra"));
    auto csrRowPtr = Rcpp::IntegerVector(sparseV.slot("ia"));
    auto csrColInd = Rcpp::IntegerVector(sparseV.slot("ja"));
    rows = dim[0];
    columns = dim[1];
    context->setDataMatrixCSR(rows, columns, values.size(), &values[0], &csrRowPtr[0], &csrColInd[0], nmfgpu::IndexBase::One);
  } else if(sparseV.is("dgRMatrix")) {
    auto dim = Rcpp::IntegerVector(sparseV.slot("Dim"));
    values = Rcpp::as<StorageType>(sparseV.slot("x"));
    auto csrRowPtr = Rcpp::IntegerVector(sparseV.slot("p"));
    auto csrColInd = Rcpp::IntegerVector(sparseV.slot("j"));
    rows = dim[0];
    columns = dim[1];
    context->setDataMatrixCSR(rows, columns, values.size(), &values[0], &csrRowPtr[0], &csrColInd[0], nmfgpu::IndexBase::Zero);
  } else if(sparseV.is("matrix.csc")) {
    auto dim = Rcpp::IntegerVector(sparseV.slot("dimension"));
    values = Rcpp::as<StorageType>(sparseV.slot("ra"));
    auto cscColPtr = Rcpp::IntegerVector(sparseV.slot("ia"));
    auto cscRowInd = Rcpp::IntegerVector(sparseV.slot("ja"));
    rows = dim[0];
    columns = dim[1];
    context->setDataMatrixCSC(rows, columns, values.size(), &values[0], &cscColPtr[0], &cscRowInd[0], nmfgpu::IndexBase::One);
  } else if(sparseV.is("dgCMatrix")) {
    auto dim = Rcpp::IntegerVector(sparseV.slot("Dim"));
    values = Rcpp::as<StorageType>(sparseV.slot("x"));
    auto cscColPtr = Rcpp::IntegerVector(sparseV.slot("p"));
    auto cscRowInd = Rcpp::IntegerVector(sparseV.slot("i"));
    rows = dim[0];
    columns = dim[1];
    context->setDataMatrixCSC(rows, columns, values.size(), &values[0], &cscColPtr[0], &cscRowInd[0], nmfgpu::IndexBase::Zero);
  } else if(sparseV.is("matrix.coo")) {
    auto dim = Rcpp::IntegerVector(sparseV.slot("dimension"));
    values = Rcpp::as<StorageType>(sparseV.slot("ra"));
    auto cooRowInd = Rcpp::IntegerVector(sparseV.slot("ia"));
    auto cooColInd = Rcpp::IntegerVector(sparseV.slot("ja"));
    rows = dim[0];
    columns = dim[1];
    context->setDataMatrixCOO(rows, columns, values.size(), &values[0], &cooRowInd[0], &cooColInd[0], nmfgpu::IndexBase::One);
  } else if(sparseV.is("dgTMatrix")) {
    auto dim = Rcpp::IntegerVector(sparseV.slot("Dim"));
    values = Rcpp::as<StorageType>(sparseV.slot("x"));
    auto cooRowInd = Rcpp::IntegerVector(sparseV.slot("i"));
    auto cooColInd = Rcpp::IntegerVector(sparseV.slot("j"));
    rows = dim[0];
    columns = dim[1];
    context->setDataMatrixCOO(rows, columns, values.size(), &values[0], &cooRowInd[0], &cooColInd[0], nmfgpu::IndexBase::Zero);
  } else {
    Rcpp::Rcerr << "[ERROR] Unknown sparce matrix format!" << std::endl;
    return false;
  }
  
  return true;
}

// [[Rcpp::export]]
SEXP adapterComputeSinglePrecisionSparse(const std::string& algorithm, const std::string& initMethod, Rcpp::RObject sparseV, 
                                   int features, int seed, double threshold, unsigned maxiter, unsigned runs, Rcpp::List parameters, bool verbose, bool ssnmf) {
  g_funcNmfSetVerbosity(verbose ? nmfgpu::Verbosity::Summary : nmfgpu::Verbosity::None);
  
  nmfgpu::NmfDescription<float> context;
  
  context.useConstantBasisVectors = ssnmf;
  
  SEXP result = R_NilValue;
  
  std::unique_ptr<std::vector<float>> values;
  if(!Details::fillMatrixDescriptionFromRObject<float, std::vector<float>>(context.inputMatrix, values, sparseV)) {
    return R_NilValue;
  }
  
  result = computeSinglePrecisionUnifiedDataMatrix(context, algorithm, initMethod, context.inputMatrix.rows, context.inputMatrix.columns, features, seed, threshold, maxiter, runs, parameters, ssnmf);

  
  return result;
}

// [[Rcpp::export]]
SEXP adapterComputeDoublePrecisionSparse(const std::string& algorithm, const std::string& initMethod, Rcpp::RObject sparseV, 
                                   int features, int seed, double threshold, unsigned maxiter, unsigned runs, Rcpp::List parameters, bool verbose, bool ssnmf) {
  g_funcNmfSetVerbosity(verbose ? nmfgpu::Verbosity::Summary : nmfgpu::Verbosity::None);
  
  nmfgpu::NmfDescription<double> context;
  
  context.useConstantBasisVectors = ssnmf;
  
  SEXP result = R_NilValue;
  std::unique_ptr<Rcpp::NumericVector> values;
  if(!Details::fillMatrixDescriptionFromRObject<double, Rcpp::NumericVector>(context.inputMatrix, values, sparseV)) {
    return R_NilValue;
  }
  
  result = computeDoublePrecisionUnifiedDataMatrix(context, algorithm, initMethod, context.inputMatrix.rows, context.inputMatrix.columns, features, seed, threshold, maxiter, runs, parameters, ssnmf);
  
  return result;
}

// [[Rcpp::export]]
void adapterSetCallback(Rcpp::Function func) {
  
}