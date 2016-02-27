#include "Helper.h"

namespace Details {
  void fillDenseMatrixDescriptionFromRObject(nmfgpu::MatrixDescription<float>& desc, std::unique_ptr<std::vector<float>>& values, Rcpp::RObject input) {
    // Dense matrix from NumericMatrix
    auto matrix                 = Rcpp::NumericMatrix(input);
    values                      = make_unique<std::vector<float>>(Rcpp::as<std::vector<float>>(matrix));
    //values                      = make_unique<ValueStorageType>(valuesConverted);
    //values                      = convertMatrixValues<ValueStorageType>(matrix);
    desc.format                 = nmfgpu::StorageFormat::Dense;
    desc.rows                   = matrix.nrow();
    desc.columns                = matrix.ncol();
    desc.dense.values           = &(*values)[0];
    desc.dense.leadingDimension = matrix.nrow();
  }

  void fillDenseMatrixDescriptionFromRObject(nmfgpu::MatrixDescription<double>& desc, std::unique_ptr<Rcpp::NumericVector>& values, Rcpp::RObject input) {
    // Dense matrix from NumericMatrix
    auto matrix                 = Rcpp::NumericMatrix(input);
    desc.format                 = nmfgpu::StorageFormat::Dense;
    desc.rows                   = matrix.nrow();
    desc.columns                = matrix.ncol();
    desc.dense.values           = &matrix[0];
    desc.dense.leadingDimension = matrix.nrow();
  }
}