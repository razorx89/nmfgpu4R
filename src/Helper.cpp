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