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

#pragma once

#include <memory>
#include "nmfgpu.h"
#include <Rcpp.h>

#ifdef __cpp_lib_make_unique
using std::make_unique;
#else
namespace Details {
  /** Implementation of a C++14 make_unique function, as it is not proposed by the C++11 standard.
  @tparam T Type of the object to be constructed.
  @tparam Args List of type names for the argument list.
  @param args List of argument values which will be forwarded to the constructor of @p T.
  @returns Returns a constructed and initialized unique_ptr of type @p T
  @see http://herbsutter.com/gotw/_102/ */
  template<typename T, typename... Args>
  std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
  }
}
#endif

namespace Details {
  
  void fillDenseMatrixDescriptionFromRObject(nmfgpu::MatrixDescription<float>& desc, std::unique_ptr<std::vector<float>>& values, Rcpp::RObject input);
  
  void fillDenseMatrixDescriptionFromRObject(nmfgpu::MatrixDescription<double>& desc, std::unique_ptr<Rcpp::NumericVector>& values, Rcpp::RObject input);
  
  template<typename NumericType, typename ValueStorageType>
  bool fillMatrixDescriptionFromRObject(nmfgpu::MatrixDescription<NumericType>& desc, std::unique_ptr<ValueStorageType>& values, Rcpp::RObject input) {
    if(input.isS4()) {
      auto sparseV = Rcpp::as<Rcpp::S4>(input);
      
      // Set dimension
      if(sparseV.is("matrix.csr") || sparseV.is("matrix.csc") || sparseV.is("matrix.coo")) {
        auto dim = Rcpp::IntegerVector(sparseV.slot("dimension"));
        desc.rows     = dim[0];
        desc.columns  = dim[1];
      } else if(sparseV.is("dgRMatrix") || sparseV.is("dgCMatrix") || sparseV.is("dgTMatrix")) {
        auto dim      = Rcpp::IntegerVector(sparseV.slot("Dim"));
        desc.rows     = dim[0];
        desc.columns  = dim[1];
      } else {
        Rcpp::Rcerr << "[ERROR] Unknown sparse matrix format!" << std::endl;
        return false;
      }
      
      // Set values
      if(sparseV.is("matrix.csr")) {
        values                  = make_unique<ValueStorageType>(Rcpp::as<ValueStorageType>(sparseV.slot("ra")));
        desc.format             = nmfgpu::StorageFormat::CSR;
        desc.csr.values         = &(*values)[0];
        desc.csr.nnz            = values->size();
        desc.csr.rowPtr         = &Rcpp::IntegerVector(sparseV.slot("ia"))[0];
        desc.csr.columnIndices  = &Rcpp::IntegerVector(sparseV.slot("ja"))[0];
        desc.csr.base           = nmfgpu::IndexBase::One;
      } else if(sparseV.is("dgRMatrix")) {
        values                  = make_unique<ValueStorageType>(Rcpp::as<ValueStorageType>(sparseV.slot("x")));
        desc.format             = nmfgpu::StorageFormat::CSR;
        desc.csr.values         = &(*values)[0];
        desc.csr.nnz            = values->size();
        desc.csr.rowPtr         = &Rcpp::IntegerVector(sparseV.slot("p"))[0];
        desc.csr.columnIndices  = &Rcpp::IntegerVector(sparseV.slot("j"))[0];
        desc.csr.base           = nmfgpu::IndexBase::Zero;
      } else if(sparseV.is("matrix.csc")) {
        values              = make_unique<ValueStorageType>(Rcpp::as<ValueStorageType>(sparseV.slot("ra")));
        desc.format         = nmfgpu::StorageFormat::CSC;
        desc.csc.values     = &(*values)[0];
        desc.csc.nnz        = values->size();
        desc.csc.columnPtr  = &Rcpp::IntegerVector(sparseV.slot("ia"))[0];
        desc.csc.rowIndices = &Rcpp::IntegerVector(sparseV.slot("ja"))[0];
        desc.csc.base       = nmfgpu::IndexBase::One;
      } else if(sparseV.is("dgCMatrix")) {
        values              = make_unique<ValueStorageType>(Rcpp::as<ValueStorageType>(sparseV.slot("x")));
        desc.format         = nmfgpu::StorageFormat::CSC;
        desc.csc.values     = &(*values)[0];
        desc.csc.nnz        = values->size();
        desc.csc.columnPtr  = &Rcpp::IntegerVector(sparseV.slot("p"))[0];
        desc.csc.rowIndices = &Rcpp::IntegerVector(sparseV.slot("i"))[0];
        desc.csc.base       = nmfgpu::IndexBase::Zero;
      } else if(sparseV.is("matrix.coo")) {
        values                  = make_unique<ValueStorageType>(Rcpp::as<ValueStorageType>(sparseV.slot("ra")));
        desc.format             = nmfgpu::StorageFormat::COO;
        desc.coo.values         = &(*values)[0];
        desc.coo.nnz            = values->size();
        desc.coo.rowIndices     = &Rcpp::IntegerVector(sparseV.slot("ia"))[0];
        desc.coo.columnIndices  = &Rcpp::IntegerVector(sparseV.slot("ja"))[0];
        desc.coo.base           = nmfgpu::IndexBase::One;
      } else if(sparseV.is("dgTMatrix")) {
        values                  = make_unique<ValueStorageType>(Rcpp::as<ValueStorageType>(sparseV.slot("x")));
        desc.format             = nmfgpu::StorageFormat::COO;
        desc.coo.values         = &(*values)[0];
        desc.coo.nnz            = values->size();
        desc.coo.rowIndices     = &Rcpp::IntegerVector(sparseV.slot("i"))[0];
        desc.coo.columnIndices  = &Rcpp::IntegerVector(sparseV.slot("j"))[0];
        desc.coo.base           = nmfgpu::IndexBase::Zero;
      }
      
      return true;
    }
    
    fillDenseMatrixDescriptionFromRObject(desc, values, input);
    
    return true;
  }
}
