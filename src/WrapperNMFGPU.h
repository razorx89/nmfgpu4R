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

#include <functional>
#include "nmfgpu.h"
#include <string>

extern void* g_libraryHandle;
extern std::function<nmfgpu::ResultType()>                                             g_funcNmfInitialize;
extern std::function<nmfgpu::ResultType()>                                             g_funcNmfFinalize;
extern std::function<int()>                                                            g_funcVersion;
extern std::function<nmfgpu::ResultType(nmfgpu::ISummary**)>                           g_funcNmfCreateSummary;
extern std::function<nmfgpu::ResultType(nmfgpu::NmfDescription<float>*, nmfgpu::ISummary*)>  g_funcNmfComputeSingle;
extern std::function<nmfgpu::ResultType(nmfgpu::NmfDescription<double>*, nmfgpu::ISummary*)> g_funcNmfComputeDouble;
extern std::function<void(nmfgpu::Verbosity)>                                          g_funcNmfSetVerbosity;
extern std::function<nmfgpu::ResultType(nmfgpu::KMeansDescription<float>*)>            g_funcKMeansComputeSingle;
extern std::function<nmfgpu::ResultType(nmfgpu::KMeansDescription<double>*)>           g_funcKMeansComputeDouble;
extern std::function<nmfgpu::ResultType(unsigned)>                                     g_funcChooseGpu;
extern std::function<unsigned()>                                                       g_funcGetNumberOfGpu;
extern std::function<nmfgpu::ResultType(unsigned,nmfgpu::GpuInformation*)>                     g_funcGetInfoForGpuIndex;

bool initializeLibrary(std::string nmfgpuRoot);
void finalizeLibrary();

// [[Rcpp::export]]
std::string getVersionString();