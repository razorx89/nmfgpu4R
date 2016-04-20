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

#ifdef _WIN32
  #include <Windows.h>
  #undef ERROR 
#else
  #include <dlfcn.h>
#endif
#include <fstream>
#include <Rcpp.h>
#include <string>
#include "WrapperNMFGPU.h"

namespace Details {
  void* loadLibrary(const char* path) {
#ifdef _WIN32
    return LoadLibrary(path);
#else
    return dlopen(path, RTLD_NOW);
#endif
  } 
  
  void closeLibrary(void* handle) {
#ifdef _WIN32
    if(FALSE ==FreeLibrary(reinterpret_cast<HMODULE>(handle))) {
      Rcpp::Rcerr << "[ERROR] Failed to unload nmfgpu library!" << std::endl;
    }
#else
    dlclose(handle);
#endif
  }
  
  template<typename Ret, typename ...Args>
  void loadFunction(std::function<Ret(Args...)>& output, const char* name) {
#ifdef _WIN32
    auto ptr = reinterpret_cast<Ret(*)(Args...)>(GetProcAddress(reinterpret_cast<HMODULE>(g_libraryHandle), name));
#else
    auto ptr = reinterpret_cast<Ret(*)(Args...)>(dlsym(g_libraryHandle, name));
#endif
    
    if(ptr == nullptr) {
      output = nullptr;
    } else {
      output = [ptr](Args&&... args) -> Ret {
        return ptr(std::forward<Args>(args)...);
      };
    }
  }
  
  std::string buildLibraryPath(std::string nmfgpuLib) {
    
    if(nmfgpuLib.back() == '/'
       || nmfgpuLib.back() == '\\') {
      nmfgpuLib += "lib/";
    } else {
      nmfgpuLib += "/lib/";
    }
    
  #if __x86_64__
    const std::string platformSuffix = "64";
  #else
    const std::string platformSuffix = "32";
  #endif
    
    std::string libName, libExt;
  #if _WIN32
    libName = "nmfgpu";
    libExt = ".dll";
  #elif __linux__
    libName = "libnmfgpu";
    libExt = ".so";
  #elif __APPLE__ && __MACH__
    libName = "libnmfgpu";
    libExt = ".dylib";
  #else
    Rcpp::Rcerr << "[ERROR] Unrecognized platform (Supported: Windows, Linux, Mac OS X)!" << std::endl;
    return false;
  #endif
  
    nmfgpuLib += libName;
    nmfgpuLib += platformSuffix;
    nmfgpuLib += libExt;
    
    return std::move(nmfgpuLib);
  }
}

// Create variable instances
void* g_libraryHandle = nullptr;
std::function<nmfgpu::ResultType()>                                                 g_funcNmfInitialize;
std::function<nmfgpu::ResultType()>                                                 g_funcNmfFinalize;
std::function<int()>                                                                g_funcVersion;
std::function<nmfgpu::ResultType(nmfgpu::ISummary**)>                               g_funcNmfCreateSummary;
std::function<nmfgpu::ResultType(nmfgpu::NmfDescription<float>*, nmfgpu::ISummary*)>   g_funcNmfComputeSingle;
std::function<nmfgpu::ResultType(nmfgpu::NmfDescription<double>*, nmfgpu::ISummary*)>  g_funcNmfComputeDouble;
std::function<void(nmfgpu::Verbosity)>                                              g_funcNmfSetVerbosity;
std::function<nmfgpu::ResultType(nmfgpu::KMeansDescription<float>*)>                g_funcKMeansComputeSingle;
std::function<nmfgpu::ResultType(nmfgpu::KMeansDescription<double>*)>               g_funcKMeansComputeDouble;
std::function<nmfgpu::ResultType(unsigned)>                                         g_funcChooseGpu;
std::function<unsigned()>                                                           g_funcGetNumberOfGpu;
std::function<nmfgpu::ResultType(unsigned,nmfgpu::GpuInformation*)>                 g_funcGetInfoForGpuIndex;

bool initializeLibrary(std::string nmfgpuRoot) {
  // Build complete path to shared object
  auto nmfgpuLib = Details::buildLibraryPath(std::move(nmfgpuRoot));
  
  // Check if library exists
  if(!std::ifstream(nmfgpuLib).good()) {
    Rcpp::Rcerr << "[ERROR] Necessary library file '" << nmfgpuLib << "' not installed!" << std::endl;
    return false;
  }
  
  // Load shared library
  g_libraryHandle = Details::loadLibrary(nmfgpuLib.c_str());
  if(g_libraryHandle == nullptr) {
#if __APPLE__ && __MACH__
    Rcpp::Rcerr << dlerror() << std::endl;
#endif
    Rcpp::Rcerr << "[ERROR] Failed to load nmfgpu library!" << std::endl;
    return false;
  }
  
  Details::loadFunction(g_funcNmfInitialize, "nmfgpu_initialize");
  Details::loadFunction(g_funcNmfFinalize, "nmfgpu_finalize");
  Details::loadFunction(g_funcVersion, "nmfgpu_version");
  Details::loadFunction(g_funcNmfCreateSummary, "nmfgpu_create_summary");
  Details::loadFunction(g_funcNmfComputeSingle, "nmfgpu_compute_single");
  Details::loadFunction(g_funcNmfComputeDouble, "nmfgpu_compute_double");
  Details::loadFunction(g_funcNmfSetVerbosity, "nmfgpu_set_verbosity");
  Details::loadFunction(g_funcKMeansComputeSingle, "nmfgpu_compute_kmeans_single");
  Details::loadFunction(g_funcKMeansComputeDouble, "nmfgpu_compute_kmeans_double");
  Details::loadFunction(g_funcChooseGpu, "nmfgpu_choose_gpu");
  Details::loadFunction(g_funcGetNumberOfGpu, "nmfgpu_get_number_of_gpu");
  Details::loadFunction(g_funcGetInfoForGpuIndex, "nmfgpu_get_information_for_gpu_index");
  
  if(g_funcNmfInitialize == nullptr
    || g_funcNmfFinalize == nullptr
    || g_funcVersion == nullptr
    || g_funcNmfComputeSingle == nullptr
    || g_funcNmfComputeDouble == nullptr
    || g_funcNmfSetVerbosity == nullptr
    || g_funcKMeansComputeSingle == nullptr
    || g_funcKMeansComputeDouble == nullptr
    || g_funcChooseGpu == nullptr
    || g_funcGetNumberOfGpu == nullptr
    || g_funcGetInfoForGpuIndex == nullptr) {
    Rcpp::Rcerr << "[ERROR] Failed to load one or more function addresses from the nmfgpu library!" << std::endl; 
    return false;
  }
  
  if(g_funcVersion() != NMFGPU_VERSION) {
      Rcpp::Rcerr << "[ERROR] Installed nmfgpu library is incompatible!" << std::endl;
      return false;
  }
  
  if(nmfgpu::ResultType::Success != g_funcNmfInitialize()) {
      Rcpp::Rcerr << "[ERROR] Failed to startup nmfgpu library!" << std::endl;
      return false;
  }
  
  return true;
}

void finalizeLibrary() {  
  g_funcNmfFinalize();
  
  Details::closeLibrary(g_libraryHandle);
  g_libraryHandle = nullptr;
  g_funcNmfInitialize = nullptr;
  g_funcNmfFinalize = nullptr;
  g_funcNmfCreateSummary = nullptr;
  g_funcNmfComputeSingle = nullptr;
  g_funcNmfComputeDouble = nullptr;
  g_funcNmfSetVerbosity = nullptr;
  g_funcChooseGpu = nullptr;
  g_funcGetNumberOfGpu = nullptr;
  g_funcGetInfoForGpuIndex = nullptr;
}

#define xstr(s) str(s)
#define str(s) #s

std::string getVersionString() {
  return "v" xstr(NMFGPU_MAJOR) "." xstr(NMFGPU_MINOR) "." xstr(NMFGPU_PATCH);
}