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