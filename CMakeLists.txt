# LLMServe - lean, headless, CUDA-only build of the Strata engine for Qwen3.8-Flash-Next (GGUF arch qwen4exp).
#
# Targets: `strata` (the engine, also the `--serve` loop the Python server drives), `libstrata_kq` (K-quant MMQ prompt
# kernels, dlopen()ed from the exe's folder), plus the small CLIs strata-device / strata-gguf / strata-dequant / strata-plan.
# Intel (SYCL), Metal, AMD (HIP/gfx906), DeepSeek-V4.1 and Qwen3.6 (qwen35moe) support and the test/bench programs of the
# upstream tree are not part of this fork.
#
# The kernels need compute capability 7.5 or newer (the guard below refuses anything older): silently building for an older
# arch would produce a binary that runs and is wrong about its own performance.  Use CUDA 13 for sm_120 (RTX 50).
cmake_minimum_required(VERSION 3.24)
project(strata VERSION 0.1.51 LANGUAGES CXX)
add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}")
# issue #31: release builds keep a PDB beside the exe (not shipped in it), so a minidump from a stalled engine can
# be read back to the exact source line
if(MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zi>)
  add_link_options(/DEBUG /OPT:REF /OPT:ICF)
endif()

# Portable build (what the Docker image uses): ggml's CPU backend for any AVX2 CPU
# instead of this machine's (Strata's own AVX-512 kernels are per-file and chosen at run time either way), and
# the MSVC runtime linked in, so no Visual C++ redistributable is needed.
option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" OFF)
# EXPERIMENTAL, opt-in: an engine for CPUs without AVX2 (#394, #595, #623), compiled on that PC.  ggml-cpu, which then
# computes every CPU expert row, is built for this floor instead of AVX2 or the build host: "avx" (Sandy/Ivy Bridge:
# AVX, no FMA/F16C/AVX2) or "none" (SSE4.2 + POPCNT: Nehalem/Westmere).  Strata's own AVX2/AVX-512 kernels stay in the
# binary behind their CPU checks; the router lookahead gets an AVX1 kernel.  Empty (the default) changes nothing.
set(STRATA_ISA_FLOOR "" CACHE STRING "experimental older-CPU build: empty (default, AVX2 floor), avx, or none (SSE4.2)")
set_property(CACHE STRATA_ISA_FLOOR PROPERTY STRINGS "" avx none)
if(NOT STRATA_ISA_FLOOR STREQUAL "" AND NOT STRATA_ISA_FLOOR STREQUAL "avx" AND NOT STRATA_ISA_FLOOR STREQUAL "none")
  message(FATAL_ERROR "STRATA_ISA_FLOOR must be empty, avx or none (got '${STRATA_ISA_FLOOR}')")
endif()
if(STRATA_ISA_FLOOR AND STRATA_PORTABLE)
  message(FATAL_ERROR "STRATA_ISA_FLOOR (an older-CPU build) and STRATA_PORTABLE (the AVX2 release build) exclude each other")
endif()
if(STRATA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

# #3202: Windows hands a program its command line in the system's ANSI code page (cp1251 / cp866 on a Russian Windows, cp1252 on an
# English one), but the engine reads every path as UTF-8 (the run configs and the server write UTF-8). A model folder with
# Cyrillic letters arrived as other bytes, or as "?" where the page has no such letter, and the model did not open. This manifest
# makes the process's ANSI code page UTF-8 (Windows 10 1903 and newer; older Windows ignores it and nothing changes), so the
# command line, fopen and std::ifstream all take UTF-8.
function(strata_utf8_manifest target)
  if(MSVC)
    target_sources(${target} PRIVATE ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cmake/strata-utf8.manifest)
  endif()
endfunction()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(STRATA_ENABLE_CUDA "Build the CUDA targets (this fork is CUDA-only)" ON)
# MMQ prompt kernels for Q4_K / Q5_K / Q5_1 / Q6_K (Unsloth's UD-Q4/Q5/Q6_K_XL experts).  They are their own library
# (strata_kq, loaded only for a model with these formats: the engine loads every kernel of its own binary at start,
# CUDA_MODULE_LOADING=EAGER, and these took ~3.8 MB of VRAM - 3 IQ3_XXS expert slots on a P100 - from every model).  Used at
# run time with STRATA_MMQ_KQ=1 (opt-in: it changes the prompt's rounding against the FP16 path).
option(STRATA_MMQ_TQ_OVERLAY "Gyro trellis MMQ prompt kernels on an upstream llama.cpp: an overlay of its ggml-cuda headers written at configure time (cmake/strata_ggml_tq_overlay.cmake)" ON)
option(STRATA_MMQ_KQUANTS "Build the MMQ prompt kernels (libstrata_kq) for Q4_K / Q5_K / Q5_1 / Q6_K: the experts of Unsloth's K-quant files" ON)
# Q6_K gate/up in the grouped expert kernels (Unsloth's UD-Q6_K_XL, and layer 2 of UD-Q5_K_XL; docs/QUANTS.md).  On by
# default: one more instance per grouped kernel in iq_kernels.cu, measured to leave every expert-cache slot count as it was.  -DSTRATA_Q6K_EXPERTS=OFF drops it.
option(STRATA_Q6K_EXPERTS "Build the Q6_K gate/up grouped expert kernels (UD-Q6_K_XL, UD-Q5_K_XL)" ON)
if(STRATA_Q6K_EXPERTS)
  add_compile_definitions(STRATA_Q6K_EXPERTS=1)
endif()
option(STRATA_ORCA_Q4KS_MMQ "Build the GGML Q5_0 MMQ instance for OrcaRouter Q4_K_S" OFF)
# CUDA: the pointer-table MMQ behind STRATA_MMQ_DIRECT=1 (the prompt path's MMQ reads resident experts in their cache
# slots through a Strata-side copy of llama.cpp's MMQ scheduler, src/prefill/moe_mmq_tab.cuh).  Off by default, like
# the K-quant options above: the engine loads every kernel when it starts (CUDA_MODULE_LOADING=EAGER), and these nine
# instances (+15 MB of binary) take 36-38 MiB of VRAM per GPU on sm_86 - expert slots every model would lose, with
# STRATA_MMQ_DIRECT unset too.  They also add ~70% of the upstream MMQ instances' compile time per CUDA arch (175 s
# of CPU for the nine at sm_86 alone).  Off, the build's device code is byte for byte the one without this feature.  CUDA
# only; switched off at configure time when the llama.cpp tree's mmq.cuh is not one the copy
# was verified against.
option(STRATA_MMQ_TAB "Build the pointer-table prompt MMQ for STRATA_MMQ_DIRECT=1 (CUDA; ~37 MiB VRAM per GPU)" OFF)
option(STRATA_WERROR "Treat compiler warnings as errors" OFF)
# defined here, before the GPU section reads it
option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)

if(STRATA_ENABLE_CUDA)
  # #585: CMake links CUDA code with the STATIC runtime by default, and the targets below also link CUDA::cudart (the
  # DLL's import library).  MSVC with a CUDA 12.x toolkit - the experimental Pascal/Volta build - then fails with
  # LNK1169 (every cudart symbol defined twice); GNU ld lets it pass.  That build links the shared runtime alone.  The
  # ready-made engine (CUDA 13, no STRATA_EXPERIMENTAL_SM60) is unchanged; -DCMAKE_CUDA_RUNTIME_LIBRARY overrides.
  if(MSVC AND STRATA_EXPERIMENTAL_SM60 AND NOT DEFINED CMAKE_CUDA_RUNTIME_LIBRARY)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  endif()
  enable_language(CUDA)
  # `enable_language(CUDA)` gives the .cu compile rules but NOT the imported targets, so `CUDA::cudart` has to
  # be brought in explicitly for the host-side tests that call cudaMalloc themselves.
  find_package(CUDAToolkit REQUIRED)
  option(STRATA_EXPERIMENTAL_SM60 "community build for GPUs below compute capability 7.5: Pascal sm_60, Volta sm_70 (#236)" OFF)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures" FORCE)
  endif()
  foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX REPLACE "-.*$" "" _base "${_a}")     # 120-real / 120-virtual are still 120
    # Turing port: developed and measured on sm_120 (RTX 50); the tf32 mma in the QSA scorer has a
    # portable fp32-FMA fallback, so RTX 20 (75), 30 (86), RTX 40 (89) and RTX 50 (120) build too.
    # Pre-sm_75 is refused.
    if(_base STREQUAL "native" OR _base STREQUAL "all" OR _base STREQUAL "all-major")
      continue()
    endif()
    # STRATA_EXPERIMENTAL_SM60: a Pascal (sm_60, GP100-class) or Volta (sm_70, V100 / Titan V, #236) build.
    # These sit below every architecture the kernels were written for, so two intrinsics are replaced below
    # their arch - `__dp4a` (6.1+) and `__nanosleep` (7.0+) - in include/strata/kernels/dp4a.hpp, which is
    # bit-exact for the first and merely a shorter pause for the second (sm_70 has both).  Community-tested,
    # not in the ready-made engine; RTX 20/30/40/50 cards must use the release engine instead.  It needs a CUDA
    # 12.x toolkit: CUDA 13 dropped sm_60 and sm_70 (nvcc fails CMake's compiler test with "compute_70").
    if(_base LESS 75 AND NOT STRATA_EXPERIMENTAL_SM60)
      message(FATAL_ERROR
        "Strata needs an NVIDIA GPU of compute capability 7.5 or newer (RTX 20 / 30 / 40 / 50 series), but "
        "CMAKE_CUDA_ARCHITECTURES is '${CMAKE_CUDA_ARCHITECTURES}'.  (A Pascal sm_60 or Volta sm_70 build is "
        "-DSTRATA_EXPERIMENTAL_SM60=ON; upstream does not support it.)")
    endif()
    if(_base LESS 60)
      message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES '${CMAKE_CUDA_ARCHITECTURES}': even the experimental build "
                          "needs compute capability 6.0 (sm_60) or newer")
    endif()
    # #220 / #224: RTX 50 engines built with CUDA 12.8 crashed in the prompt path on Linux (13.0 and newer did not)
    if(_base GREATER_EQUAL 120 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
      message(WARNING "Building for sm_${_base} (RTX 50) with CUDA ${CMAKE_CUDA_COMPILER_VERSION}: engines built with "
                      "CUDA 12.8 crashed on long prompts on these cards (issues #220, #224). Use CUDA 13.0 or newer.")
    endif()
  endforeach()
  if(STRATA_EXPERIMENTAL_SM60)
    # every target sees it: the Pascal / Volta code paths (the Volta prompt attention, PR #600) are compiled into this
    # build only, so the ready-made engine (sm_75+) keeps exactly the code it had
    add_compile_definitions(STRATA_EXPERIMENTAL_SM60=1)
  endif()
  message(STATUS "Strata: CUDA enabled, arch ${CMAKE_CUDA_ARCHITECTURES}")
  set(_strata_gpu_runtime_target CUDA::cudart)
  set(_strata_gpu_blas_target CUDA::cublas)
  set(_strata_gpu_include_directories ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

# Warnings live on an INTERFACE target the project's own code links, NOT in add_compile_options: applied
# globally they would also hit FetchContent'd Catch2, whose -W4 warnings are not this project's to fix and
# whose /WX failures would be indistinguishable from our own.
#
# Every flag is guarded by COMPILE_LANGUAGE:CXX.  Without that guard nvcc receives bare `/W4 /utf-8` and dies
# with "A single input file is required for a non-link phase" - it parses them as inputs rather than host
# flags.  Host flags for CUDA must go through -Xcompiler, which is what COMPILE_LANGUAGE:CUDA does below.
add_library(strata_warnings INTERFACE)
target_compile_options(strata_warnings INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wextra>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/W4>)
if(STRATA_WERROR)
  target_compile_options(strata_warnings INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/WX>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Werror>)
endif()

# ---- strata_artifact -------------------------------------------------------
# Header-only, and that is a deliberate property rather than an unfinished split: the GGUF reader and the
# scalar dequantizers hold no state worth hiding, and a header-only boundary cannot introduce a
# duplicate-symbol or missing-declaration bug into code that is already validated against the real artifact.
add_library(strata_artifact INTERFACE)
target_include_directories(strata_artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_artifact INTERFACE strata_warnings)

add_executable(strata-gguf src/artifact/gguf_reader.cpp)
target_link_libraries(strata-gguf PRIVATE strata_artifact)

add_executable(strata-dequant src/artifact/dequant.cpp)
target_link_libraries(strata-dequant PRIVATE strata_artifact)
# this file is now nothing but the CLI entry point, and its inherited guard spells that as a "selftest"
target_compile_definitions(strata-dequant PRIVATE STRATA_DEQUANT_SELFTEST)

# ---- strata_plan ----------------------------------------------------------
# Also header-only.  A "planner" that hides state would be one whose decision is not inspectable, and the
# whole point of P1.S9 is that the decision is arithmetic you can check by hand.
add_library(strata_plan INTERFACE)
target_include_directories(strata_plan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_plan INTERFACE strata_warnings)

add_executable(strata-plan src/plan/plan_main.cpp)
target_link_libraries(strata-plan PRIVATE strata_plan)

# ---- strata_core (GPU runtime) ---------------------------------------------
# Only built with -DSTRATA_ENABLE_CUDA=ON (the default), so the header-only targets above still configure without a toolkit.
if(STRATA_ENABLE_CUDA)
  add_library(strata_core STATIC src/core/device.cu src/core/pinned.cu src/platform/memory.cpp src/core/graph.cpp src/core/weights.cpp
                               src/core/layout.cpp)
  target_include_directories(strata_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # the DFlash2 drafter (src/core/dflash2.cpp, src/kernels/cuda/dflash2.cu)
  target_compile_definitions(strata_core PUBLIC STRATA_HAVE_DFLASH2=1)
  if(STRATA_ENABLE_CUDA AND STRATA_EXPERIMENTAL_SM60)
    # the runtime device check (src/core/device.cu) admits the cards this build was made for (#236)
    target_compile_definitions(strata_core PRIVATE STRATA_EXPERIMENTAL_SM60=1)
  endif()
  # The selected runtime is PUBLIC because graph.cpp is a HOST translation unit that calls cudaGraphLaunch and
  # cudaEventQuery: the .cu sources get cudart through the CUDA language rules, a .cpp does not, and the
  # linker error ("unresolved external symbol cudaGraphInstantiate") names the function rather than the cause.
  target_link_libraries(strata_core PUBLIC strata_plan strata_warnings ${_strata_gpu_runtime_target})

  add_executable(strata-device src/core/device_main.cpp)
  target_link_libraries(strata-device PRIVATE strata_core)

  # ---- strata_kernels (P2.S2) ----
  # The kernels link `strata_artifact` for the scalar dequantizers.
  add_library(strata_kernels STATIC src/kernels/cuda/dequant_s2.cu src/kernels/cuda/s2_gemv.cu
                          src/kernels/cuda/s_gemv.cu
                          src/kernels/cuda/s_gemv_pair.cu
                          src/kernels/cuda/s2_gemv_quads.cu
                          src/kernels/cuda/s2_gemv_fast.cu
                          src/kernels/cuda/router_top10.cu
                          src/kernels/cuda/quantize_act.cu
                          src/kernels/cuda/rope.cu
                          src/kernels/cuda/sampler.cu
                          src/kernels/cuda/s2_gemv_q8.cu
    src/kernels/cuda/s2_expert_grouped.cu
                          src/kernels/cuda/shared_expert.cu
                          src/kernels/cuda/gr.cu
                          src/kernels/cuda/gdn.cu
                          src/kernels/cuda/qsa.cu
                          src/kernels/cuda/elementwise.cu
                          src/kernels/cuda/bf16_gemv.cu
                          src/kernels/cuda/native_mmvq.cu
                          src/kernels/cuda/native_bf16.cu
                          src/kernels/cuda/native_gr_norm.cu
                          src/kernels/cuda/native_moe.cu
                          src/kernels/cuda/native_gdn.cu
                          src/kernels/cuda/native_gr_postops.cu
                          src/kernels/cuda/native_router.cu
                          src/kernels/cuda/route_prior.cu
                          src/kernels/cuda/native_gdn_preprocess.cu
                          src/kernels/cuda/native_qsa.cu
                          src/kernels/cuda/native_rope.cu
                          src/kernels/cuda/native_ple_postops.cu
                          src/kernels/cuda/native_flash_attn.cu
                          src/kernels/cuda/native_qsa_indexer.cu
                          src/kernels/cuda/native_qsa_score.cu
                          src/kernels/cuda/ple.cu
                          src/kernels/cuda/kv_q8.cu
                          src/kernels/cuda/kv_stream.cu
                          src/kernels/cuda/kv_q4.cu
                          src/kernels/cuda/fused_gr.cu
                          src/kernels/cuda/cvec.cu
                          src/kernels/cuda/qsa_decode_attn.cu
                          src/kernels/cuda/qsa_prompt_attn.cu
                          src/kernels/cuda/fused_gdn.cu
                          src/kernels/cuda/dequant_bf16.cu
                          src/kernels/cuda/qsa_select.cu
                          src/kernels/cuda/verify_kernels.cu
                          src/kernels/cuda/dflash2.cu
                          src/kernels/cuda/iq_kernels.cu
                          src/kernels/ngram.cpp
                          src/ngram/ple_reader.cpp
                          src/platform/direct_file.cpp)
  target_link_libraries(strata_kernels PUBLIC strata_core strata_artifact)
  # plan v0.3 P6: llama.cpp's block layouts and codebook grids for the i-quant formats (MIT, third_party/ggml)
  target_include_directories(strata_kernels PRIVATE third_party/ggml)
  # Match the pinned llama.cpp CUDA arithmetic contract for this adapter only.
  if(STRATA_ENABLE_CUDA)
    set_source_files_properties(src/kernels/cuda/native_mmvq.cu src/kernels/cuda/native_bf16.cu
      src/kernels/cuda/native_gr_norm.cu
      src/kernels/cuda/native_moe.cu
      src/kernels/cuda/native_gdn.cu
      src/kernels/cuda/native_gr_postops.cu
      src/kernels/cuda/native_router.cu
      src/kernels/cuda/native_gdn_preprocess.cu
      src/kernels/cuda/native_qsa.cu
      src/kernels/cuda/native_rope.cu
      src/kernels/cuda/native_ple_postops.cu
      src/kernels/cuda/native_flash_attn.cu
      src/kernels/cuda/native_qsa_indexer.cu
      src/kernels/cuda/native_qsa_score.cu
      PROPERTIES COMPILE_OPTIONS "--use_fast_math")
  endif()

  # ---- strata_engine (P2.S5) -----------------------------------------------------
  # The COMPOSED layer lives here and not in `strata_core`, because it calls kernels and `strata_kernels`
  # already links `strata_core` - putting `gdn_layer` in core would make the two mutually dependent.  A third
  # library is the honest shape of the dependency graph: core knows about devices and weights, kernels know
  # about arithmetic, and this one knows the ORDER the arithmetic happens in.
  add_library(strata_engine STATIC src/core/layer.cpp src/core/session.cpp src/core/expert_source.cpp src/core/remote_experts.cpp
    src/core/remote_expert_opt.cu
    src/core/expert_cache.cpp src/core/native_head.cpp src/core/native_dense.cpp src/core/verify.cpp src/core/mtp.cpp src/core/dflash2.cpp
    src/core/conversation_snapshot.cpp src/core/conversation_state.cpp src/core/conversation_memory.cpp src/core/peer_experts.cpp
    src/core/arch_defaults.cpp src/core/nv_defaults.cpp src/core/vmm.cpp src/core/conversation_file.cpp
    src/core/conversation_pool.cpp)
  target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})

  # `strata generate` (P2.S6).  The FIRST user-facing program: everything under it is a component, and this is
  # what composes them into a token.  No tokenizer - the prompt arrives as ids - because Gate C1 compares logits
  # against llama.cpp on the SAME ids and a tokenizer on one side of that comparison is a second variable.
  # plan v0.3 P5: batched prompt processing (tensor-core GEMMs through cuBLAS).
  add_library(strata_prefill STATIC src/prefill/gemm.cu src/prefill/kernels.cu src/prefill/prefill.cpp)
  target_include_directories(strata_prefill PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
  target_link_libraries(strata_prefill PUBLIC strata_engine ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  if(STRATA_ENABLE_CUDA)
    # v100/gdn-chunk: the chunked Volta GDN prompt recurrence.  The dispatch (src/prefill/kernels.cu) is
    # Volta-only (cc 7.0) and gated on STRATA_EXPERIMENTAL_SM60; the path itself stays opt-in (STRATA_GDN_CHUNK).
    target_sources(strata_prefill PRIVATE src/prefill/gdn_chunk.cu)
  endif()
  add_executable(strata src/program/generate.cpp)
  strata_utf8_manifest(strata)
  target_include_directories(strata PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(strata PRIVATE strata_engine strata_prefill strata_spec ${_strata_gpu_runtime_target})
endif()

# ---- strata_kernels_cpu (P2.S3) ----
#
# A SEPARATE LIBRARY from `strata_kernels`, and `/arch:AVX512` is set on the SOURCE rather than on the target
# so that the flag applies only to the TU that needs it.  Both parts matter:
#
#   * the CPU expert kernel uses `_mm256_dpbusd_epi32` (AVX512-VNNI) and `_mm256_multishift_epi64_epi8`
#     (AVX512-VBMI); without the flag MSVC reports undeclared intrinsics and never mentions AVX-512;
#   * a TU compiled WITH `/arch:AVX512` may use AVX-512 in ANY of its code, so the scalar fallback and the
#     test drivers must NOT be in it - otherwise the fallback is not a fallback, it is the same bet with
#     worse opt-out.  `cpu_require_expert_support()` therefore refuses at startup on a CPU without the
#     features, and the fallback exists for tests, exactly as P2.S3 specifies.
# Plan v0.3 P6: speculation components that need no GPU.
add_library(strata_spec STATIC src/spec/suffix_drafter.cpp src/spec/controller.cpp src/spec/draft_policy.cpp
            src/spec/draft_source.cpp)
target_include_directories(strata_spec PUBLIC include)

add_library(strata_kernels_cpu STATIC src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp
            src/kernels/cpu/expert_layout.cpp src/kernels/cpu/q2_avx2.cpp src/kernels/cpu/hadamard.cpp)
target_include_directories(strata_kernels_cpu PUBLIC include)
# Plan v0.3 P6: only the kernel files get the AVX-512 / AVX2 flags; the pool and the probe run on any x86-64,
# and the engine picks the kernels at run time (cpu_avx512_ok).
if(MSVC)
  set_source_files_properties(src/kernels/cpu/expert.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  # GCC/Clang spell it differently, and `-mavx512vnni` alone is not enough: VBMI is a separate feature bit.
  set_source_files_properties(src/kernels/cpu/expert.cpp
                              PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()
# AVX-VNNI (Alder Lake, Sapphire Rapids and later): the AVX-2 expert kernels' vpdpwssd / vpdpbusd copies (q2_avx2.cpp,
# iq_avx2.cpp; *_rows.inl), taken at run time where cpu_avxvnni_ok().  GCC and Clang compile only those functions with
# target("avxvnni") - never -mavxvnni on a file, which would let the compiler use it anywhere - and MSVC takes the
# intrinsics under /arch:AVX2.  A compiler without them (GCC < 11, Clang < 12) builds the files without the copies.
include(CheckCXXSourceCompiles)
if(MSVC)
  set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
else()
  set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma -mf16c")
endif()
check_cxx_source_compiles([=[
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avxvnni")))
#endif
static __m256i dots(__m256i a, __m256i b) { return _mm256_dpbusd_avx_epi32(_mm256_dpwssd_avx_epi32(a, a, b), a, b); }
int main() {
    volatile int x = 1;
    return _mm256_extract_epi32(dots(_mm256_set1_epi32(x), _mm256_set1_epi32(x)), 0) == 0;
}
]=] STRATA_HAVE_AVXVNNI_INTRINSICS)
unset(CMAKE_REQUIRED_FLAGS)
if(STRATA_HAVE_AVXVNNI_INTRINSICS)
  set(_strata_avxvnni STRATA_AVXVNNI=1)
else()
  set(_strata_avxvnni STRATA_AVXVNNI=0)
endif()
set_property(SOURCE src/kernels/cpu/q2_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${_strata_avxvnni})
find_package(Threads REQUIRED)
target_link_libraries(strata_kernels_cpu PUBLIC Threads::Threads)

# ---- plan v0.3 P6: the i-quant models' CPU experts (IQ2_XS / IQ3_XXS files) ----
#
# The CPU computes a native expert with ggml-cpu's own dot products (`ggml_get_type_traits_cpu`), so an IQ
# weight means exactly what it means in llama.cpp.  ggml is built static from a llama.cpp checkout: a local one
# with -DSTRATA_GGML_DIR=..., otherwise the pinned upstream commit (third_party/ggml/VERSION.txt).
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")
if(STRATA_ORCA_Q4KS_MMQ AND NOT STRATA_NATIVE_EXPERTS)
  message(FATAL_ERROR "STRATA_ORCA_Q4KS_MMQ requires STRATA_NATIVE_EXPERTS=ON for GGML sources")
endif()
if(STRATA_NATIVE_EXPERTS)
  enable_language(C)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_STATIC ON CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  if(STRATA_PORTABLE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
      set(${_f} ON CACHE BOOL "" FORCE)
    endforeach()
  elseif(STRATA_ISA_FLOOR)
    # the experimental older-CPU build (STRATA_ISA_FLOOR above): one ggml-cpu for the floor, ggml's "sandybridge"
    # feature set (SSE4.2 + AVX) or SSE4.2 alone - not the build host's, which may be newer than the PC it runs on.
    # From the Strata_Dirigo fork (rwkeyes).  ggml's defaults turn the rest on when GGML_NATIVE is off: all off here.
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
    set(GGML_SSE42 ON CACHE BOOL "" FORCE)
    if(STRATA_ISA_FLOOR STREQUAL "avx")
      set(GGML_AVX ON CACHE BOOL "" FORCE)
    else()
      set(GGML_AVX OFF CACHE BOOL "" FORCE)
    endif()
    foreach(_f GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2 GGML_AVX_VNNI GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI
               GGML_AVX512_BF16 GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16)
      set(${_f} OFF CACHE BOOL "" FORCE)
    endforeach()
  else()
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  endif()
  if(STRATA_GGML_DIR)
    add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  else()
    include(FetchContent)
    FetchContent_Declare(strata_llamacpp
      GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
      GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d
      GIT_SHALLOW FALSE)
    FetchContent_GetProperties(strata_llamacpp)
    if(NOT strata_llamacpp_POPULATED)
      FetchContent_Populate(strata_llamacpp)
    endif()
    add_subdirectory(${strata_llamacpp_SOURCE_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  endif()
  # part of strata_kernels_cpu (the pool calls these and they call its CPU probe: one library, so GNU ld's
  # one-pass static linking has no cycle to resolve)
  target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/native_expert.cpp src/kernels/cpu/iq_avx512.cpp
                                           src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/kq_avx2.cpp
                                           src/kernels/cpu/kq_avx1.cpp src/kernels/cpu/q2_avx1.cpp src/kernels/cpu/iq_avx1.cpp
                                           src/kernels/cpu/nvfp4_avx512.cpp src/kernels/cpu/nvfp4_avx2.cpp
                                           src/kernels/cpu/tq_cpu.cpp src/kernels/cpu/tq_avx2.cpp src/kernels/cpu/tq_avx512.cpp)
  target_include_directories(strata_kernels_cpu PRIVATE third_party/ggml)
  if(MSVC)
    set_source_files_properties(src/kernels/cpu/iq_avx512.cpp src/kernels/cpu/nvfp4_avx512.cpp
                                PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/kernels/cpu/kq_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/kernels/cpu/nvfp4_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/kernels/cpu/tq_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/kernels/cpu/tq_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    # the router lookahead's AVX1 dot (older CPUs; called behind cpu_avx1_ok): no FMA, no 256-bit integer ops
    set_source_files_properties(src/kernels/cpu/kq_avx1.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
    # the older CPUs' Q2_0 rows and activation quantizer (q2_avx1.cpp, issue #1699): 128-bit integer lanes,
    # no FMA/F16C, its own #error guard; called only behind cpu_avx1_ok() && !cpu_avx2_ok()
    set_source_files_properties(src/kernels/cpu/q2_avx1.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
    # the older CPUs' i-quant expert rows (iq_avx1.hpp): 128-bit SSSE3/SSE2 integer, SSE2 float, no FMA/F16C
    set_source_files_properties(src/kernels/cpu/iq_avx1.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
  else()
    set_source_files_properties(src/kernels/cpu/iq_avx512.cpp src/kernels/cpu/nvfp4_avx512.cpp PROPERTIES COMPILE_OPTIONS
                                "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mavx512vpopcntdq;-mbmi2;-mf16c;-mfma")
    set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/kernels/cpu/kq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/kernels/cpu/nvfp4_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    # the trellis kernels: AVX2 + FMA, and AVX-512 F/BW/VL/VBMI/VNNI (exactly what cpu_avx512_ok() checks)
    set_source_files_properties(src/kernels/cpu/tq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/kernels/cpu/tq_avx512.cpp PROPERTIES COMPILE_OPTIONS
                                "-mavx512f;-mavx512bw;-mavx512vl;-mavx512vnni;-mavx512vbmi;-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/kernels/cpu/kq_avx1.cpp PROPERTIES COMPILE_OPTIONS "-mavx")
    # the older CPUs' Q2_0 rows and activation quantizer (q2_avx1.cpp, issue #1699): SSE4.2 + AVX with
    # AVX2/FMA/F16C explicitly off, so a -march=native build host cannot leak AVX2 code into the file that
    # runs on a Sandy Bridge Xeon
    set_source_files_properties(src/kernels/cpu/q2_avx1.cpp PROPERTIES COMPILE_OPTIONS
                                "-msse4.2;-mavx;-mno-avx2;-mno-fma;-mno-f16c")
    # the older CPUs' i-quant expert rows (iq_avx1.hpp): SSE4.2 + AVX with AVX2/FMA/F16C explicitly off, so a
    # -march=native build host cannot leak AVX2 code into the file that runs on a Sandy Bridge Xeon
    set_source_files_properties(src/kernels/cpu/iq_avx1.cpp PROPERTIES COMPILE_OPTIONS
                                "-msse4.2;-mavx;-mno-avx2;-mno-fma;-mno-f16c")
  endif()
  set_property(SOURCE src/kernels/cpu/iq_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${_strata_avxvnni})
  if(STRATA_ISA_FLOOR STREQUAL "avx")
    target_compile_definitions(strata_kernels_cpu PRIVATE STRATA_ISA_FLOOR_AVX=1)
  elseif(STRATA_ISA_FLOOR STREQUAL "none")
    target_compile_definitions(strata_kernels_cpu PRIVATE STRATA_ISA_FLOOR_NONE=1)
  endif()
  target_link_libraries(strata_kernels_cpu PUBLIC ggml-cpu ggml-base)
  target_compile_definitions(strata_kernels_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)
  set(_strata_mmq_types q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0)
  # APR: the agention llama.cpp fork has MMQ instances for its trellis expert types (f62fdead0); any other ggml
  # tree leaves them to the FP16 prompt path
  if(STRATA_GGML_DIR AND EXISTS "${STRATA_GGML_DIR}/ggml/src/ggml-cuda/template-instances/mmq-instance-tqk6.cu")
    list(APPEND _strata_mmq_types tq2_t tqk6 tqk7)
    set(_strata_mmq_tq ON)
  elseif(STRATA_MMQ_TQ_OVERLAY AND STRATA_ENABLE_CUDA)
    # an upstream llama.cpp: the trellis MMQ instances come from an overlay of the pinned ggml-cuda headers written at
    # configure time (cmake/strata_ggml_tq_overlay.cmake); off with a message if the llama.cpp tree is not the pinned one
    if(STRATA_GGML_DIR)
      set(_strata_tq_llama "${STRATA_GGML_DIR}")
    else()
      set(_strata_tq_llama "${strata_llamacpp_SOURCE_DIR}")
    endif()
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_ggml_tq_overlay.cmake)
    set(_strata_tq_overlay_dir "${CMAKE_BINARY_DIR}/ggml-tq-overlay")
    strata_ggml_tq_overlay("${_strata_tq_llama}" "${_strata_tq_overlay_dir}")
    if(STRATA_TQ_OVERLAY_OK)
      configure_file(third_party/ggml/ggml-common-tq.h "${_strata_tq_overlay_dir}/ggml-common-tq.h" COPYONLY)
      configure_file(src/kernels/cuda/tq_lut_i8.cuh "${_strata_tq_overlay_dir}/tq_lut_i8.cuh" COPYONLY)
      set(_strata_mmq_tq ON)
      set(_strata_mmq_tq_overlay ON)
      message(STATUS "trellis MMQ (Gyro prompts): overlay on the pinned ggml-cuda in ${_strata_tq_overlay_dir}")
    endif()
  endif()
  # #296: OrcaRouter Q4_K_S's Q5_0 experts (Q4_K and Q5_1 come with STRATA_MMQ_KQUANTS on CUDA, S5's branch)
  if(STRATA_ORCA_Q4KS_MMQ)
    list(APPEND _strata_mmq_types q5_0)
  endif()
  if(STRATA_ENABLE_CUDA)
    # prompt-speed plan step 2b: llama.cpp's MMQ kernels (ggml-cuda, MIT) for the prompt path's experts, compiled
    # from the same checkout as ggml above - only mmq.cuh's instances for the packs' types and the quantizer, with
    # the few host symbols of ggml-cuda.cu they use in src/prefill/ggml_cuda_host.cu
    if(STRATA_GGML_DIR)
      set(_strata_gg ${STRATA_GGML_DIR}/ggml)
    else()
      set(_strata_gg ${strata_llamacpp_SOURCE_DIR}/ggml)
    endif()
    set(_strata_mmq ${_strata_gg}/src/ggml-cuda)
    set(_strata_quantize_cu ${_strata_mmq}/quantize.cu)   # the overlay's copy when the trellis overlay is on: it includes mmq.cuh
    if(_strata_mmq_tq_overlay)
      set(_strata_quantize_cu ${_strata_tq_overlay_dir}/quantize.cu)
    endif()
    # #136: the Q2_0 pack's prompt experts on Strata's own int8 kernels (moe_fused.cu, opt-in STRATA_PF_FUSED=1; CUDA
    # only: an sm_75 device or an sm_75-only build reports them unavailable)
    add_library(strata_mmq STATIC src/prefill/moe_mmq.cu src/prefill/moe_fused.cu src/prefill/moe_fused_iq.cu
                src/prefill/mmq_nvfp4_w4a8.cu
                src/prefill/ggml_cuda_host.cu ${_strata_quantize_cu})
    # q8_0: the drafter's matrices (E-9); STRATA_ORCA_Q4KS_MMQ adds q5_0 (#296, in _strata_mmq_types), and
    # STRATA_MMQ_KQUANTS q4_k q5_k q5_1 q6_k, the experts of Unsloth's UD-Q4/Q5/Q6_K_XL (docs/QUANTS.md; the dispatch
    # follows eddoursul/Strata 7706c1b).  MODEL-UPDATE Phase 1: on CUDA they are a library of their own, strata_kq
    # (below), loaded only for a model with these experts: the engine loads every kernel of its own binary at start
    # (CUDA_MODULE_LOADING=EAGER), and these four took ~3.8 MB of VRAM (3 expert slots of IQ3_XXS on a P100) from every
    # model.
    set(_strata_mmq_cuda_types ${_strata_mmq_types})
    set(_strata_kq_types q4_k q5_k q5_1 q6_k)
    set(_strata_kq_dso OFF)
    if(STRATA_MMQ_KQUANTS)
      set(_strata_kq_dso ON)
    endif()
    foreach(_t IN LISTS _strata_mmq_cuda_types)
      target_sources(strata_mmq PRIVATE ${_strata_mmq}/template-instances/mmq-instance-${_t}.cu)
    endforeach()
    if(_strata_mmq_tq_overlay)
      foreach(_t tq2_t tqk6 tqk7)
        target_sources(strata_mmq PRIVATE ${_strata_tq_overlay_dir}/template-instances/mmq-instance-${_t}.cu)
      endforeach()
      # first on the include path: the patched mmq.cuh / ggml.h shadow the pinned ones for this target only
      target_include_directories(strata_mmq BEFORE PRIVATE ${_strata_tq_overlay_dir} ${_strata_tq_overlay_dir}/include)
    endif()
    # NVFP4: W4A8 (default) in mmq_nvfp4_w4a8.cu with the plain architectures; the opt-in W4A4 needs RTX 50's FP4 MMA,
    # which exists only in the arch-specific target, so its one unit alone is built with 12x -> 12xa (the default
    # binary is unchanged and runs on sm_121; a card without that image falls back to W4A8 at run time)
    set(_strata_w4a4_archs ${CMAKE_CUDA_ARCHITECTURES})
    # (two steps: CMake before 3.29 rejects a \2 whose optional group did not match, "120" alone)
    list(TRANSFORM _strata_w4a4_archs REPLACE "^(12[0-9])$" "\\1a")
    list(TRANSFORM _strata_w4a4_archs REPLACE "^(12[0-9])-(real|virtual)$" "\\1a-\\2")
    add_library(strata_mmq_w4a4 STATIC src/prefill/mmq_nvfp4_w4a4.cu)
    set_target_properties(strata_mmq_w4a4 PROPERTIES CUDA_ARCHITECTURES "${_strata_w4a4_archs}")
    target_include_directories(strata_mmq_w4a4 PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq})
    target_compile_options(strata_mmq_w4a4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
    if(MSVC)
      target_compile_options(strata_mmq_w4a4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
    endif()
    target_link_libraries(strata_mmq_w4a4 PUBLIC ggml-base ${_strata_gpu_runtime_target})
    target_link_libraries(strata_mmq PRIVATE strata_mmq_w4a4)
    # STRATA_MMQ_TAB: the pointer-table copy of MMQ's scheduler mirrors mmq.cuh from mul_mat_q_process_tile through
    # mul_mat_q_case (struct mmq_args included).  A fingerprint of that text instead of a compile probe: nothing is
    # compiled at configure time, and a llama.cpp tree with another scheduler turns the feature off instead of
    # failing the build.  _strata_mmq_tab_known: the hashes moe_mmq_tab.cuh was verified against (3cf03257, the
    # fetched pin; the vendored third_party/llama.cpp tree has the same mmq.cuh).
    set(_strata_mmq_tab_on OFF)
    if(STRATA_MMQ_TAB)
      set(_strata_mmq_tab_known 36e4dc716c504470ec09cc0852d29a6ac5eace257f6e5e72db3805ca4f043eb0)
      set(_strata_mmq_tab_hash "")
      if(EXISTS ${_strata_mmq}/mmq.cuh)
        file(READ ${_strata_mmq}/mmq.cuh _strata_mmq_src)
        string(REPLACE "\r\n" "\n" _strata_mmq_src "${_strata_mmq_src}")
        string(FIND "${_strata_mmq_src}" "static __device__ __forceinline__ void mul_mat_q_process_tile" _strata_tab_a)
        string(FIND "${_strata_mmq_src}" "#define DECL_MMQ_CASE" _strata_tab_b)
        if(_strata_tab_a GREATER -1 AND _strata_tab_b GREATER _strata_tab_a)
          math(EXPR _strata_tab_len "${_strata_tab_b} - ${_strata_tab_a}")
          string(SUBSTRING "${_strata_mmq_src}" ${_strata_tab_a} ${_strata_tab_len} _strata_tab_region)
          string(SHA256 _strata_mmq_tab_hash "${_strata_tab_region}")
        endif()
        unset(_strata_mmq_src)
        unset(_strata_tab_region)
      endif()
      if(_strata_mmq_tab_hash AND _strata_mmq_tab_hash IN_LIST _strata_mmq_tab_known)
        set(_strata_mmq_tab_on ON)
      else()
        message(STATUS "STRATA_MMQ_TAB off: mmq.cuh differs from the mirrored scheduler (${_strata_mmq_tab_hash})")
      endif()
    endif()
    if(_strata_mmq_tab_on)
      set(_strata_tab_cuh ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/moe_mmq_tab.cuh)
      # the native packs' nine types, the ones prefill_mmq_tab_test checks bit for bit; the optional Q5_0 / K-quant
      # instances (STRATA_ORCA_Q4KS_MMQ, STRATA_MMQ_KQUANTS) get no pointer table and keep the gather
      set(_strata_mmq_tab_types q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0)
      foreach(_t IN LISTS _strata_mmq_tab_types)
        string(TOUPPER ${_t} _strata_tab_T)
        configure_file(src/prefill/mmq_tab_instance.cu.in ${CMAKE_BINARY_DIR}/mmq_tab/mmq-tab-${_t}.cu @ONLY)
        target_sources(strata_mmq PRIVATE ${CMAKE_BINARY_DIR}/mmq_tab/mmq-tab-${_t}.cu)
      endforeach()
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_TAB=1)
      message(STATUS "STRATA_MMQ_TAB on: pointer-table MMQ for ${_strata_mmq_tab_types}")
    endif()
    target_include_directories(strata_mmq PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq}
                               PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(strata_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
    if(MSVC)
      target_compile_options(strata_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
    endif()
    # GGML's CUDA error reporting calls cuGetErrorString from the driver API.
    target_link_libraries(strata_mmq PUBLIC ggml-base ${_strata_gpu_runtime_target})
    if(TARGET CUDA::cuda_driver)
      target_link_libraries(strata_mmq PUBLIC CUDA::cuda_driver)
    endif()
    if(STRATA_MMQ_KQUANTS)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_KQUANTS=1)
    endif()
    if(_strata_kq_dso)
      # strata_kq: the K-quant MMQ instances with their own copy of ggml_cuda_host.cu's symbols and the static CUDA
      # runtime, next to the engine (libstrata_kq.so / strata_kq.dll); moe_mmq.cu opens it when a model asks for one
      # of these formats (STRATA_MMQ_KQ=1).  Without the file the engine runs them on the FP16 prompt path, as before.
      target_compile_definitions(strata_mmq PRIVATE STRATA_KQ_DSO=1)
      target_sources(strata_mmq PRIVATE src/prefill/mmq_kq_loader.cpp)
      if(NOT WIN32)
        target_link_libraries(strata_mmq PUBLIC ${CMAKE_DL_LIBS})
      endif()
      add_library(strata_kq SHARED src/prefill/mmq_kq_lib.cu src/prefill/ggml_cuda_host.cu)
      foreach(_t IN LISTS _strata_kq_types)
        target_sources(strata_kq PRIVATE ${_strata_mmq}/template-instances/mmq-instance-${_t}.cu)
      endforeach()
      target_include_directories(strata_kq PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/include)
      target_compile_options(strata_kq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
      if(MSVC)
        target_compile_options(strata_kq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
      endif()
      set_target_properties(strata_kq PROPERTIES CUDA_RUNTIME_LIBRARY Static POSITION_INDEPENDENT_CODE ON
                            CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden)
      # ggml-base is static: position-independent so a shared library can hold it (the same code otherwise)
      set_target_properties(ggml-base PROPERTIES POSITION_INDEPENDENT_CODE ON)
      target_link_libraries(strata_kq PRIVATE ggml-base)
      if(TARGET CUDA::cuda_driver)
        target_link_libraries(strata_kq PRIVATE CUDA::cuda_driver)
      endif()
      add_dependencies(strata_mmq strata_kq)
    endif()
    if(STRATA_ORCA_Q4KS_MMQ)
      target_compile_definitions(strata_mmq PRIVATE STRATA_ORCA_Q4KS_MMQ=1)
    endif()
    if(_strata_mmq_tq)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_TQ=1)
    endif()
    target_link_libraries(strata_prefill PUBLIC strata_mmq)
    target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_MMQ=1 STRATA_PREFILL_FUSED=1)
  endif()
endif()
