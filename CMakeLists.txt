cmake_minimum_required(VERSION 3.26)

project(FluidSolver LANGUAGES CXX)

option(LBM_FAST "Enable unsafe optimizations." OFF)
option(LBM_SANITIZE "Enable sanitizers." OFF)
option(LBM_USE_SCOREP "Enable profiling using Score-P." OFF)
option(LBM_BUILD_BENCH "Build benchmarks." OFF)

# = Add dependencies ===============================================================================
# = Igor =========================================
add_subdirectory(${CMAKE_SOURCE_DIR}/Submodules/Igor)
# = Add dependencies ===============================================================================

find_package(OpenMP REQUIRED)
find_package(HDF5 COMPONENTS C CXX HL)
if (NOT HDF5_FOUND)
  add_compile_definitions(LBM_DISABLE_HDF)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(cmake/compiler_flags.cmake)
add_compile_definitions(LBM_BASE_DIR="${CMAKE_SOURCE_DIR}")

add_subdirectory(${CMAKE_SOURCE_DIR}/examples/)

if (LBM_BUILD_BENCH)
  find_package(benchmark REQUIRED)
  add_subdirectory(${CMAKE_SOURCE_DIR}/bench/)
endif()
