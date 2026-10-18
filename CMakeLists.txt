cmake_minimum_required(VERSION 3.12)
project(KFDNF_Filter LANGUAGES CXX)

# Default to a Release build: it defines NDEBUG, which switches off Eigen's run-time
# assertions (without it a shape mismatch calls abort() and kills the Python process).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(KFDNF_NATIVE    "Compile with -march=native (binary only runs on this CPU family)" ON)
option(KFDNF_FAST_MATH "Compile with -ffast-math (NaN/Inf semantics are kept, see README)" ON)

# Finds Python and Pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Finds Eigen3
find_package(Eigen3 REQUIRED)

# Creates the Python module (whose name will be "kfdnf_fast"); bindings.cpp includes kfdnf_core.cpp
pybind11_add_module(kfdnf_fast cpp/bindings.cpp)

# Links Eigen to our module
target_link_libraries(kfdnf_fast PRIVATE Eigen3::Eigen)

# The compiled module is placed next to the Python modules, in src/
set_target_properties(kfdnf_fast PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Optimizations (GCC / Clang flags)
if(NOT MSVC)
  target_compile_options(kfdnf_fast PRIVATE -O3)
  if(KFDNF_NATIVE)
    target_compile_options(kfdnf_fast PRIVATE -march=native)
  endif()
  if(KFDNF_FAST_MATH)
    # -ffast-math implies -ffinite-math-only, which lets the compiler assume there are no
    # NaN/Inf and drop std::isnan / Eigen::allFinite() checks: undo that part.
    target_compile_options(kfdnf_fast PRIVATE -ffast-math -fno-finite-math-only)
  endif()
endif()
