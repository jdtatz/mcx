cmake_minimum_required(VERSION 3.16)

add_subdirectory(ubj)
add_subdirectory(zmat)
link_libraries(ubj zmat)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden) 
set(CMAKE_CUDA_VISIBILITY_PRESET hidden) 
set(CMAKE_VISIBILITY_INLINES_HIDDEN hidden)
if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
  # BAD, one should never change compiler flags manually with cmake
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=compute_35")
else()
  set(CMAKE_CUDA_ARCHITECTURES 35-virtual)
endif()
if(WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-DCJSON_HIDE_SYMBOLS)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  option(TEST_RACING "Test Racing Conditions" OFF)
  add_compile_definitions(MCX_DEBUG)
  if(TEST_RACING)
    add_compile_definitions(TEST_RACING)
  endif()
else()
  include(CheckIPOSupported)
  check_ipo_supported(RESULT result OUTPUT output)
  if(result)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
      add_link_options("-fuse-ld=lld")
    endif()
    if(MSVC)
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"-GL\"")
      add_link_options("/LTCG")
    endif()
    message(VERBOSE "Link-Time-Optimization is enabled")
  else()
    message(WARNING "Link-Time-Optimization is not supported: ${output}")
  endif()
endif()

option(ATOMIC "Use CUDA atomic operations" ON)
option(CACHEBOX "Use MCX cachebox" OFF)
option(SAVE_DETECTORS "Turn on ability to save detected photons" ON)
option(USE_HALF "Use half-precision for ray-tracing" OFF)

if(ATOMIC)
  add_compile_definitions(USE_ATOMIC)
endif()

if(CACHEBOX)
  add_compile_definitions(USE_CACHEBOX)
endif()

if(SAVE_DETECTORS)
  add_compile_definitions(SAVE_DETECTORS)
endif()

if(USE_HALF)
  add_compile_definitions(USE_HALF)
endif()

set(prngChoices "xorshiro128+" "xorshift128+" "POSIX erand48"
                "Mersenne Twister" "Logistic Lattice ring 5")
set(prng
    "xorshiro128+"
    CACHE STRING "Which PRNG to use")
set_property(CACHE prng PROPERTY STRINGS ${prngChoices})

list(FIND prngChoices ${prng} index)
if(index EQUAL -1)
  message(FATAL_ERROR "prng must be one of ${prngChoices}")
elseif(index EQUAL 0)
  add_compile_definitions(USE_XOROSHIRO128P_RAND)
elseif(index EQUAL 1)
  add_compile_definitions(USE_XORSHIFT128P_RAND)
elseif(index EQUAL 2)
  add_compile_definitions(USE_POSIX_RAND)
elseif(index EQUAL 3)
  add_compile_definitions(USE_MT_RAND)
endif()

find_package(OpenMP)
if(OpenMP_FOUND)
  link_libraries(OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
  # VERY BAD, one should never change compiler flags manually with cmake
  # but it is needed as nvcc wouldn't use openmp otherwise.
  # There is no OpenMP::OpenMP_CUDA, and OpenMP::OpenMP_CXX isn't used for
  # the nvcc host compiler
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"${OpenMP_CXX_FLAGS}\"")
  endif()
endif()

include_directories("." ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(SRCS
    mcx_core.cu
    mcx_utils.c
    mcx_shapes.c
    mcx_shapes.h
    mcx_bench.c
    mcx_bench.h
    tictoc.c
    cjson/cJSON.c
)

add_executable(mcx mcextreme.c ${SRCS})

find_package(Matlab COMPONENTS MEX_COMPILER)
if(Matlab_FOUND)
  option(USE_MEX_COMPILER "Use matlab mex compiler" ON)
  if(USE_MEX_COMPILER)
    matlab_add_mex(
      NAME mcxlab
      MODULE
      SRC mcxlab.cpp ${SRCS}
      OUTPUT_NAME "mcx")
  else()
    add_library(mcxlab MODULE mcxlab.cpp ${SRCS})
    target_include_directories(mcxlab PUBLIC ${Matlab_INCLUDE_DIRS})
    target_link_libraries(mcxlab ${Matlab_MEX_LIBRARY} ${Matlab_MX_LIBRARY})
    set_target_properties(
      mcxlab
      PROPERTIES PREFIX ""
                 OUTPUT_NAME "mcx"
                 SUFFIX ".${Matlab_MEX_EXTENSION}"
                 C_VISIBILITY_PRESET hidden
                 CXX_VISIBILITY_PRESET hidden
                 CUDA_VISIBILITY_PRESET hidden
                 CMAKE_VISIBILITY_INLINES_HIDDEN TRUE)
  endif()
endif()

find_package(Python3 COMPONENTS Interpreter Development.Module)
if(Python3_FOUND)
  add_custom_command(
    OUTPUT _pymcx.c
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/cffi_build.py
    DEPENDS Python3::Interpreter cffi_build.py
  )
  # DO NOT USE WITH_SOABI, its makes it abi3 incompatible
  Python3_add_library(_pymcx MODULE ${CMAKE_CURRENT_BINARY_DIR}/_pymcx.c ${SRCS})
  # This might be needed only on windows
  if(WIN32)
    target_link_directories(_pymcx PUBLIC ${Python3_LIBRARY_DIRS})
  endif()
  install(TARGETS _pymcx LIBRARY DESTINATION "python/pymcx")
endif()
