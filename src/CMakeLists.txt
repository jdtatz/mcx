cmake_minimum_required(VERSION 3.13)

add_subdirectory(ubj)
add_subdirectory(zmat)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=compute_35")
else()
    set(CMAKE_CUDA_ARCHITECTURES 35-virtual)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    option(TEST_RACING "Test Racing Conditions" OFF)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMCX_DEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMCX_DEBUG")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DMCX_DEBUG")
    if(TEST_RACING)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTEST_RACING")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTEST_RACING")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DTEST_RACING")
    endif()
else()
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

option(ATOMIC "Use CUDA atomic operations" ON)
option(CACHEBOX "Use MCX cachebox" OFF)
option(SAVE_DETECTORS "Turn on ability to save detected photons" ON)
option(USE_HALF "Use half-precision for ray-tracing" OFF)
option(EXCLUSIONARY_DETECTORS "Turn on ability to use exclusionary detectors" OFF)

if(ATOMIC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUSE_ATOMIC")
endif()

if(CACHEBOX)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUSE_CACHEBOX")
endif()

if(SAVE_DETECTORS)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DSAVE_DETECTORS")
endif()

if(USE_HALF)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUSE_HALF")
endif()

if(EXCLUSIONARY_DETECTORS)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DEXCLUSIONARY_DETECTORS")
endif()

set(prngChoices "xorshiro128+" "xorshift128+" "POSIX erand48" "Mersenne Twister" "Logistic Lattice ring 5")
set(prng "xorshiro128+" CACHE STRING "Which PRNG to use")
set_property(CACHE prng PROPERTY STRINGS ${prngChoices})

list(FIND prngChoices ${prng} index)
if(index EQUAL -1)
    message(FATAL_ERROR "prng must be one of ${prngChoices}")
elseif(index EQUAL 0)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUSE_XOROSHIRO128P_RAND")
elseif(index EQUAL 1)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUSE_XORSHIFT128P_RAND")
elseif(index EQUAL 2)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUSE_POSIX_RAND")
elseif(index EQUAL 3)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUSE_MT_RAND")
endif()


find_package(OpenMP REQUIRED)

link_libraries(OpenMP::OpenMP_C OpenMP::OpenMP_CXX ubj zmat)

include_directories("." ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

SET (SRCS mcx_core.cu mcx_core.h
        mcx_utils.c mcx_utils.h
        mcx_shapes.c mcx_shapes.h
        tictoc.c tictoc.h
        cjson/cJSON.c cjson/cJSON.h
        )

add_executable(mcx mcextreme.c ${SRCS})

find_package(MATLAB COMPONENTS MEX_COMPILER)
if(MATLAB_found)
    matlab_add_mex(mcxlab SHARED mcxlab.cpp ${SRCS})
endif()

if(SKBUILD)
    find_package(PythonExtensions REQUIRED)
    find_package(Cython REQUIRED)
    add_cython_target(cymcx pymcx.pyx PY3)
    add_library(pymcx MODULE ${cymcx} ${SRCS})
    python_extension_module(pymcx)
    # Needed for scikit-build
    install(TARGETS pymcx LIBRARY DESTINATION "pymcx")
endif()
