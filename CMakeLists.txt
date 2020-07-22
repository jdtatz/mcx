cmake_minimum_required(VERSION 3.12)
project(MCX LANGUAGES C CXX CUDA)

set(PACKAGE_VERSION "0.1.1")

add_subdirectory(src/ubj)
add_subdirectory(src/zmat)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 35-virtual)

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
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"${OpenMP_CXX_FLAGS}\"")

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

SET (SRCS src/mcx_core.cu src/mcx_core.h
        src/mcx_utils.c src/mcx_utils.h
        src/mcx_shapes.c src/mcx_shapes.h
        src/tictoc.c src/tictoc.h
        src/cjson/cJSON.c src/cjson/cJSON.h
        )

add_library(libmcx SHARED src/mcx_ffi.c src/mcx_ffi.h ${SRCS})
add_executable(mcx src/mcextreme.c ${SRCS})

target_link_libraries(libmcx ${OpenMP_C_LIBRARIES} ubj zmat)
target_link_libraries(mcx ${OpenMP_C_LIBRARIES} ubj zmat)

set_target_properties(libmcx PROPERTIES PREFIX "")

find_package(MATLAB COMPONENTS MEX_COMPILER)
if(MATLAB_found)
    matlab_add_mex(mcxlab SHARED src/mcxlab.cpp ${SRCS} LINK_TO ${OpenMP_CXX_LIBRARIES})
endif()

if(SKBUILD)
    install(TARGETS libmcx LIBRARY DESTINATION pymcx)
endif()
