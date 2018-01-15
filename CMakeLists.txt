cmake_minimum_required(VERSION 3.9)
project(MCX LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP)
if (OPENMP_C_FOUND AND OPENMP_CXX_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUSE_XORSHIFT128P_RAND -DUSE_ATOMIC -DUSE_CACHEBOX -DSAVE_DETECTORS")

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

SET (SRCS src/mcx_core.cu src/mcx_core.h
        src/mcx_utils.c src/mcx_utils.h
        src/mcx_shapes.c src/mcx_shapes.h
        src/tictoc.c src/tictoc.h
        src/cjson/cJSON.c src/cjson/cJSON.h)

add_library(OBJS OBJECT ${SRCS})
add_executable(mcx src/mcextreme.c $<TARGET_OBJECTS:OBJS>)
add_library(libmcx SHARED src/mcx_ffi.c src/mcx_ffi.h $<TARGET_OBJECTS:OBJS>)
