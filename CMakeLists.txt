cmake_minimum_required(VERSION 3.9)
project(MCX LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

option(USE_XORSHIFT128P_RAND "USE xorshift128+ RNG" ON)
option(USE_POSIX_RAND "Use POSIX erand48 RNG" OFF)
option(USE_MT_RAND "Use Mersenne Twister RNG (MT)" OFF)
option(USE_LOGISTIC_RNG "Use Logistic Lattice ring 5 RNG" OFF)

if(USE_XORSHIFT128P_RAND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUSE_XORSHIFT128P_RAND")
elif(USE_POSIX_RAND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUSE_POSIX_RAND")
elif(USE_MT_RAND)
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
        src/cjson/cJSON.c src/cjson/cJSON.h)

add_library(libmcx SHARED src/mcx_ffi.c src/mcx_ffi.h ${SRCS})
add_executable(mcx src/mcextreme.c ${SRCS})

target_link_libraries(libmcx ${OpenMP_CXX_LIBRARIES})
target_link_libraries(mcx ${OpenMP_CXX_LIBRARIES})

find_package(MATLAB COMPONENTS MEX_COMPILER)
if(MATLAB_found)
    matlab_add_mex(mcxlab SHARED src/mcxlab.cpp ${SRCS} LINK_TO ${OpenMP_CXX_LIBRARIES})
endif()

SET_TARGET_PROPERTIES(libmcx PROPERTIES PREFIX "")

find_package(PythonInterp)
if(PYTHONINTERP_FOUND)
    add_custom_target(pymcx ALL DEPENDS libmcx
        COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:libmcx>" "${CMAKE_BINARY_DIR}/pymcx/$<TARGET_FILE_NAME:libmcx>"
        COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_SOURCE_DIR}/pymcx/__init__.py" "${CMAKE_BINARY_DIR}/pymcx/__init__.py"
        COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/setup.py" "bdist_wheel" "-d${CMAKE_BINARY_DIR}"
    )
endif()