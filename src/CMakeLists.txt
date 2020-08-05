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
    add_compile_definitions(MCX_DEBUG)
    if(TEST_RACING)
        add_compile_definitions(TEST_RACING)
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

if(EXCLUSIONARY_DETECTORS)
    add_compile_definitions(EXCLUSIONARY_DETECTORS)
endif()

set(prngChoices "xorshiro128+" "xorshift128+" "POSIX erand48" "Mersenne Twister" "Logistic Lattice ring 5")
set(prng "xorshiro128+" CACHE STRING "Which PRNG to use")
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

find_package(Matlab COMPONENTS MEX_COMPILER)
if(Matlab_FOUND)
    option(USE_MEX_COMPILER "Use matlab mex compiler" ON)
    if(USE_MEX_COMPILER)
        matlab_add_mex(NAME mcxlab MODULE SRC mcxlab.cpp ${SRCS} OUTPUT_NAME "mcx")
    else()
        add_library(mcxlab MODULE mcxlab.cpp ${SRCS})
        target_include_directories(mcxlab PUBLIC ${Matlab_INCLUDE_DIRS})
        target_link_libraries(mcxlab ${Matlab_MEX_LIBRARY} ${Matlab_MX_LIBRARY})
        set_target_properties(mcxlab PROPERTIES
                PREFIX ""
                OUTPUT_NAME "mcx"
                SUFFIX ".${Matlab_MEX_EXTENSION}"
                C_VISIBILITY_PRESET hidden
                CXX_VISIBILITY_PRESET hidden
                CUDA_VISIBILITY_PRESET hidden
                CMAKE_VISIBILITY_INLINES_HIDDEN TRUE
                )
    endif()
endif()

if(SKBUILD)
    find_package(PythonExtensions REQUIRED)
    find_package(Cython REQUIRED)
    add_compile_definitions(NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
    add_cython_target(cymcx pymcx.pyx PY3)
    add_library(pymcx MODULE ${cymcx} ${SRCS})
    python_extension_module(pymcx)
    # Needed b/c scikit-build doesn't support standalone extension modules
    file(WRITE "${CMAKE_INSTALL_PREFIX}/pymcx/__init__.py" "from .pymcx import *\n")
    # Needed for scikit-build
    install(TARGETS pymcx LIBRARY DESTINATION "pymcx")
endif()
