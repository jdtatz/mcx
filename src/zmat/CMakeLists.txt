cmake_minimum_required(VERSION 3.13)
project(zmat)

cmake_policy(SET CMP0076 NEW)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(ZLIB REQUIRED)

include_directories(lz4)
include_directories(easylzma)
include_directories(easylzma/pavlov)

add_library(zmat INTERFACE)
target_sources(
  zmat
  INTERFACE zmatlib.c
            lz4/lz4.c
            lz4/lz4hc.c
            easylzma/compress.c
            easylzma/decompress.c
            easylzma/lzma_header.c
            easylzma/lzip_header.c
            easylzma/common_internal.c
            easylzma/pavlov/LzmaEnc.c
            easylzma/pavlov/LzmaDec.c
            easylzma/pavlov/LzmaLib.c
            easylzma/pavlov/LzFind.c
            easylzma/pavlov/Bra.c
            easylzma/pavlov/BraIA64.c
            easylzma/pavlov/Alloc.c
            easylzma/pavlov/7zCrc.c)
# Needed to prevent lz4 lib from overriding visibility presets
target_compile_definitions(zmat INTERFACE "LZ4LIB_VISIBILITY=\t")

target_link_libraries(zmat INTERFACE ZLIB::ZLIB)

if(Matlab_FOUND)
  matlab_add_mex(
    NAME zipmat
    SRC zmat.cpp
    LINK_TO mex mx zmat)
endif()
