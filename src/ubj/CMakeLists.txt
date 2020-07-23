cmake_minimum_required(VERSION 2.8)
project(ubj)

cmake_policy(SET CMP0076 NEW)

add_library(ubj INTERFACE)
target_sources(ubj INTERFACE ubjw.c ubjr.c ubjrw.c ubj_internal.h ubj.h)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
