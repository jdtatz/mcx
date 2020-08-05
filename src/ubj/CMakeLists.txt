cmake_minimum_required(VERSION 3.13)
project(ubj)

cmake_policy(SET CMP0076 NEW)

add_library(ubj INTERFACE)
target_sources(ubj INTERFACE ubjw.c ubjr.c ubjrw.c)
