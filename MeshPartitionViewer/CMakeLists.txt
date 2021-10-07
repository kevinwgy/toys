cmake_minimum_required(VERSION 3.10)

# set the project name
project(MeshPartitionViewer VERSION 1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# compiler flags (turn on most of the warnings, except unused variables and parameters
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-reorder -Wno-sign-compare -Wno-unused-variable -Wno-unused-parameter -Wno-unused-but-set-variable")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# folders for executable and libraries
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# add include and link directories 
include_directories("${CMAKE_SOURCE_DIR}")

# add the executable
add_executable(dec2xpost
dec2xpost.cpp)

