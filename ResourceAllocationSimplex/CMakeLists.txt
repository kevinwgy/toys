cmake_minimum_required(VERSION 3.10)

# set the project name
project(ResourceAllocationSimplex VERSION 1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# use mpicc and mpicxx to compile
#SET(CMAKE_C_COMPILER mpicc)
#SET(CMAKE_CXX_COMPILER mpicxx)

# compiler flags (turn on all the warnings)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-reorder")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# folders for executable and libraries
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# add include and link directories 
include_directories("${CMAKE_SOURCE_DIR}")
link_directories(${CMAKE_SOURCE_DIR}/parser)

# compile the parser
add_custom_target(
   extern_lib
   COMMAND $(MAKE)
   WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/parser
)

# add the executable
add_executable(allocator
Main.cpp
IoData.cpp)

# link to libraries
target_link_libraries(allocator parser)
add_dependencies(allocator extern_lib)
