cmake_minimum_required(VERSION 3.10)

# set the project name
project(xp2exo VERSION 1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# set compilers
SET(CMAKE_C_COMPILER gcc)
SET(CMAKE_CXX_COMPILER g++)

# compiler flags (turn on most of the warnings, except unused variables and parameters
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-reorder -Wno-unused-parameter -Wno-write-strings -Wno-format-overflow")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# find NetCDF
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(NetCDF REQUIRED)

# add include and link directories 
include_directories("${CMAKE_SOURCE_DIR}")
include_directories(${CMAKE_SOURCE_DIR}/exodus-6.09/exodus/cbind/include)
link_directories(${CMAKE_SOURCE_DIR}/exodus-6.09/exodus/cbind)

# add the executable
add_executable(xp2exo xp2exo.cpp)

# link to libraries
target_link_libraries(xp2exo exodus netcdf)
