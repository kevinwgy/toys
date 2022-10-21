PROJECT (PseudoStructure)

cmake_minimum_required(VERSION 2.6)

FIND_PACKAGE(MPI)
add_definitions(-DUSE_MPI -DMPICH_IGNORE_CXX_SEEK)

SET(INC_DIR .)
SET(INC_DIR ${INC_DIR} ${MPI_INCLUDE_PATH})

INCLUDE_DIRECTORIES( ${INC_DIR} )

add_subdirectory(Comm)
add_subdirectory(Hetero)

add_executable(cplCode Main.C)

SET_TARGET_PROPERTIES(comm hetero
                PROPERTIES COMPILE_FLAGS -fPIC)

ADD_LIBRARY(cplLib SHARED Main.C)

TARGET_LINK_LIBRARIES(cplCode comm hetero ${MPI_LIBRARIES} ${CMAKE_DL_LIBS})
TARGET_LINK_LIBRARIES(cplLib comm hetero ${CMAKE_DL_LIBS})
