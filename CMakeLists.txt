cmake_minimum_required(VERSION 3.15)

project(ketpy)

add_subdirectory(libket EXCLUDE_FROM_ALL)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})

SET_PROPERTY(SOURCE src/ket.i
             PROPERTY CPLUSPLUS ON)
SWIG_ADD_LIBRARY(ketpy 
                 LANGUAGE python
                 OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/ket
                 SOURCES src/ket.i src/pyket.cpp)

SET_TARGET_PROPERTIES(ketpy PROPERTIES 
                      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ket
                      CXX_STANDARD 17)
                      
SWIG_LINK_LIBRARIES(ketpy ${PYTHON_LIBRARIES} ket)
