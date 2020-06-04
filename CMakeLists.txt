cmake_minimum_required(VERSION 3.15)

project(pyket)

add_subdirectory(libket EXCLUDE_FROM_ALL)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})

SET_PROPERTY(SOURCE src/ket.i
             PROPERTY CPLUSPLUS ON)
SWIG_ADD_LIBRARY(pyket 
                 LANGUAGE python
                 OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/ket
                 SOURCES src/ket.i src/pyket.cpp)

set_target_properties(pyket PROPERTIES 
                      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ket
                      CXX_STANDARD 17)
                      
SWIG_LINK_LIBRARIES(pyket ${PYTHON_LIBRARIES} ket)
