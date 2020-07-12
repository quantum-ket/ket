cmake_minimum_required(VERSION 3.15)

project(ketpy)

if(DEFINED SNAP)
    add_definitions(-DSNAP)
endif()

add_subdirectory(libket)
add_subdirectory(kbw)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})

SET_PROPERTY(SOURCE src/ket.i
             PROPERTY CPLUSPLUS ON)
SWIG_ADD_LIBRARY(ketpy 
                 LANGUAGE python
                 OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/ket
                 SOURCES src/ket.i)

SET_TARGET_PROPERTIES(ketpy PROPERTIES 
                      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ket
                      CXX_STANDARD 17)
                      
SWIG_LINK_LIBRARIES(ketpy ${PYTHON_LIBRARIES} ket)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ket/__main__.py ${CMAKE_CURRENT_BINARY_DIR}/ket COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ket/import_ket.py ${CMAKE_CURRENT_BINARY_DIR}/ket COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ket/ketpp.py ${CMAKE_CURRENT_BINARY_DIR}/ket COPYONLY)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ket DESTINATION ${PYTHON_PACKAGES})
