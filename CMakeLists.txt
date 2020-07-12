cmake_minimum_required(VERSION 3.15)

project(ketpy)

add_subdirectory(libket)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

conan_cmake_run(REQUIRES swig/4.0.1
                BASIC_SETUP 
                BUILD missing)

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})

SET_PROPERTY(SOURCE src/ket.i
             PROPERTY CPLUSPLUS ON)
SWIG_ADD_LIBRARY(ketpy 
                 LANGUAGE python
                 SOURCES src/ket.i)

SET_TARGET_PROPERTIES(ketpy PROPERTIES 
                      CXX_STANDARD 17)
                      
SWIG_LINK_LIBRARIES(ketpy ${PYTHON_LIBRARIES} ket)

install(TARGETS ketpy DESTINATION ket)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py DESTINATION ket)
