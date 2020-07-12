cmake_minimum_required(VERSION 3.15)

project(ketpy)

if(NOT CMAKE_BUILD_TYPE) 
    set(CMAKE_BUILD_TYPE Debug)
endif()

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
    message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
    file(DOWNLOAD "https://github.com/conan-io/cmake-conan/raw/v0.15/conan.cmake"
                  "${CMAKE_BINARY_DIR}/conan.cmake")
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

conan_cmake_run(REQUIRES swig/4.0.1 boost/1.71.0
                BASIC_SETUP CMAKE_TARGETS
                BUILD missing)

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})

file(GLOB_RECURSE LIBKET_SRC libket/src/*.cpp)

SET_PROPERTY(SOURCE src/ket.i
             PROPERTY CPLUSPLUS ON)
SWIG_ADD_LIBRARY(ketpy 
                 LANGUAGE python
                 SOURCES src/ket.i 
                         ${LIBKET_SRC})

SET_TARGET_PROPERTIES(ketpy PROPERTIES 
                      CXX_STANDARD 17)
                      
TARGET_LINK_LIBRARIES(ketpy ${PYTHON_LIBRARIES} CONAN_PKG::boost)

install(TARGETS ketpy DESTINATION ket)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py DESTINATION ket)
