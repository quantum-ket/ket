cmake_minimum_required(VERSION 3.15)

project(ketpy)

if(NOT CMAKE_BUILD_TYPE) 
    set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
    message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
    file(DOWNLOAD "https://github.com/conan-io/cmake-conan/raw/v0.15/conan.cmake"
                  "${CMAKE_BINARY_DIR}/conan.cmake")
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

conan_cmake_run(REQUIRES boost/1.71.0
                BASIC_SETUP CMAKE_TARGETS
                BUILD missing)

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})

file(GLOB_RECURSE LIBKET_SRC libket/src/*.cpp)

ADD_LIBRARY(ket SHARED
            src/ket.cpp 
            ${LIBKET_SRC})

SET_TARGET_PROPERTIES(ket PROPERTIES 
                      CXX_STANDARD 17
                      PREFIX "_"
                      POSITION_INDEPENDENT_CODE ON
                     )
                      
TARGET_LINK_LIBRARIES(ket ${PYTHON_LIBRARIES} CONAN_PKG::boost)

install(TARGETS ket DESTINATION ket)
