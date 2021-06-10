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

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/libket/cmake;${CMAKE_MODULE_PATH}")

find_package(GMP REQUIRED)
            
FIND_PACKAGE(Python COMPONENTS Development.Embed)
INCLUDE_DIRECTORIES(${Python_INCLUDE_DIRS})

file(GLOB_RECURSE LIBKET_SRC libket/src/*.cpp)

ADD_LIBRARY(ket SHARED
            src/ket.cpp 
            ${LIBKET_SRC})

SET_TARGET_PROPERTIES(ket PROPERTIES 
                      CXX_STANDARD 17
                      PREFIX "_"
                      POSITION_INDEPENDENT_CODE ON
                     )
                      
TARGET_LINK_LIBRARIES(ket ${Python_LIBRARIES} CONAN_PKG::boost gmp)

install(TARGETS ket DESTINATION ket)
