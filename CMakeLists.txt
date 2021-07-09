cmake_minimum_required(VERSION 3.15)
project(ketpy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
    message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
    file(DOWNLOAD "https://github.com/conan-io/cmake-conan/raw/v0.15/conan.cmake"
                  "${CMAKE_BINARY_DIR}/conan.cmake")
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

conan_cmake_run(REQUIRES boost/1.71.0
                         gmp/6.2.1
                BASIC_SETUP CMAKE_TARGETS
                BUILD missing)
        
include_directories(${PYTHON_INCLUDE_DIR})

file(GLOB_RECURSE LIBKET_SRC libket/src/*.cpp)

add_library(libket STATIC ${LIBKET_SRC})
set_target_properties(libket PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(libket CONAN_PKG::boost 
                             CONAN_PKG::gmp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

find_package(SWIG REQUIRED)
include(UseSWIG)

set_property(SOURCE src/ket.i
             PROPERTY CPLUSPLUS ON)

swig_add_library(ket
                 LANGUAGE python
                 SOURCES src/ket.i)

target_link_libraries(ket libket 
                          -static-libgcc 
                          -static-libstdc++)

install(TARGETS ket DESTINATION ket)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ket.py DESTINATION ket)
