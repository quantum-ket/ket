cmake_minimum_required(VERSION 3.15)
project(ketpy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
  message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
  file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/v0.16.1/conan.cmake"
                "${CMAKE_BINARY_DIR}/conan.cmake"
                EXPECTED_HASH SHA256=396e16d0f5eabdc6a14afddbcfff62a54a7ee75c6da23f32f7a31bc85db23484
                TLS_VERIFY ON)
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

conan_cmake_run(REQUIRES boost/1.71.0
                         gmp/6.2.1
                BASIC_SETUP CMAKE_TARGETS
                BUILD missing)
        
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${CMAKE_BINARY_DIR}/libssh/include/)


file(GLOB_RECURSE LIBKET_SRC libket/src/*.cpp)

set(BUILD_STATIC_LIB ON)
add_subdirectory(libssh EXCLUDE_FROM_ALL)

add_library(libket STATIC ${LIBKET_SRC})
set_target_properties(libket PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(libket CONAN_PKG::boost 
                             CONAN_PKG::gmp
                             ssh-static)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

find_package(SWIG REQUIRED)
include(UseSWIG)

set_property(SOURCE src/ket.i
             PROPERTY CPLUSPLUS ON)

swig_add_library(ket
                 LANGUAGE python
                 SOURCES src/ket.i 
                         src/build_info.cpp)
target_compile_definitions(ket PRIVATE KET_BUILD_INFO=${KET_BUILD_INFO})

target_link_libraries(ket libket 
                          -static-libgcc 
                          -static-libstdc++)

install(TARGETS ket DESTINATION ket)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ket.py DESTINATION ket)
