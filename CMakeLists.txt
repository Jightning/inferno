cmake_minimum_required(VERSION 3.20)
project(inferno CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "CMAKE_BUILD_TYPE must be set (Debug or Release)")
endif()

add_compile_options(-Wall -Wextra -Werror)

# Debug: sanitizers catch UB the moment it happens instead of corrupting logits.
# Release: full optimization for benchmarking; -march=native enables AVX2/FMA.
# CI runners are heterogeneous, so they build with INFERNO_NATIVE=OFF.
option(INFERNO_NATIVE "Compile for the host CPU (-march=native)" ON)
set(SANITIZE_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer)
add_compile_options(
    "$<$<CONFIG:Debug>:${SANITIZE_FLAGS}>"
    "$<$<CONFIG:Release>:-O3>"
    "$<$<AND:$<CONFIG:Release>,$<BOOL:${INFERNO_NATIVE}>>:-march=native>"
)
add_link_options("$<$<CONFIG:Debug>:${SANITIZE_FLAGS}>")

set(THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/third_party)
set(CATCH2_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/catch2)

configure_file(
    ${THIRD_PARTY_DIR}/catch2/catch_user_config.hpp.in
    ${CATCH2_GENERATED_DIR}/catch_user_config.hpp
)

# Phase 1+ sources get appended here as components land (loader/, model/, ...).
set(INFERNO_SOURCES
    # src
    src/main.cpp
    src/loader/config.cpp
    src/loader/mmap_file.cpp
)
add_executable(inferno ${INFERNO_SOURCES})
target_include_directories(inferno SYSTEM PRIVATE ${THIRD_PARTY_DIR} ${CMAKE_CURRENT_BINARY_DIR})

enable_testing()
add_executable(inferno_tests
    tests/test_smoke.cpp
    tests/loader/config.cpp
    tests/loader/mmap_file.cpp
)
target_include_directories(inferno_tests PRIVATE src)
target_include_directories(inferno_tests SYSTEM PRIVATE ${THIRD_PARTY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME inferno_tests COMMAND inferno_tests)