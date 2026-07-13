cmake_minimum_required(VERSION 3.15)
project(Lancius VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler Flags
add_compile_options(-Wall -Wextra -O3 -march=native -mavx2 -mfma)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Source Files
file(GLOB_RECURSE LANCIUS_SRCS
    "src/core/*.c"
    "src/ir/*.c"
    "src/math/*.c"
    "src/runtime/*.c"
    "src/compiler/*.c"
)

# Static Library (For internal examples)
add_library(lancius_static STATIC ${LANCIUS_SRCS})
target_include_directories(lancius_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lancius_static PUBLIC OpenMP::OpenMP_C m pthread)
set_target_properties(lancius_static PROPERTIES OUTPUT_NAME lancius)

# Shared Library (For V1.0 FFI / Python / Rust bindings)
add_library(lancius_shared SHARED ${LANCIUS_SRCS})
target_include_directories(lancius_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lancius_shared PUBLIC OpenMP::OpenMP_C m pthread)
set_target_properties(lancius_shared PROPERTIES OUTPUT_NAME lancius)

# Export visibility
set_target_properties(lancius_shared PROPERTIES C_VISIBILITY_PRESET hidden)
target_compile_options(lancius_shared PRIVATE -fvisibility=hidden)

message(STATUS "Lancius V1.0 Build System Configured.")
