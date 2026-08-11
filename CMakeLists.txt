cmake_minimum_required(VERSION 3.15)
project(Lancius VERSION 11.1.0 LANGUAGES C)

set(LANCIUS_MILESTONE "v11S")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler Flags
add_compile_options(-Wall -Wextra -O3)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64")
    add_compile_options(-mavx2 -mfma)
endif()

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

message(STATUS "Lancius v11S build configured.")

# --- v11S: Install rules ---
include(GNUInstallDirs)

install(TARGETS lancius_static lancius_shared
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/lancius/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lancius
    FILES_MATCHING PATTERN "*.h"
)

install(FILES include/lancius.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# --- v11S: pkg-config ---
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/lancius.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/lancius.pc
    @ONLY
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/lancius.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)
