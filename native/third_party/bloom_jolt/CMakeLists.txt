cmake_minimum_required(VERSION 3.16)
project(bloom_jolt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# JoltPhysics build options.
# Matches the "Distribution" flavour: no asserts, no profile, no debug renderer,
# interprocedural optimisation on. We keep object stream off — we don't use
# Jolt's scene serialisation (Bloom handles save/load itself).
# ---------------------------------------------------------------------------
set(INTERPROCEDURAL_OPTIMIZATION        ON  CACHE BOOL "" FORCE)
set(USE_STATIC_MSVC_RUNTIME_LIBRARY     OFF CACHE BOOL "" FORCE)
set(TARGET_UNIT_TESTS                   OFF CACHE BOOL "" FORCE)
set(TARGET_HELLO_WORLD                  OFF CACHE BOOL "" FORCE)
set(TARGET_PERFORMANCE_TEST             OFF CACHE BOOL "" FORCE)
set(TARGET_SAMPLES                      OFF CACHE BOOL "" FORCE)
set(TARGET_VIEWER                       OFF CACHE BOOL "" FORCE)
set(OBJECT_LAYER_BITS                   16  CACHE STRING "" FORCE)
set(USE_ASSERTS                         OFF CACHE BOOL "" FORCE)
set(PROFILE_ENABLED                     OFF CACHE BOOL "" FORCE)
set(DEBUG_RENDERER_IN_DEBUG_AND_RELEASE OFF CACHE BOOL "" FORCE)
set(CROSS_PLATFORM_DETERMINISTIC        OFF CACHE BOOL "" FORCE)
set(FLOATING_POINT_EXCEPTIONS_ENABLED   OFF CACHE BOOL "" FORCE)
set(DOUBLE_PRECISION                    OFF CACHE BOOL "" FORCE)
set(GENERATE_DEBUG_SYMBOLS              OFF CACHE BOOL "" FORCE)
set(OVERRIDE_CXX_FLAGS                  OFF CACHE BOOL "" FORCE)
set(ENABLE_OBJECT_STREAM                OFF CACHE BOOL "" FORCE)
set(ENABLE_ALL_WARNINGS                 OFF CACHE BOOL "" FORCE)

# ---------------------------------------------------------------------------
# Pull in JoltPhysics — the real CMakeLists lives in the sibling submodule.
# JOLT_PHYSICS_ROOT can be overridden by the caller (build.rs).
# ---------------------------------------------------------------------------
if(NOT DEFINED JOLT_PHYSICS_ROOT)
    set(JOLT_PHYSICS_ROOT "${CMAKE_CURRENT_LIST_DIR}/../JoltPhysics")
endif()

if(NOT EXISTS "${JOLT_PHYSICS_ROOT}/Build/CMakeLists.txt")
    message(FATAL_ERROR
        "JoltPhysics submodule not found at ${JOLT_PHYSICS_ROOT}.\n"
        "Run: git submodule update --init --recursive")
endif()

add_subdirectory("${JOLT_PHYSICS_ROOT}/Build" JoltBuild EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# bloom_jolt — the thin C shim Rust links against.
# ---------------------------------------------------------------------------
add_library(bloom_jolt STATIC
    src/bloom_jolt.cpp
)

target_include_directories(bloom_jolt
    PUBLIC  include
    PRIVATE "${JOLT_PHYSICS_ROOT}"
)

target_link_libraries(bloom_jolt PUBLIC Jolt)
target_compile_features(bloom_jolt PUBLIC cxx_std_17)

# Hide C++ symbols from the static archive — only the extern "C" API is exported.
set_target_properties(bloom_jolt PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Release defaults for the shim itself — Jolt pulls its own flags.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

install(TARGETS bloom_jolt ARCHIVE DESTINATION lib)
install(TARGETS Jolt       ARCHIVE DESTINATION lib)
install(FILES include/bloom_jolt.h DESTINATION include)
