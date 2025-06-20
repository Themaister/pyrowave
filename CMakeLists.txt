cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 99)
project(pyrowave LANGUAGES CXX C)

if (CMAKE_COMPILER_IS_GNUCXX OR (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang"))
    set(PYROWAVE_CXX_FLAGS -Wshadow -Wall -Wextra -Wno-comment -Wno-missing-field-initializers -Wno-empty-body -fvisibility=hidden)
    if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
        set(PYROWAVE_CXX_FLAGS ${PYROWAVE_CXX_FLAGS} -Wno-backslash-newline-escape)
    endif()
    if (NOT (${CMAKE_BUILD_TYPE} MATCHES "Release"))
        message("Enabling frame pointer for profiling/debug.")
        set(PYROWAVE_CXX_FLAGS ${PYROWAVE_CXX_FLAGS} -fno-omit-frame-pointer)
    endif()
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
        message("Enabling SSE3 support.")
        set(PYROWAVE_CXX_FLAGS ${PYROWAVE_CXX_FLAGS} -msse3)
    endif()
elseif (MSVC)
    set(PYROWAVE_CXX_FLAGS /D_CRT_SECURE_NO_WARNINGS /wd4267 /wd4244 /wd4309 /wd4005 /MP)
endif()

add_library(pyrowave STATIC
        pyrowave_config.hpp shaders/slangmosh.hpp
        pyrowave_encoder.hpp pyrowave_encoder.cpp
        pyrowave_decoder.hpp pyrowave_decoder.cpp
        pyrowave_common.hpp pyrowave_common.cpp)

target_include_directories(pyrowave PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(pyrowave PRIVATE ${PYROWAVE_CXX_FLAGS})

option(PYROWAVE_STANDALONE "Build PyroWave as a standalone sandbox." ON)

option(PYROWAVE_FP32_STORAGE "Configure for full FP32 and extended precision." OFF)
option(PYROWAVE_FP32_MATH "Configure for FP32 arithmetic, but reduced range storage." ON)
if (PYROWAVE_FP32_STORAGE)
    target_compile_definitions(pyrowave PRIVATE PYROWAVE_PRECISION=2)
elseif(PYROWAVE_FP32_MATH)
    target_compile_definitions(pyrowave PRIVATE PYROWAVE_PRECISION=1)
else()
    target_compile_definitions(pyrowave PRIVATE PYROWAVE_PRECISION=0)
endif()

if (PYROWAVE_STANDALONE)
    add_library(pyrowave-utils STATIC yuv4mpeg.cpp yuv4mpeg.hpp)

    # Standalone build, build the sandbox.
    set(GRANITE_RENDERER ON CACHE BOOL "" FORCE)
    set(GRANITE_VULKAN_FOSSILIZE OFF CACHE BOOL "" FORCE)
    set(GRANITE_SHIPPING OFF CACHE BOOL "" FORCE)
    set(GRANITE_PLATFORM "SDL" CACHE STRING "" FORCE)
    add_subdirectory(Granite EXCLUDE_FROM_ALL)

    add_granite_offline_tool(pyrowave-sandbox sandbox.cpp)
    target_link_libraries(pyrowave-sandbox PRIVATE pyrowave)
    target_compile_definitions(pyrowave-sandbox PRIVATE ASSET_DIRECTORY=\"${CMAKE_CURRENT_SOURCE_DIR}/shaders\")
    target_link_libraries(pyrowave-sandbox PRIVATE granite-renderer pyrowave-utils)

    add_granite_offline_tool(pyrowave-bench bench.cpp)
    target_link_libraries(pyrowave-bench PRIVATE pyrowave)
    target_link_libraries(pyrowave-bench PRIVATE granite-vulkan pyrowave-utils)

    add_granite_offline_tool(pyrowave-psnr psnr.cpp)
    target_link_libraries(pyrowave-psnr PRIVATE pyrowave-utils)

    add_granite_application(pyrowave-viewer viewer.cpp)
    target_link_libraries(pyrowave-viewer PRIVATE pyrowave pyrowave-utils)
    target_compile_definitions(pyrowave-viewer PRIVATE ASSET_DIRECTORY=\"${CMAKE_CURRENT_SOURCE_DIR}/shaders\")
endif()

target_link_libraries(pyrowave PRIVATE granite-vulkan granite-math)
