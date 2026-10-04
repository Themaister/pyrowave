cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 99)
project(pyrowave-metal LANGUAGES CXX C OBJCXX)
include(GNUInstallDirs)
include(CTest)

# Note, this should match the version in the main library
set(PYROWAVE_METAL_API_VERSION_MAJOR 0)
set(PYROWAVE_METAL_API_VERSION_MINOR 5)
set(PYROWAVE_METAL_API_VERSION_PATCH 0)
set(PYROWAVE_METAL_API_VERSION ${PYROWAVE_METAL_API_VERSION_MAJOR}.${PYROWAVE_METAL_API_VERSION_MINOR}.${PYROWAVE_METAL_API_VERSION_PATCH})

set(PYROWAVE_CXX_FLAGS -Wshadow -Wall -Wextra -fvisibility=hidden)

# Runtime MSL is embedded in shaders/pyrowave_msl.h by shaders/transpile.sh.
# It is committed, so building does not require glslang or spirv-cross.
set(PYROWAVE_METAL_SOURCES
		pyrowave_metal.h
		pyrowave_common.mm pyrowave_common.hpp
		pyrowave_decoder.mm
		pyrowave_encoder.mm
		pyrowave_bitstream.cpp pyrowave_bitstream.hpp
		shaders/pyrowave_msl.h)

add_library(pyrowave-metal SHARED ${PYROWAVE_METAL_SOURCES})

target_compile_definitions(pyrowave-metal PRIVATE PYROWAVE_EXPORT_SYMBOLS)
target_compile_options(pyrowave-metal PRIVATE ${PYROWAVE_CXX_FLAGS})
target_compile_options(pyrowave-metal PRIVATE
	$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
)
target_link_options(pyrowave-metal PRIVATE "-fobjc-link-runtime")

find_library(IOSURFACE_LIBRARY NAMES IOSurface REQUIRED)
find_library(METAL_LIBRARY NAMES Metal REQUIRED)
target_link_libraries(pyrowave-metal PRIVATE ${IOSURFACE_LIBRARY} ${METAL_LIBRARY})

set_target_properties(pyrowave-metal
		PROPERTIES VERSION ${PYROWAVE_METAL_API_VERSION}
		SOVERSION ${PYROWAVE_METAL_API_VERSION_MAJOR}
		DLL_NAME_WITH_SOVERSION ON)

install(TARGETS pyrowave-metal
		RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
		LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
		ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
		PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pyrowave)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/pyrowave_metal.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pyrowave)

if (BUILD_TESTING)
	add_executable(pyrowave-bitstream-test tests/bitstream_roundtrip.cpp pyrowave_bitstream.cpp)
	target_include_directories(pyrowave-bitstream-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
	target_compile_options(pyrowave-bitstream-test PRIVATE ${PYROWAVE_CXX_FLAGS})
	add_test(NAME pyrowave-bitstream-roundtrip COMMAND pyrowave-bitstream-test)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	option(PYROWAVE_METAL_CLI "Build the native macOS Metal performance CLI." ON)
else()
	option(PYROWAVE_METAL_CLI "Build the native macOS Metal performance CLI." OFF)
endif()

if (PYROWAVE_METAL_CLI)
	if (NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
		message(FATAL_ERROR "The Metal benchmark CLI requires macOS.")
	endif()

	# Keep private timing hooks out of the installed shared library/API. The CLI
	# links its own static backend, so it can also run without a dylib search path.
	add_library(pyrowave-metal-bench-backend STATIC ${PYROWAVE_METAL_SOURCES}
		pyrowave_bench.h experimental_dequant.msl.h experimental_idwt.msl.h
		experimental_native_dequant.msl.h experimental_native_idwt.msl.h
		experimental_fused_idwt.msl.h experimental_render.msl.h)
	target_compile_definitions(pyrowave-metal-bench-backend PRIVATE PYROWAVE_METAL_BENCH_HOOKS)
	target_compile_options(pyrowave-metal-bench-backend PRIVATE ${PYROWAVE_CXX_FLAGS}
		$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
	target_link_libraries(pyrowave-metal-bench-backend PRIVATE ${IOSURFACE_LIBRARY} ${METAL_LIBRARY})

	find_library(FOUNDATION_LIBRARY NAMES Foundation REQUIRED)
	add_executable(pyrowave-metal-bench bench.mm)
	target_compile_options(pyrowave-metal-bench PRIVATE ${PYROWAVE_CXX_FLAGS}
		$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
	target_link_options(pyrowave-metal-bench PRIVATE "-fobjc-link-runtime")
	target_link_libraries(pyrowave-metal-bench PRIVATE pyrowave-metal-bench-backend ${FOUNDATION_LIBRARY})
	install(TARGETS pyrowave-metal-bench RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
