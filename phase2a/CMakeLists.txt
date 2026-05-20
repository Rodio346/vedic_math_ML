cmake_minimum_required(VERSION 3.14)
project(phase2a LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/O2 /W3)
else()
  add_compile_options(-O2 -Wall -Wextra)
endif()

include(FetchContent)
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

set(PHASE2A_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/src)

function(phase2a_exe name source)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE ${PHASE2A_INCLUDE})
  if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(${name} PRIVATE nlohmann_json::nlohmann_json)
  endif()
endfunction()

phase2a_exe(test_digits tests/test_digits.cpp)
phase2a_exe(test_algorithms tests/test_algorithms.cpp)
phase2a_exe(verify tests/verify_counters.cpp)

add_custom_target(
  run_tests
  COMMAND test_digits
  COMMAND test_algorithms
  COMMAND verify
  DEPENDS test_digits test_algorithms verify
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Running Phase 2A unit tests and counter parity"
)

phase2a_exe(benchmark src/benchmark.cpp)
# This enforces that all three test binaries must build AND execute successfully
# before the benchmark binary is built. run_tests is a custom target whose
# COMMAND steps exit non-zero on failure, which causes CMake to abort the build.
# This guarantee holds as long as test commands are not wrapped in a shell that
# swallows exit codes (e.g. do not add '|| true' or similar).
add_dependencies(benchmark run_tests)

enable_testing()
add_test(NAME phase2a_run_tests COMMAND run_tests)
