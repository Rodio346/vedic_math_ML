cmake_minimum_required(VERSION 3.18)
project(phase3 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O2>)
else()
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O2>)
endif()

include(FetchContent)
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

set(PHASE3_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels
  ${CMAKE_CURRENT_SOURCE_DIR}/host
)

function(phase3_target name source)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE ${PHASE3_INCLUDE_DIRS})
  target_link_libraries(${name} PRIVATE nlohmann_json::nlohmann_json)
endfunction()

phase3_target(validator host/validator.cu)
phase3_target(benchmark host/benchmark.cu)

set(PAIRS_JSON ${CMAKE_CURRENT_SOURCE_DIR}/../pairs.json)

add_custom_target(
  run_validator
  COMMAND $<TARGET_FILE:validator> --input ${PAIRS_JSON}
  DEPENDS validator
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running correctness validation — must pass before benchmark builds"
)

add_dependencies(benchmark run_validator)

enable_testing()
add_test(NAME phase3_run_validator COMMAND run_validator)
