cmake_minimum_required(VERSION 3.20)
project(mlir_tutorial LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

add_subdirectory(toy)
add_subdirectory(passes)
add_subdirectory(tools)

add_custom_target(mlir_tutorial ALL DEPENDS toy-dialect toy-passes)

install(TARGETS toy-dialect toy-passes DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
