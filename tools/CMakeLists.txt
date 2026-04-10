add_executable(toy-opt toy-opt.cpp)

mlir_check_link_libraries(toy-opt)

target_link_libraries(toy-opt PRIVATE
  toy-dialect
  toy-passes
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIROptLib
  MLIRSupport
)

target_include_directories(toy-opt PRIVATE
  ${PROJECT_SOURCE_DIR}/include
)
