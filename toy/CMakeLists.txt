set(TOY_TBLGEN_EXTRA_ARGS)
if(MLIR_MAIN_INCLUDE_DIR)
  list(APPEND TOY_TBLGEN_EXTRA_ARGS "-I${MLIR_MAIN_INCLUDE_DIR}")
endif()
foreach(inc ${MLIR_INCLUDE_DIRS})
  if(inc)
    list(APPEND TOY_TBLGEN_EXTRA_ARGS "-I${inc}")
  endif()
endforeach()
foreach(inc ${LLVM_INCLUDE_DIRS})
  if(inc)
    list(APPEND TOY_TBLGEN_EXTRA_ARGS "-I${inc}")
  endif()
endforeach()

set(LLVM_TARGET_DEFINITIONS ToyOps.td)
mlir_tablegen(ToyOps.h.inc -gen-op-decls ${TOY_TBLGEN_EXTRA_ARGS})
mlir_tablegen(ToyOps.cpp.inc -gen-op-defs ${TOY_TBLGEN_EXTRA_ARGS})
mlir_tablegen(ToyDialect.h.inc -gen-dialect-decls -dialect=toy ${TOY_TBLGEN_EXTRA_ARGS})
mlir_tablegen(ToyDialect.cpp.inc -gen-dialect-defs -dialect=toy ${TOY_TBLGEN_EXTRA_ARGS})
add_public_tablegen_target(ToyOpsIncGen)
add_public_tablegen_target(ToyDialectIncGen)

add_mlir_dialect_library(toy-dialect
  Ops.cpp
  DialectRegistration.cpp
  DEPENDS
  ToyOpsIncGen
  ToyDialectIncGen
  LINK_LIBS PUBLIC
  MLIRIR
)

target_include_directories(toy-dialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)
