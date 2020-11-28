cmake_minimum_required(VERSION 3.1)

project(table-gen)

set(LLVM_ROOT /home/chenhoujiang/nmt/llvm-project)
set(LLVM_INCLUDE_DIR
    ${LLVM_ROOT}/llvm/include
    ${LLVM_ROOT}/build/include)
set(LLVM_LIBRARIES
    ${LLVM_ROOT}/build/lib/libLLVMSupport.a
)
set(MLIR_LIBRARIES
    ${LLVM_ROOT}/build/lib/libMLIRAnalysis.a
    ${LLVM_ROOT}/build/lib/libMLIRIR.a
    ${LLVM_ROOT}/build/lib/libMLIRParser.a
)

include_directories(${LLVM_ROOT}/mlir/include)
set(CMAKE_CXX_FLAGS "-std=c++14")

include(${LLVM_ROOT}/llvm/cmake/modules/TableGen.cmake)
function(mlir_tablegen ofn)
  tablegen(MLIR ${ARGV})
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)
endfunction()

set(MLIR_TABLEGEN_EXE ${LLVM_ROOT}/build/bin/mlir-tblgen)
set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cc.inc -gen-op-defs)
add_public_tablegen_target(TSOpsIncGen)

add_executable(ts_ops ts.cc)
add_dependencies(ts_ops TSOpsIncGen)
target_link_libraries(ts_ops ${LLVM_LIBRARIES} ${MLIR_LIBRARIES})
