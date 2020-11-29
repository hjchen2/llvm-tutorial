cmake_minimum_required(VERSION 3.1)

project(table-gen)

set(LLVM_ROOT /home/chenhoujiang/nmt/llvm-project)
set(LLVM_INCLUDE_DIR
    ${LLVM_ROOT}/llvm/include
    ${LLVM_ROOT}/build/include)
set(LLVM_LIBRARIES
    ${LLVM_ROOT}/build/lib/libLLVMSupport.a
    ${LLVM_ROOT}/build/lib/libLLVMDemangle.a
)
set(MLIR_LIBRARIES
    ${LLVM_ROOT}/build/lib/libMLIRAnalysis.a
    ${LLVM_ROOT}/build/lib/libMLIRIR.a
    ${LLVM_ROOT}/build/lib/libMLIRParser.a
    ${LLVM_ROOT}/build/lib/libMLIRPass.a
    ${LLVM_ROOT}/build/lib/libMLIRSideEffectInterfaces.a
    ${LLVM_ROOT}/build/lib/libMLIRTransforms.a
    ${LLVM_ROOT}/build/lib/libMLIRCopyOpInterface.a
    ${LLVM_ROOT}/build/lib/libMLIRTransformUtils.a
    ${LLVM_ROOT}/build/lib/libMLIRInferTypeOpInterface.a
    ${LLVM_ROOT}/build/lib/libMLIRVector.a
    ${LLVM_ROOT}/build/lib/libMLIRLinalgOps.a
    ${LLVM_ROOT}/build/lib/libMLIRLoopAnalysis.a
    ${LLVM_ROOT}/build/lib/libMLIRInferTypeOpInterface.a
    ${LLVM_ROOT}/build/lib/libMLIRAffineOps.a
    ${LLVM_ROOT}/build/lib/libMLIRLoopLikeInterface.a
    ${LLVM_ROOT}/build/lib/libMLIRPresburger.a
    ${LLVM_ROOT}/build/lib/libMLIRSCF.a
    ${LLVM_ROOT}/build/lib/libMLIRAffineEDSC.a
    ${LLVM_ROOT}/build/lib/libMLIRCallInterfaces.a
    ${LLVM_ROOT}/build/lib/libMLIRControlFlowInterfaces.a
    ${LLVM_ROOT}/build/lib/libMLIREDSC.a
    ${LLVM_ROOT}/build/lib/libMLIRViewLikeInterface.a
    ${LLVM_ROOT}/build/lib/libMLIRVectorInterfaces.a
    ${LLVM_ROOT}/build/lib/libMLIRStandardOps.a
    ${LLVM_ROOT}/build/lib/libMLIRIR.a
    ${LLVM_ROOT}/build/lib/libMLIRSupport.a
)

include_directories(${LLVM_INCLUDE_DIR}
                    ${LLVM_ROOT}/mlir/include
                    ${LLVM_ROOT}/build/tools/mlir/include
                    ${CMAKE_BINARY_DIR})
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

add_executable(ts_ops ts.cc Dialect.cc)
add_dependencies(ts_ops TSOpsIncGen)
target_link_libraries(ts_ops ${MLIR_LIBRARIES} ${LLVM_LIBRARIES} -pthread -ltinfo -lrt -ldl -lm)
