cmake_minimum_required(VERSION 3.1)

project(table-gen)

set(LLVM_ROOT /home/chenhoujiang/nmt/llvm-project)
set(LLVM_INCLUDE_DIR
    ${LLVM_ROOT}/llvm/include
    ${LLVM_ROOT}/build/include)
set(LLVM_LIBRARIES
    ${LLVM_ROOT}/build/lib/libLLVMCore.a
    ${LLVM_ROOT}/build/lib/libLLVMSupport.a
    ${LLVM_ROOT}/build/lib/libLLVMX86CodeGen.a
    ${LLVM_ROOT}/build/lib/libLLVMX86Desc.a
    ${LLVM_ROOT}/build/lib/libLLVMX86Info.a
    ${LLVM_ROOT}/build/lib/libLLVMOrcJIT.a
    ${LLVM_ROOT}/build/lib/libLLVMX86CodeGen.a
    ${LLVM_ROOT}/build/lib/libLLVMAsmPrinter.a
    ${LLVM_ROOT}/build/lib/libLLVMDebugInfoDWARF.a
    ${LLVM_ROOT}/build/lib/libLLVMCFGuard.a
    ${LLVM_ROOT}/build/lib/libLLVMGlobalISel.a
    ${LLVM_ROOT}/build/lib/libLLVMSelectionDAG.a
    ${LLVM_ROOT}/build/lib/libLLVMCodeGen.a
    ${LLVM_ROOT}/build/lib/libLLVMX86Desc.a
    ${LLVM_ROOT}/build/lib/libLLVMMCDisassembler.a
    ${LLVM_ROOT}/build/lib/libLLVMX86Info.a
    ${LLVM_ROOT}/build/lib/libLLVMOrcJIT.a
    ${LLVM_ROOT}/build/lib/libLLVMOrcError.a
    ${LLVM_ROOT}/build/lib/libLLVMPasses.a
    ${LLVM_ROOT}/build/lib/libLLVMCoroutines.a
    ${LLVM_ROOT}/build/lib/libLLVMHelloNew.a
    ${LLVM_ROOT}/build/lib/libLLVMObjCARCOpts.a
    ${LLVM_ROOT}/build/lib/libLLVMExecutionEngine.a
    ${LLVM_ROOT}/build/lib/libLLVMTarget.a
    ${LLVM_ROOT}/build/lib/libLLVMRuntimeDyld.a
    ${LLVM_ROOT}/build/lib/libLLVMJITLink.a
    ${LLVM_ROOT}/build/lib/libLLVMipo.a
    ${LLVM_ROOT}/build/lib/libLLVMBitWriter.a
    ${LLVM_ROOT}/build/lib/libLLVMFrontendOpenMP.a
    ${LLVM_ROOT}/build/lib/libLLVMScalarOpts.a
    ${LLVM_ROOT}/build/lib/libLLVMAggressiveInstCombine.a
    ${LLVM_ROOT}/build/lib/libLLVMInstCombine.a
    ${LLVM_ROOT}/build/lib/libLLVMVectorize.a
    ${LLVM_ROOT}/build/lib/libLLVMIRReader.a
    ${LLVM_ROOT}/build/lib/libLLVMAsmParser.a
    ${LLVM_ROOT}/build/lib/libLLVMInstrumentation.a
    ${LLVM_ROOT}/build/lib/libLLVMLinker.a
    ${LLVM_ROOT}/build/lib/libLLVMTransformUtils.a
    ${LLVM_ROOT}/build/lib/libLLVMAnalysis.a
    ${LLVM_ROOT}/build/lib/libLLVMProfileData.a
    ${LLVM_ROOT}/build/lib/libLLVMObject.a
    ${LLVM_ROOT}/build/lib/libLLVMBitReader.a
    ${LLVM_ROOT}/build/lib/libLLVMCore.a
    ${LLVM_ROOT}/build/lib/libLLVMRemarks.a
    ${LLVM_ROOT}/build/lib/libLLVMBitstreamReader.a
    ${LLVM_ROOT}/build/lib/libLLVMMCParser.a
    ${LLVM_ROOT}/build/lib/libLLVMMC.a
    ${LLVM_ROOT}/build/lib/libLLVMDebugInfoCodeView.a
    ${LLVM_ROOT}/build/lib/libLLVMDebugInfoMSF.a
    ${LLVM_ROOT}/build/lib/libLLVMTextAPI.a
    ${LLVM_ROOT}/build/lib/libLLVMBinaryFormat.a
    ${LLVM_ROOT}/build/lib/libLLVMSupport.a
    ${LLVM_ROOT}/build/lib/libLLVMDemangle.a
)
set(MLIR_LIBRARIES
    ${LLVM_ROOT}/build/lib/libMLIRAffineOps.a
    ${LLVM_ROOT}/build/lib/libMLIRAffineEDSC.a
    ${LLVM_ROOT}/build/lib/libMLIRAffineTransforms.a
    ${LLVM_ROOT}/build/lib/libMLIRAffineUtils.a
    ${LLVM_ROOT}/build/lib/libMLIRAVX512.a
    ${LLVM_ROOT}/build/lib/libMLIRGPU.a
    ${LLVM_ROOT}/build/lib/libMLIRLinalgAnalysis.a
    ${LLVM_ROOT}/build/lib/libMLIRLinalgEDSC.a
    ${LLVM_ROOT}/build/lib/libMLIRLinalgOps.a
    ${LLVM_ROOT}/build/lib/libMLIRLinalgTransforms.a
    ${LLVM_ROOT}/build/lib/libMLIRLinalgUtils.a
    ${LLVM_ROOT}/build/lib/libMLIRLLVMIRTransforms.a
    ${LLVM_ROOT}/build/lib/libMLIRLLVMIR.a
    ${LLVM_ROOT}/build/lib/libMLIRLLVMAVX512.a
    ${LLVM_ROOT}/build/lib/libMLIRNVVMIR.a
    ${LLVM_ROOT}/build/lib/libMLIRROCDLIR.a
    ${LLVM_ROOT}/build/lib/libMLIROpenACC.a
    ${LLVM_ROOT}/build/lib/libMLIROpenMP.a
    ${LLVM_ROOT}/build/lib/libMLIRPDL.a
    ${LLVM_ROOT}/build/lib/libMLIRPDLInterp.a
    ${LLVM_ROOT}/build/lib/libMLIRQuant.a
    ${LLVM_ROOT}/build/lib/libMLIRSCF.a
    ${LLVM_ROOT}/build/lib/libMLIRSCFTransforms.a
    ${LLVM_ROOT}/build/lib/libMLIRSDBM.a
    ${LLVM_ROOT}/build/lib/libMLIRShape.a
    ${LLVM_ROOT}/build/lib/libMLIRShapeOpsTransforms.a
    ${LLVM_ROOT}/build/lib/libMLIRSPIRV.a
    ${LLVM_ROOT}/build/lib/libMLIRSPIRVSerialization.a
    ${LLVM_ROOT}/build/lib/libMLIRSPIRVTransforms.a
    ${LLVM_ROOT}/build/lib/libMLIRStandardOps.a
    ${LLVM_ROOT}/build/lib/libMLIRStandardOpsTransforms.a
    ${LLVM_ROOT}/build/lib/libMLIRVector.a
    ${LLVM_ROOT}/build/lib/libMLIRAffineToStandard.a
    ${LLVM_ROOT}/build/lib/libMLIRAVX512ToLLVM.a
    ${LLVM_ROOT}/build/lib/libMLIRGPUToGPURuntimeTransforms.a
    ${LLVM_ROOT}/build/lib/libMLIRGPUToNVVMTransforms.a
    ${LLVM_ROOT}/build/lib/libMLIRGPUToROCDLTransforms.a
    ${LLVM_ROOT}/build/lib/libMLIRGPUToSPIRVTransforms.a
    ${LLVM_ROOT}/build/lib/libMLIRGPUToVulkanTransforms.a
    ${LLVM_ROOT}/build/lib/libMLIRLinalgToLLVM.a
    ${LLVM_ROOT}/build/lib/libMLIRLinalgToSPIRVTransforms.a
    ${LLVM_ROOT}/build/lib/libMLIRLinalgToStandard.a
    ${LLVM_ROOT}/build/lib/libMLIROpenMPToLLVM.a
    ${LLVM_ROOT}/build/lib/libMLIRSCFToGPU.a
    ${LLVM_ROOT}/build/lib/libMLIRSCFToSPIRV.a
    ${LLVM_ROOT}/build/lib/libMLIRSCFToStandard.a
    ${LLVM_ROOT}/build/lib/libMLIRShapeToStandard.a
    ${LLVM_ROOT}/build/lib/libMLIRSPIRVToLLVM.a
    ${LLVM_ROOT}/build/lib/libMLIRStandardToLLVM.a
    ${LLVM_ROOT}/build/lib/libMLIRStandardToSPIRVTransforms.a
    ${LLVM_ROOT}/build/lib/libMLIRVectorToROCDL.a
    ${LLVM_ROOT}/build/lib/libMLIRVectorToLLVM.a
    ${LLVM_ROOT}/build/lib/libMLIRVectorToSCF.a
    ${LLVM_ROOT}/build/lib/libMLIRAnalysis.a
    ${LLVM_ROOT}/build/lib/libMLIRCallInterfaces.a
    ${LLVM_ROOT}/build/lib/libMLIRExecutionEngine.a
    ${LLVM_ROOT}/build/lib/libMLIRIR.a
    ${LLVM_ROOT}/build/lib/libMLIRLLVMIR.a
    ${LLVM_ROOT}/build/lib/libMLIRParser.a
    ${LLVM_ROOT}/build/lib/libMLIRPass.a
    ${LLVM_ROOT}/build/lib/libMLIRSideEffectInterfaces.a
    ${LLVM_ROOT}/build/lib/libMLIRSupport.a
    ${LLVM_ROOT}/build/lib/libMLIRTargetLLVMIR.a
    ${LLVM_ROOT}/build/lib/libMLIRTransforms.a
    ${LLVM_ROOT}/build/lib/libMLIRLinalgAnalysis.a
    ${LLVM_ROOT}/build/lib/libMLIRPDL.a
    ${LLVM_ROOT}/build/lib/libMLIRAVX512.a
    ${LLVM_ROOT}/build/lib/libMLIRLLVMAVX512.a
    ${LLVM_ROOT}/build/lib/libMLIRNVVMIR.a
    ${LLVM_ROOT}/build/lib/libMLIRROCDLIR.a
    ${LLVM_ROOT}/build/lib/libMLIRSPIRVSerialization.a
    ${LLVM_ROOT}/build/lib/libMLIRVectorToLLVM.a
    ${LLVM_ROOT}/build/lib/libMLIRLinalgUtils.a
    ${LLVM_ROOT}/build/lib/libMLIRLinalgEDSC.a
    ${LLVM_ROOT}/build/lib/libMLIRGPU.a
    ${LLVM_ROOT}/build/lib/libMLIRAffineToStandard.a
    ${LLVM_ROOT}/build/lib/libMLIRShape.a
    ${LLVM_ROOT}/build/lib/libMLIRDialect.a
    ${LLVM_ROOT}/build/lib/libMLIRSPIRV.a
    ${LLVM_ROOT}/build/lib/libMLIRStandardToLLVM.a
    ${LLVM_ROOT}/build/lib/libMLIRTransforms.a
    ${LLVM_ROOT}/build/lib/libMLIRVector.a
    ${LLVM_ROOT}/build/lib/libMLIRAffineEDSC.a
    ${LLVM_ROOT}/build/lib/libMLIRLinalgOps.a
    ${LLVM_ROOT}/build/lib/libMLIRTransformUtils.a
    ${LLVM_ROOT}/build/lib/libMLIRLoopAnalysis.a
    ${LLVM_ROOT}/build/lib/libMLIRCopyOpInterface.a
    ${LLVM_ROOT}/build/lib/libMLIRTargetLLVMIRModuleTranslation.a
    ${LLVM_ROOT}/build/lib/libMLIRLLVMIRTransforms.a
    ${LLVM_ROOT}/build/lib/libMLIRPass.a
    ${LLVM_ROOT}/build/lib/libMLIRAnalysis.a
    ${LLVM_ROOT}/build/lib/libMLIRAffineOps.a
    ${LLVM_ROOT}/build/lib/libMLIRSCF.a
    ${LLVM_ROOT}/build/lib/libMLIRStandardOps.a
    ${LLVM_ROOT}/build/lib/libMLIRViewLikeInterface.a
    ${LLVM_ROOT}/build/lib/libMLIRVectorInterfaces.a
    ${LLVM_ROOT}/build/lib/libMLIREDSC.a
    ${LLVM_ROOT}/build/lib/libMLIRLoopLikeInterface.a
    ${LLVM_ROOT}/build/lib/libMLIRInferTypeOpInterface.a
    ${LLVM_ROOT}/build/lib/libMLIRPresburger.a
    ${LLVM_ROOT}/build/lib/libMLIRLLVMIR.a
    ${LLVM_ROOT}/build/lib/libMLIROpenMP.a
    ${LLVM_ROOT}/build/lib/libMLIRCallInterfaces.a
    ${LLVM_ROOT}/build/lib/libMLIRSideEffectInterfaces.a
    ${LLVM_ROOT}/build/lib/libMLIRControlFlowInterfaces.a
    ${LLVM_ROOT}/build/lib/libMLIRTranslation.a
    ${LLVM_ROOT}/build/lib/libMLIRParser.a
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

file(GLOB TS_SRCS *.cc)
add_executable(ts ${TS_SRCS})
add_dependencies(ts TSOpsIncGen)
target_link_libraries(ts ${MLIR_LIBRARIES} ${LLVM_LIBRARIES} -pthread -ltinfo -lrt -ldl -lm -lz)
