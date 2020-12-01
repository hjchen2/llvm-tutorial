#ifndef MLIR_TUTORIAL_TS_PASSES_H
#define MLIR_TUTORIAL_TS_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace ts {
/// Create a pass for lowering to operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<mlir::Pass> createLowerToAffinePass();

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

} // end namespace ts
} // end namespace mlir

#endif // MLIR_TUTORIAL_TS_PASSES_H
