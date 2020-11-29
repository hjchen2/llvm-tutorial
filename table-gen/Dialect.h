#ifndef MLIR_TS_DIALECT_H_
#define MLIR_TS_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace ts {

class TSDialect : public mlir::Dialect {
public:
  explicit TSDialect(mlir::MLIRContext *ctx);

  /// Provide a utility accessor to the dialect namespace. This is used by
  /// several utilities for casting between dialects.
  static llvm::StringRef getDialectNamespace() { return "ts"; }
};

} // namespace ts
} // namespace mlir

#define GET_OP_CLASSES
#include "Ops.h.inc"

#endif // MLIR_TS_DIALECT_H_

