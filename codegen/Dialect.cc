#include "Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::ts;

TSDialect::TSDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx, TypeID::get<TSDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "Ops.cc.inc"
      >();
}

#define GET_OP_CLASSES
#include "Ops.cc.inc"
