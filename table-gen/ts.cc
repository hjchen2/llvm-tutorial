#include "Ops.h.inc"
#include "Dialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::ts;

using llvm::ArrayRef;
using llvm::makeArrayRef;

class ModuleGen {
public:
  ModuleGen(mlir::MLIRContext& context) : builder(&context) {}

  mlir::ModuleOp genModule() {
    mlir::ModuleOp theModule = mlir::ModuleOp::create(Loc());
    theModule.push_back(genFunc());
    return theModule;
  }

private:
  mlir::Location Loc() {
    return builder.getUnknownLoc();
  }

  mlir::Value genFuncBody() {
    mlir::OperationState state(Loc(), "state");
    ArrayRef<int64_t> shape = {1};
    std::vector<double> data(1);
  
    auto dataType = mlir::RankedTensorType::get(shape, builder.getF64Type());
    auto dataAttribute = mlir::DenseElementsAttr::get(dataType,
                                                      llvm::makeArrayRef(data));
    return builder.create<ConstantOp>(Loc(), dataType, dataAttribute);
  }
  
  mlir::FuncOp genFunc() {
    llvm::SmallVector<mlir::Type, 4> arg_types;
    auto func_type = builder.getFunctionType(arg_types, llvm::None);
    mlir::FuncOp function = mlir::FuncOp::create(Loc(), "test", func_type);
    
    auto &entryBlock = *function.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
  
    genFuncBody();
    // builder.create<ReturnOp>(Loc());
    return function;
  }

private:
  mlir::OpBuilder builder;
};

int main() {
  mlir::MLIRContext context(/*loadAllDialects=*/false);
  context.getOrLoadDialect<mlir::ts::TSDialect>();
  ModuleGen impl(context);

  mlir::ModuleOp module = impl.genModule();
  module.dump();
  return 0;
}
