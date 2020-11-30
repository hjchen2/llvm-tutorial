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
    ArrayRef<int64_t> shape = {1};
    std::vector<double> data(1);
    auto dataType = mlir::RankedTensorType::get(shape, builder.getF64Type());
    auto dataAttribute = mlir::DenseElementsAttr::get(dataType,
                                                      llvm::makeArrayRef(data));
    return builder.create<ConstantOp>(Loc(), /*element_type=*/dataType,
                                      /*value=*/dataAttribute);
  }
  
  mlir::FuncOp genFunc() {
    // llvm::SmallVector<mlir::Type, 4> arg_types;
    // auto func_type = builder.getFunctionType(arg_types, llvm::None);
    auto func_type = builder.getFunctionType(/*input_types=*/llvm::None,
                                             /*result_type=*/llvm::None);
    mlir::FuncOp function =
        mlir::FuncOp::create(Loc(), /*func_name=*/"test", func_type);
    
    auto &entryBlock = *function.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
  
    genFuncBody();

    // Fix function result type.
    auto result_type = mlir::UnrankedTensorType::get(builder.getF64Type());
    function.setType(builder.getFunctionType(function.getType().getInputs(),
                                             result_type));
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
  /*
  module {
  func @test() -> tensor<*xf64> {
    %0 = "ts.constant"() {value = dense<0.000000e+00> : tensor<1xf64>} : () -> tensor<1xf64>
  }
  */
  module.dump();
  return 0;
}
