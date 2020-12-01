#include "Ops.h.inc"
#include "Dialect.h"
#include "Passes.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

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
        mlir::FuncOp::create(Loc(), /*func_name=*/"get_constant_func", func_type);
    
    auto &entryBlock = *function.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
  
    mlir::Value value = genFuncBody();

    // Add return operation.
    // ReturnOp returnOp = builder.create<ReturnOp>(Loc(), value);
    // if (returnOp.getNumOperands() > 0) {
    //   // Fix function result type.
    //   // auto result_type = mlir::UnrankedTensorType::get(builder.getF64Type());
    //   auto result_type = value.getType();
    //   function.setType(builder.getFunctionType(function.getType().getInputs(),
    //                                            result_type));
    // }
    ReturnOp returnOp = builder.create<ReturnOp>(Loc(), llvm::None);
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
    func @get_constant_func() -> tensor<*xf64> {
      %0 = "ts.constant"() {value = dense<0.000000e+00> : tensor<1xf64>} : () -> tensor<1xf64>
      ts.return %0 : tensor<1xf64>
    }
  }
  */
  module.dump();

  mlir::PassManager pm(&context);
  pm.addPass(createLowerToAffinePass());
  pm.addPass(createLowerToLLVMPass());
  pm.run(module);

  module.dump();

  // llvm::LLVMContext llvmContext;
  // auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  // if (!llvmModule) {
  //   llvm::errs() << "Failed to emit LLVM IR\n";
  //   return -1;
  // }
  // llvm::errs() << *llvmModule << "\n";

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*opt_level=*/0, /*size_level=*/0, /*targetMachine=*/nullptr);
  auto maybeEngine = mlir::ExecutionEngine::create(module, optPipeline);
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invoke("get_constant_func");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}
