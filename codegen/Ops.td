#ifndef TS_OPS
#define TS_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TS_Dialect : Dialect {
  let name = "ts";
  let cppNamespace = "::mlir::ts";
}

class TS_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TS_Dialect, mnemonic, traits>;

def ConstantOp : TS_Op<"constant"> {
  let summary = "constant operation";

  let arguments = (ins F64ElementsAttr:$value);

  let results = (outs F64Tensor);
}

def ReturnOp : TS_Op<"return", [NoSideEffect, HasParent<"FuncOp">,
                                Terminator]> {
  let summary = "return operation";

  let arguments = (ins Variadic<F64Tensor>:$input);

  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";
}

#endif  // TS_OPS
