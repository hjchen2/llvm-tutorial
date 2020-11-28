#ifndef TS_OPS
#define TS_OPS

include "mlir/IR/OpBase.td"

def TS_Dialect : Dialect {
  let name = "ts";
  let cppNamespace = "::mlir::ts";
}

class TS_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TS_Dialect, mnemonic, traits>;

def ConstantOp : TS_Op<"constant"> {
  let summary = "constant";

  let arguments = (ins F64ElementsAttr:$value);

  let results = (outs F64Tensor);
}

#endif  // TS_OPS
