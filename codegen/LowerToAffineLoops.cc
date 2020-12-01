#include "Ops.h.inc"
#include "Dialect.h"
#include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;

static MemRefType convertTensorToMemRef(TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return MemRefType::get(type.getShape(), type.getElementType());
}

static Type getTensorTypeFromMemRefType(Type type) {
  if (auto memref = type.dyn_cast<MemRefType>())
    return RankedTensorType::get(memref.getShape(), memref.getElementType());
  if (auto memref = type.dyn_cast<UnrankedMemRefType>())
    return UnrankedTensorType::get(memref.getElementType());
  return NoneType::get(type.getContext());
}

static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc.getOperation()->getBlock();
  alloc.getOperation()->moveBefore(&parentBlock->front());

  // // Make sure to deallocate this alloc at the end of the block. This is fine
  // // as ts functions have no control flow.
  // auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
  // dealloc.getOperation()->moveBefore(&parentBlock->back());
  return alloc;
}

struct ConstantOpLowering : public OpRewritePattern<ts::ConstantOp> {
  using OpRewritePattern<ts::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ts::ConstantOp op,
                                PatternRewriter &rewriter) const final {
    DenseElementsAttr constantValue = op.value();
    Location loc = op.getLoc();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = op.getType().cast<TensorType>();
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(
              0, *std::max_element(valueShape.begin(), valueShape.end())))
       constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, i));
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
    }
    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.getValues<FloatAttr>().begin();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<AffineStoreOp>(
            loc, rewriter.create<ConstantOp>(loc, *valueIt++), alloc,
            llvm::makeArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

struct ReturnOpLowering : public OpRewritePattern<ts::ReturnOp> {
  using OpRewritePattern<ts::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ts::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    ValueRange input;
    if (op.getNumOperands() > 0) {
      ts::ReturnOpAdaptor adaptor(op);
      input = adaptor.input();
    }
    std::vector<Value> values;
    for (auto it = input.begin(); it != input.end(); ++it) {
      Value value = rewriter.create<TensorLoadOp>(op.getLoc(), *it);
      auto dataType = mlir::RankedTensorType::get({1}, rewriter.getF64Type());
      value.setType(dataType);
      values.push_back(value);
    }
    // We lower "ts.return" directly to "std.return".
    rewriter.replaceOpWithNewOp<ReturnOp>(op, ValueRange(values));
    return success();
  }
};

class TSToAffineLoweringPass
    : public PassWrapper<TSToAffineLoweringPass, FunctionPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, StandardOpsDialect>();
  }
  void runOnFunction() final;
};

void TSToAffineLoweringPass::runOnFunction() {
  auto function = getFunction();

  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, StandardOpsDialect>();

  target.addIllegalDialect<ts::TSDialect>();
  // target.addLegalOp<ts::PrintOp>();
  OwningRewritePatternList patterns;
  patterns.insert<ConstantOpLowering, ReturnOpLowering>(&getContext());
  applyPartialConversion(getFunction(), target, patterns);
}

std::unique_ptr<Pass> mlir::ts::createLowerToAffinePass() {
  return std::make_unique<TSToAffineLoweringPass>();
}
