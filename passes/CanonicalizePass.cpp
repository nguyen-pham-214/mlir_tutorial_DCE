#include "ToyCanonicalizePass.h"

#include "toy/ToyOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace toy;

namespace {

struct ToyCanonicalizePass
    : public PassWrapper<ToyCanonicalizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyCanonicalizePass)

  llvm::StringRef getArgument() const final { return "toy-canonicalize"; }
  llvm::StringRef getDescription() const final {
    return "Canonicalize Toy ops";
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    
    // Create empty pattern set (patterns would be added here)
    RewritePatternSet patterns(&getContext());
    
    // TODO: Add canonicalization patterns here
    // patterns.add<SomePattern>(&getContext());
    
    // Apply patterns greedily
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}

std::unique_ptr<Pass> toy::createToyCanonicalizePass() {
  return std::make_unique<ToyCanonicalizePass>();
}

void toy::detail::registerToyCanonicalizePass() {
  static PassRegistration<ToyCanonicalizePass> canonicalizeReg;
  (void)canonicalizeReg;
}
