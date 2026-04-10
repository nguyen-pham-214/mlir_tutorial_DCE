#include "mlir/IR/DialectRegistry.h"
#include "toy/ToyDialect.h"

using namespace mlir;
using namespace toy;

void toy::registerToyDialect(DialectRegistry &registry) {
  registry.insert<ToyDialect>();
}
