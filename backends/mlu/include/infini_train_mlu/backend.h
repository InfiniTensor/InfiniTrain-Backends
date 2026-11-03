#pragma once

namespace infini_train::mlu {

// Registers MLU as the process-wide PrivateUse1 provider. Device runtime
// initialization remains lazy until the first device operation.
void RegisterBackend();

} // namespace infini_train::mlu
