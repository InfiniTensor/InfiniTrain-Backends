#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::mlu {

// Pure host-side views (Tensor(Tensor, offset, dims) aliases the storage), so
// the MLU implementation is identical to the CPU/MACA ones - no device work.
std::shared_ptr<Tensor> NoOpForward(const std::shared_ptr<Tensor> &input, const std::vector<int64_t> &dims) {
    const int64_t num_elements = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
    CHECK_EQ(input->NumElements(), num_elements);

    auto output = std::make_shared<Tensor>(*input, 0, dims);
    return output;
}

std::shared_ptr<Tensor> NoOpBackward(const std::vector<int64_t> &dims, const std::shared_ptr<Tensor> &grad_output) {
    auto num_elements = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
    CHECK_EQ(num_elements, grad_output->NumElements());

    auto grad_input = std::make_shared<Tensor>(*grad_output, 0, dims);
    return grad_input;
}
} // namespace infini_train::kernels::mlu

namespace infini_train::kernels::mlu {

REGISTER_KERNEL(infini_train::Device::DeviceType::kPrivateUse1, NoOpForward, infini_train::kernels::mlu::NoOpForward)
REGISTER_KERNEL(infini_train::Device::DeviceType::kPrivateUse1, NoOpBackward, infini_train::kernels::mlu::NoOpBackward)

} // namespace infini_train::kernels::mlu
