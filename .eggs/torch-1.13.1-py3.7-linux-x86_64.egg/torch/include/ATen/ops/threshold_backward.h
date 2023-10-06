#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/threshold_backward_ops.h>

namespace at {


// aten::threshold_backward.grad_input(Tensor grad_output, Tensor self, Scalar threshold, *, Tensor(a!) grad_input) -> Tensor(a!)
inline at::Tensor & threshold_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold) {
    return at::_ops::threshold_backward_grad_input::call(grad_output, self, threshold, grad_input);
}

// aten::threshold_backward.grad_input(Tensor grad_output, Tensor self, Scalar threshold, *, Tensor(a!) grad_input) -> Tensor(a!)
inline at::Tensor & threshold_backward_outf(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold, at::Tensor & grad_input) {
    return at::_ops::threshold_backward_grad_input::call(grad_output, self, threshold, grad_input);
}

// aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor
inline at::Tensor threshold_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold) {
    return at::_ops::threshold_backward::call(grad_output, self, threshold);
}

}
