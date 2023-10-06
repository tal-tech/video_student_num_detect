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



#include <ATen/ops/round_ops.h>

namespace at {


// aten::round(Tensor self) -> Tensor
inline at::Tensor round(const at::Tensor & self) {
    return at::_ops::round::call(self);
}

// aten::round_(Tensor(a!) self) -> Tensor(a!)
inline at::Tensor & round_(at::Tensor & self) {
    return at::_ops::round_::call(self);
}

// aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & round_out(at::Tensor & out, const at::Tensor & self) {
    return at::_ops::round_out::call(self, out);
}

// aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & round_outf(const at::Tensor & self, at::Tensor & out) {
    return at::_ops::round_out::call(self, out);
}

// aten::round.decimals(Tensor self, *, int decimals) -> Tensor
inline at::Tensor round(const at::Tensor & self, int64_t decimals) {
    return at::_ops::round_decimals::call(self, decimals);
}

// aten::round_.decimals(Tensor(a!) self, *, int decimals) -> Tensor(a!)
inline at::Tensor & round_(at::Tensor & self, int64_t decimals) {
    return at::_ops::round__decimals::call(self, decimals);
}

// aten::round.decimals_out(Tensor self, *, int decimals, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & round_out(at::Tensor & out, const at::Tensor & self, int64_t decimals) {
    return at::_ops::round_decimals_out::call(self, decimals, out);
}

// aten::round.decimals_out(Tensor self, *, int decimals, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & round_outf(const at::Tensor & self, int64_t decimals, at::Tensor & out) {
    return at::_ops::round_decimals_out::call(self, decimals, out);
}

}
