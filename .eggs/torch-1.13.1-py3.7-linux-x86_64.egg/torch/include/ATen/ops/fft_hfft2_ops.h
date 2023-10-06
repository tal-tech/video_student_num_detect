#pragma once

// @generated by torchgen/gen.py from Operator.h

#include <tuple>
#include <vector>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {
namespace _ops {


struct TORCH_API fft_hfft2 {
  using schema = at::Tensor (const at::Tensor &, at::OptionalIntArrayRef, at::IntArrayRef, c10::optional<c10::string_view>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::fft_hfft2")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "fft_hfft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor")
  static at::Tensor call(const at::Tensor & self, at::OptionalIntArrayRef s, at::IntArrayRef dim, c10::optional<c10::string_view> norm);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::OptionalIntArrayRef s, at::IntArrayRef dim, c10::optional<c10::string_view> norm);
};

struct TORCH_API fft_hfft2_out {
  using schema = const at::Tensor & (const at::Tensor &, at::OptionalIntArrayRef, at::IntArrayRef, c10::optional<c10::string_view>, const at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::fft_hfft2")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "out")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "fft_hfft2.out(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)")
  static const at::Tensor & call(const at::Tensor & self, at::OptionalIntArrayRef s, at::IntArrayRef dim, c10::optional<c10::string_view> norm, const at::Tensor & out);
  static const at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::OptionalIntArrayRef s, at::IntArrayRef dim, c10::optional<c10::string_view> norm, const at::Tensor & out);
};

}} // namespace at::_ops
