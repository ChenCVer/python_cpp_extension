// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

at::Tensor affine_cuda_forward(const at::Tensor& input,          /*[B, C, H, W]*/
                               const at::Tensor& affine_matrix,  /*[B, 2, 3]*/
                               const int out_h,
                               const int out_w);

at::Tensor affine_gpu(const at::Tensor& input,          /*[B, C, H, W]*/
                      const at::Tensor& affine_matrix,  /*[B, 2, 3]*/
                      const int out_h,
                      const int out_w)
{
    CHECK_INPUT(input);
    CHECK_INPUT(affine_matrix);

    // Ensure CUDA uses the input tensor device.
    at::DeviceGuard guard(input.device());

    return affine_cuda_forward(input, affine_matrix, out_h, out_w);
}