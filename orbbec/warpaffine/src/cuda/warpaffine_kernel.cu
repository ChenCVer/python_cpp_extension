// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
// /* pytorch: 1.5.0 ~ 1.10.x */
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/DeviceGuard.h>

#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>
// /* ---------------------- */

// /* pytorch: 1.11.0 ~ latest */
// #include <cuda.h>
// #include <ATen/ATen.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>

// #include <ATen/cuda/CUDAApplyUtils.cuh>
// #include <THC/THCAtomics.cuh>


#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N)
{
    int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int max_block_num = 65000;
    return min(optimal_block_num, max_block_num);
}


template <typename scalar_t>
__global__ void affine_gpu_kernel(
    int64_t output_size,
    int32_t inHeight,
    int32_t inWidth,
    int32_t inChannel,
    int32_t outHeight,
    int32_t outWidth,
    scalar_t* dst,
    const scalar_t* src,
    const scalar_t* M,
    float delta)
{
    // thread idx
    const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= output_size)
        return;

    // (n, c, ph, pw) is an element in the pooled output
    int pw = thread_idx % outWidth;
    int ph = (thread_idx / outWidth) % outHeight;
    int c  = (thread_idx / outWidth / outHeight) % inChannel;
    int n  = thread_idx / outWidth / outHeight / inChannel;

    // convert pw, ph to x, y
    float base_x = M[1] * ph + M[2];
    float base_y = M[4] * ph + M[5];
    float x      = base_x + M[0] * pw;
    float y      = base_y + M[3] * pw;
    int32_t sx0 = (int32_t)x;
    int32_t sy0 = (int32_t)y;

    float u = x - sx0;
    float v = y - sy0;

    float tab[4];
    float taby[2], tabx[2];
    float v0, v1, v2, v3;
    taby[0] = 1.0f - v;
    taby[1] = v;
    tabx[0] = 1.0f - u;
    tabx[1] = u;

    tab[0] = taby[0] * tabx[0];
    tab[1] = taby[0] * tabx[1];
    tab[2] = taby[1] * tabx[0];
    tab[3] = taby[1] * tabx[1];

    bool flag0 = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
    bool flag1 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 < inHeight);
    bool flag2 = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
    bool flag3 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);

    int64_t input_feature_size = inHeight * inWidth * inChannel;
    int64_t in_planar_size = inHeight * inWidth;
    int64_t position1 = (n * input_feature_size + c * in_planar_size + (sy0 + 0) * inWidth + sx0);
    int64_t position2 = (n * input_feature_size + c * in_planar_size + (sy0 + 1) * inWidth + sx0);              
    v0                = flag0 ? src[position1 + 0] : delta;
    v1                = flag1 ? src[position1 + 1] : delta;
    v2                = flag2 ? src[position2 + 0] : delta;
    v3                = flag3 ? src[position2 + 1] : delta;
    scalar_t sum      = 0.0f;
    sum += v0 * tab[0] + v1 * tab[1] + v2 * tab[2] + v3 * tab[3];
    dst[thread_idx] = static_cast<scalar_t>(sum);
}


at::Tensor affine_cuda_forward(const at::Tensor& input,          /*[B, C, H, W]*/
                               const at::Tensor& affine_matrix,  /*[B, 2, 3]*/
                               const int out_h,
                               const int out_w)
{
    // build dst tensor
    auto nimgs = input.size(0);
    auto img_c = input.size(1);
    auto img_h = input.size(2);
    auto img_w = input.size(3);
    const int output_size = nimgs * img_c * out_h * out_w;
    auto output_tensor = at::zeros({nimgs, img_c, out_h, out_w}, input.options());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "affine_cuda", [&] {
        auto matrix_ptr = affine_matrix.data_ptr<scalar_t>();
        auto input_ptr = input.data_ptr<scalar_t>();
        auto output_ptr = output_tensor.data_ptr<scalar_t>();

        // launch kernel function on GPU with CUDA.
        affine_gpu_kernel<scalar_t><<<GET_BLOCKS(output_size), THREADS_PER_BLOCK,
                        0, at::cuda::getCurrentCUDAStream()>>>(output_size, img_h,
                        img_w, img_c, out_h, out_w, output_ptr, input_ptr, matrix_ptr, 0.0f);
    });    

    return  output_tensor;
}