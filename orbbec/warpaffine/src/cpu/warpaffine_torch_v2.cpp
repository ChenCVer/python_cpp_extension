#include <torch/extension.h>
// #include<opencv2/opencv.hpp>

/*
https://www.cnblogs.com/shine-lee/p/10950963.html
*/
template <typename scalar_t>
void affine_cpu_kernel(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inChannel,
    int32_t inPlanarSize,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outPlanarSize,
    scalar_t* dst,
    const scalar_t* src,
    const scalar_t* M,
    float delta)
{
    for (int32_t i = 0; i < outHeight; i++)
    {
        float base_x = M[1] * i + M[2];
        float base_y = M[4] * i + M[5];

        for (int32_t j = 0; j < outWidth; j++)
        {
            float x     = base_x + M[0] * j;
            float y     = base_y + M[3] * j;
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

            int32_t idxDst = (i * outWidth + j);

            bool flag0 = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
            bool flag1 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 < inHeight);
            bool flag2 = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
            bool flag3 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);

            for(int32_t c = 0; c < inChannel; c++)
            {
                int32_t position1 = ((sy0 + 0) * inWidth + sx0);
                int32_t position2 = ((sy0 + 1) * inWidth + sx0);              
                v0                = flag0 ? src[position1 + c * inPlanarSize + 0] : delta;
                v1                = flag1 ? src[position1 + c * inPlanarSize + 1] : delta;
                v2                = flag2 ? src[position2 + c * inPlanarSize + 0] : delta;
                v3                = flag3 ? src[position2 + c * inPlanarSize + 1] : delta;
                scalar_t sum         = 0.0f;
                sum += v0 * tab[0] + v1 * tab[1] + v2 * tab[2] + v3 * tab[3];
                dst[idxDst + c * outPlanarSize] = static_cast<scalar_t>(sum);
            }
        }
    }
}

template <typename scalar_t>
at::Tensor affine_torch_cpu(const at::Tensor& input,          /*[B, C, H, W]*/
                            const at::Tensor& affine_matrix,  /*[B, 2, 3]*/
                            const int out_h,
                            const int out_w)  
{
    AT_ASSERTM(input.device().is_cpu(),         "input must be a CPU tensor");
    AT_ASSERTM(affine_matrix.device().is_cpu(), "affine_matrix must be a CPU tensor");

    // auto input_cpy = input.contiguous();
    // auto in_tensor = input_cpy.squeeze().permute({1, 2, 0}).contiguous();
    // in_tensor = in_tensor.mul(255).clamp(0, 255).to(torch::kU8);
    // in_tensor = in_tensor.to(torch::kCPU);
    // cv::Mat resultImg(800, 600, CV_8UC3);
    // std::memcpy((void *) resultImg.data, in_tensor.data_ptr(), sizeof(torch::kU8) * in_tensor.numel());
    // cv::imwrite("input.png", resultImg);

    auto matrix_ptr = affine_matrix.contiguous().data_ptr<scalar_t>();
    auto input_ptr = input.contiguous().data_ptr<scalar_t>();
    auto nimgs = input.size(0);
    auto img_c = input.size(1);
    auto img_h = input.size(2);
    auto img_w = input.size(3);
    auto in_img_size = img_c * img_h * img_w;
    auto out_img_size = img_c * out_h * out_w;

    // build dst tensor
    auto output_tensor = at::zeros({nimgs, img_c, out_h, out_w}, input.options());
    auto output_ptr = output_tensor.contiguous().data_ptr<scalar_t>();  
    
    for(int i = 0; i < nimgs; i++)
    {   
        scalar_t* matrix = matrix_ptr + i * 6; 
        scalar_t* in = input_ptr + i * in_img_size;
        scalar_t* out = output_ptr + i * out_img_size;
        affine_cpu_kernel<scalar_t>(img_h, img_w, img_c, img_w*img_h, 
                                    out_h, out_w, out_h*out_w, out, in, matrix, 0.0f);
    }

    return output_tensor;
}


at::Tensor affine_cpu(const at::Tensor& input,          /*[B, C, H, W]*/
                      const at::Tensor& affine_matrix,  /*[B, 2, 3]*/
                      const int out_h,
                      const int out_w)
{
    at::Tensor result;
    // AT_DISPATCH_FLOATING_TYPES: input.scalar_type() => scalar_t
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "affine_cpu", [&] {
        result = affine_torch_cpu<scalar_t>(input, affine_matrix, out_h, out_w);
    });
    return result;
}