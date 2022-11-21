#include <torch/extension.h>

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
    float delta);

template <typename scalar_t>
at::Tensor affine_torch_cpu(const at::Tensor& input, /*[B, C, H, W]*/
                            const at::Tensor& from,  /*[B, 3, 3]*/
                            const at::Tensor& to,    /*[B, 3, 2]*/
                            const int out_h,
                            const int out_w)  
{
    // step1. get affine transform matrix
    AT_ASSERTM(input.device().is_cpu(), "input must be a CPU tensor");
    AT_ASSERTM(from.device().is_cpu(),  "from point must be a CPU tensor");
    AT_ASSERTM(to.device().is_cpu(),    "to point must be a CPU tensor");

    // F = ((X^T*X)^-1)*(X^T*Y)
    auto matrix_l = (torch::transpose(from,1, 2).bmm(from)).inverse();  //(X^T*X)^-1)
    auto matrix_l_ptr = matrix_l.contiguous().data_ptr<scalar_t>();
    auto matrix_r = (torch::transpose(from, 1, 2)).bmm(to);             //(X^T*Y)
    auto matrix_r_ptr = matrix_r.contiguous().data_ptr<scalar_t>();
    auto affine = matrix_l.bmm(matrix_r);
    auto affine_ptr = affine.data_ptr<scalar_t>();

    auto affine_matrix = torch::transpose(affine, 1, 2);  // ((X^T*X)^-1) * (X^T*Y))^T --> [B, 2, 3].

    // step2. affine per imgs
    // get data pointer
    auto matrix_ptr = affine_matrix.contiguous().data_ptr<scalar_t>();
    auto input_ptr = input.contiguous().data_ptr<scalar_t>();

    auto nimgs = input.size(0);
    auto img_c = input.size(1);
    auto img_h = input.size(2);
    auto img_w = input.size(3);
    auto input_size = img_c * img_h * img_w;
    auto output_size = img_c * out_h * out_w;

    // build dst tensor
    auto output_tensor = at::zeros({nimgs, img_c, out_h, out_w}, input.options());
    auto output_ptr = output_tensor.contiguous().data_ptr<scalar_t>();  
    
    for(int i = 0; i < nimgs; i++)
    {   
        scalar_t* matrix = matrix_ptr + i * 6; 
        scalar_t* in = input_ptr + i * input_size;
        scalar_t* out = output_ptr + i * output_size;
        affine_cpu_kernel<scalar_t>(img_h, img_w, img_c, img_w*img_h, 
                                    out_h, out_w, out_h*out_w, out, in, matrix, 0.0f);
    }

    return output_tensor;
}


at::Tensor affine_cpu(const at::Tensor& input, /*[B, C, H, W]*/
                      const at::Tensor& from,  /*[B, 3, 3]*/
                      const at::Tensor& to,    /*[B, 3, 2]*/
                      const int out_h,
                      const int out_w)
{
    at::Tensor result;
    // AT_DISPATCH_FLOATING_TYPES: input.scalar_type() => scalar_t
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "affine_cpu", [&] {
        result = affine_torch_cpu<scalar_t>(input, from, to, out_h, out_w);
    });

    return result;
}