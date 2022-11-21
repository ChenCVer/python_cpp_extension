#include <torch/extension.h>
#include<pybind11/numpy.h>


py::array_t<unsigned char> affine_opencv(py::array_t<unsigned char>& input, 
                                        py::array_t<float>& from_point, 
                                        py::array_t<float>& to_point);


at::Tensor affine_cpu(const at::Tensor& input,          /*[B, C, H, W]*/
                      const at::Tensor& affine_matrix,  /*[B, 2, 3]*/
                      const int out_h,
                      const int out_w);

#ifdef WITH_CUDA
at::Tensor affine_gpu(const at::Tensor& input,          /*[B, C, H, W]*/
                      const at::Tensor& affine_matrix,  /*[B, 2, 3]*/
                      const int out_h,
                      const int out_w);
#endif


at::Tensor affine_torch(const at::Tensor& input,          /*[B, C, H, W]*/
                  		const at::Tensor& affine_matrix,  /*[B, 2, 3]*/
                  		const int out_h,
                  		const int out_w)
{
	if (input.device().is_cuda())
  	{
#ifdef WITH_CUDA
    return affine_gpu(input, affine_matrix, out_h, out_w);
#else
    AT_ERROR("affine is not compiled with GPU support");
#endif
  	}
  	return affine_cpu(input, affine_matrix, out_h, out_w);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("affine_opencv", &affine_opencv, "affine with c++ opencv");
  m.def("affine_torch", &affine_torch,   "affine with c++ libtorch");
}