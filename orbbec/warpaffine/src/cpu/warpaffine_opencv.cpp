#include<vector>
#include<iostream>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>
#include<opencv2/opencv.hpp>


namespace py = pybind11;

/* Python->C++ Mat */
cv::Mat numpy_uint8_1c_to_cv_mat(py::array_t<unsigned char>& input)
{
    if (input.ndim() != 2)
        throw std::runtime_error("1-channel image must be 2 dims ");
    
    py::buffer_info buf = input.request();
    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC1, (unsigned char*)buf.ptr);
    
    return mat;
}

cv::Mat numpy_uint8_3c_to_cv_mat(py::array_t<unsigned char>& input)
{
    if (input.ndim() != 3)
        throw std::runtime_error("3-channel image must be 3 dims ");

    py::buffer_info buf = input.request();
    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);

    return mat;
}

/* C++ Mat ->numpy */
py::array_t<unsigned char> cv_mat_uint8_1c_to_numpy(cv::Mat& input)
{
    py::array_t<unsigned char> dst;
    dst = py::array_t<unsigned char>({ input.rows,input.cols }, input.data);
    
    return dst;
}

py::array_t<unsigned char> cv_mat_uint8_3c_to_numpy(cv::Mat& input)
{
    py::array_t<unsigned char> dst;
    dst = py::array_t<unsigned char>({ input.rows,input.cols,3}, input.data);
    
    return dst;
}

py::array_t<unsigned char> affine_opencv(py::array_t<unsigned char>& input, 
                                        py::array_t<float>& from_point, 
                                        py::array_t<float>& to_point)
{
    // step1: get affine transform matrix
    py::buffer_info from_p_buf = from_point.request();
    py::buffer_info from_t_buf = to_point.request();
    float* fp = (float*)from_p_buf.ptr;
    float* tp = (float*)from_t_buf.ptr;
    int fp_stride = from_p_buf.shape[1];
    int tp_stride = from_t_buf.shape[1];

    cv::Point2f src[3] = {};
    cv::Point2f dst[3] = {};

    for(int i = 0; i < from_p_buf.shape[0]; i++)
    {
        src[i] = cv::Point2f(fp[fp_stride * i + 0], fp[fp_stride * i + 1]);
        dst[i] = cv::Point2f(tp[tp_stride * i + 0], tp[tp_stride * i + 1]);      
    }

    cv::Mat H = cv::getAffineTransform(src, dst);

    // step2: run affine transform
    cv::Mat input_mat = numpy_uint8_1c_to_cv_mat(input);
    cv::Mat output;
    cv::warpAffine(input_mat, output, H, cv::Size(600, 800), cv::INTER_LINEAR);

    return cv_mat_uint8_1c_to_numpy(output);
}