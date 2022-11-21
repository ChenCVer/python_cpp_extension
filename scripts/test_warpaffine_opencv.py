import cv2
import torch  # 不能删掉, 因为需要动态加载torch的一些动态库.
import numpy as np
from orbbec.warpaffine import affine_opencv

data_path = "./demo.png"
img = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)

# python中的numpy.array()与 pybind中的py::array_t一一对应.
src_point = np.array([[262.0, 324.0], [325.0, 323.0], [295.0, 349.0]], dtype=np.float32)
dst_point = np.array([[38.29, 51.69], [73.53, 51.69], [56.02, 71.73]], dtype=np.float32)
# python interface 
mat_trans = cv2.getAffineTransform(src_point, dst_point)
res = cv2.warpAffine(img, mat_trans, (600,800))
cv2.imwrite("py_img.png", res)

# C++ interface
warpffine_img = affine_opencv(img, src_point, dst_point)
cv2.imwrite("cpp_img.png", warpffine_img)