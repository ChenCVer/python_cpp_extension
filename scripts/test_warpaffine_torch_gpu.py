import cv2
import torch
import numpy as np
from orbbec.warpaffine import affine_torch

data_path = "demo.png"

img = cv2.imread(data_path)
# transform img(numpy.array) to tensor(torch.Tensor)
# use permute
img_tensor = torch.from_numpy(img / 255.0).permute(2, 0, 1).contiguous()
img_tensor = img_tensor.unsqueeze(0).float()
img_tensor = img_tensor.cuda()
# src -> dst
# src_tensor = torch.tensor([[262.0, 324.0, 1.0], [325.0, 323.0, 1.0], [295.0, 349.0, 1.0]], dtype=torch.float32).unsqueeze(0)
# dst_tensor = torch.tensor([[38.29, 51.69], [73.53, 51.69], [56.02, 71.73]], dtype=torch.float32).unsqueeze(0)
# dst -> src
src_tensor = torch.tensor([[38.29, 51.69, 1.0], [73.53, 51.69, 1.0], [56.02, 71.73, 1.0]], dtype=torch.float32).unsqueeze(0)
dst_tensor = torch.tensor([[262.0, 324.0], [325.0, 323.0], [295.0, 349.0]], dtype=torch.float32).unsqueeze(0)
src_tensor = src_tensor.cuda()
dst_tensor = dst_tensor.cuda()

# compute affine transform matrix
matrix_l = torch.transpose(src_tensor, 1, 2).bmm(src_tensor)
matrix_l = torch.inverse(matrix_l)
matrix_r = torch.transpose(src_tensor, 1, 2).bmm(dst_tensor)
affine_matrix = torch.transpose(matrix_l.bmm(matrix_r), 1, 2)
affine_matrix = affine_matrix.contiguous().cuda()

warpffine_img = affine_torch(img_tensor, affine_matrix, 112, 112)
print(warpffine_img.shape)
warpffine_img = warpffine_img.cpu().squeeze(0).permute(1, 2, 0).numpy()
cv2.imwrite("torch_affine_gpu.png", np.uint8(warpffine_img * 255.0))
cv2.imshow("img", warpffine_img)
cv2.waitKey(0)