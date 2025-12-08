import torch 
import torch_npu
from quant_cy_npu import QType, quant_dequant_float
import numpy as np 
import To_F8
import time
np.random.seed(42)

M = 11008
N = 2048
# x = np.random.randn(M,1,N)
# x = (0.2*np.random.randn(M,N) + np.random.uniform(-0.03,0.04,(M,N))).astype(np.float32)
# # x = np.ones([16, 16]).astype(np.float32) * 0.052599
# x_torch = torch.from_numpy(x)
# x_torch =torch.Tensor([[0.00000012,0.00000762939453125, 1.125, 2.125, 4.125, 16.25, 256.5, 50959, 0],[0, -0.000000119,-0.00000762939453125, -1.125, -2.125, -4.125, -16.25, -256.5,-50959]])
#weight=torch.Tensor([[1.125, 2.125, 4.125, 16.25, 256.5],[ -1.125, -2.125, -4.125, -16.25, -256.5]])
# print(weight)
qtype = QType('hif8')
import sys
file_list = [
        # '/home/ma-user/work/bucket-pangu-green-guiyang/wangxuefei/pangu-6-24-input-distribution/PanGu/save_input_weight/data_b_0_0_0.pt',
        # '/home/ma-user/work/bucket-pangu-green-guiyang/wangxuefei/pangu-6-24-input-distribution/PanGu/save_input_weight/data_b_0_1_0.pt',
        # '/home/ma-user/work/bucket-pangu-green-guiyang/wangxuefei/pangu-6-24-input-distribution/PanGu/save_input_weight/data_b_0_2_0.pt',
    '/home/ma-user/work/bucket-pangu-green-guiyang/wangxuefei/pangu-6-24-input-distribution/PanGu/save_input_weight/data_b_0_3_0.pt'
]
    
for pt_file in file_list:
    data_dict = torch.load(pt_file)
    grad_output = data_dict["grad_output"].squeeze(1)
    total_input = data_dict["total_input"].squeeze(1)
    weight = data_dict["weight"]
#         # data_b_0_0_0.pt Testing matrix shape: grad_outputtorch.Size([4096, 1, 4096]), total_inputtorch.Size([4096, 1, 2048]), weighttorch.Size([4096, 2048])
#     print(f"Testing matrix shape: grad_output {grad_output.shape}, total_input {total_input.shape}, weight {weight.shape}")

    # print(f'try to handle grad_output x weight')

        # y1 = To_F8.To_HiF8(x)
# y1 = quant_dequant_float(weight, qtype, force_py=True).cpu().numpy()
# y2 = quant_dequant_float(weight.npu(), qtype, force_py=False).cpu().numpy()
# y1 = quant_dequant_float(x_torch.clone(), qtype, force_py=True).cpu().numpy()
y1 = To_F8.To_HiF8(weight.float().cpu())

time1 = time.time()
y2 = quant_dequant_float(weight.clone(), qtype, force_py=False).cpu().float().numpy()
print("npu time:",time.time()-time1)
print(y1)
print(y2)

diff = np.abs(y1 - y2)
print('DIFF MAX: ', diff.max())
# print("y1======",y1)
# print("y2=",y2)
# arg = diff.flatten().argmax()
# print(x.flatten()[arg], y1.flatten()[arg], y2.flatten()[arg])
# print(x[:,0])
# print(y1[:,0])
# print(y2[:,0])
