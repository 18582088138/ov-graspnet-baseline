import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import sys
import os
import numpy as np
import onnxruntime as ort

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import openvino as ov
from openvino.runtime import Core
from openvino import serialize

from models.modules import ApproachNet, CloudCrop, OperationNet, ToleranceNet
from pointnet2.pointnet2_utils import  three_nn, three_interpolate

model_name = 'ThreeNN'
device = "CPU"
ov_extension_lib_path = '../ov_custom_op/build/libopenvino_operation_extension.so'

"""
    Find the three nearest neighbors of unknown in known
    Parameters
    ----------
    unknown : torch.Tensor
        (B, n, 3) tensor of known features
    known : torch.Tensor
        (B, m, 3) tensor of unknown features

    Returns
    -------
    dist : torch.Tensor
        (B, n, 3) l2 distance to the three nearest neighbors
    idx : torch.Tensor
        (B, n, 3) index of 3 nearest neighbors
"""

print(f"========= {model_name} model initial==========")
core = Core()
core.add_extension(ov_extension_lib_path)

unknown = torch.randn([1, 64, 3], dtype=torch.float32)  # 示例数据
known = torch.randn([1, 128, 3], dtype=torch.float32)  # 示例数据

class SelfModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, unknown, known):
        dist, idx = three_nn(unknown, known)
        idx_batch_size, idx_npoint, idx_nsample = dist.shape[:3]
        idx_output = dist.float()
        matrix_multiplication_result = torch.bmm(idx_output.view(idx_batch_size * idx_npoint, idx_nsample, -1), 
                                                  idx_output.view(idx_batch_size * idx_npoint, -1, idx_nsample))
        # linear_output = self.linear_layer(matrix_multiplication_result)
        # output = F.softmax(matrix_multiplication_result, dim=-1)
        
        return matrix_multiplication_result, idx

# 实例化模型
torch_model = SelfModel()
torch_cpu_output = torch_model(unknown, known)
torch_cpu_result = torch_cpu_output[0].detach().cpu().numpy()
print("====Torch CPU result====", torch_cpu_result.shape, torch_cpu_result.dtype, type(torch_cpu_result))

onnx_model_path = f'test_model/torch_2_onnx_sub_{model_name}.onnx'
onnx_input = (unknown, known)
onnx_input_name = ['unknown', 'known']

torch.onnx.export(
        torch_model, 
        onnx_input,
        onnx_model_path,
        input_names=onnx_input_name,
        # opset_version=11,
        export_params=True,
        # verbose=True,
        )
print(f"========= {model_name} onnx export success==========")\

# unknown_np = unknown.numpy().astype(np.float32)
# known_np = known.numpy().astype(np.float32)
# onnx_input = {'unknown':unknown_np, 
#                 'known':known_np}
# onnx_session = ort.InferenceSession(onnx_model_path)
# onnx_output = onnx_session.run(None, onnx_input)
# onnx_results = onnx_output[0]
# print("==== ONNX result====", onnx_results.shape, onnx_results.dtype, type(onnx_results))
# print("=======ONNX inference Success========")

ov_model_path = f'test_model/torch_2_onnx_sub_{model_name}.xml'
ov_input = {'unknown':unknown,
            'known':known}
ov_input_name = {'unknown':([1, 64, 3]),
            'known':([1, 128, 3])}

ov_model = ov.convert_model(
            # torch_model,       #含custom op的torch model 无法导出graph
            onnx_model_path,
            input=ov_input_name, 
            example_input=ov_input, 
            extension=ov_extension_lib_path,
            verbose=True,
            )
serialize(ov_model, ov_model_path)
print(f"========= {model_name} openvino export success==========")

print("=======OpenVINO inference========")
ov_model = core.read_model(ov_model_path)
ov_compiled_model = core.compile_model(ov_model, device)
ov_infer_request = ov_compiled_model.create_infer_request()
ov_output = ov_infer_request.infer(ov_input)

ov_results = ov_output[0]
print("==== OV CPU result====", ov_results.shape, ov_results.dtype, type(ov_results))

print("=======OpenVINO inference Success========")


print("========= pytorch & openvino inference result compare ==========")
gpu_device = torch.device("cuda:0")
torch_gpu_model = torch_model.to(gpu_device)
torch_gpu_output = torch_gpu_model(unknown.to(gpu_device), known.to(gpu_device))
torch_gpu_result = torch_gpu_output[0].detach().cpu().numpy()
print("====Torch GPU result====", torch_gpu_result.shape, torch_gpu_result.dtype, type(torch_gpu_result))

gpu_mse = np.mean((torch_gpu_result - ov_results) ** 2)
gpu_max_diff = np.max(np.abs(torch_gpu_result - ov_results))
gpu_summary = np.array_equal(torch_gpu_result, ov_results)
print(f"[GPU] Mean Squared Error: {gpu_mse}")
print(f"[GPU] Max Difference: {gpu_max_diff}")

cpu_mse = np.mean((torch_cpu_result - ov_results) ** 2)
cpu_max_diff = np.max(np.abs(torch_cpu_result - ov_results))
print(f"[CPU] Mean Squared Error: {cpu_mse}")
print(f"[CPU] Max Difference: {cpu_max_diff}")

