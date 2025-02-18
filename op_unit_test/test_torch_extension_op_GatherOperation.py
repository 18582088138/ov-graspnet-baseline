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
from pointnet2.pointnet2_utils import gather_operation

model_name = 'GatherOperation'
device = "CPU"
ov_extension_lib_path = '../ov_custom_op/build/libopenvino_operation_extension.so'

"""
Parameters
    ----------
    features : torch.Tensor
        (B, C, N) tensor

    idx : torch.Tensor
        (B, npoint) tensor of the features to gather

Returns
    -------
    torch.Tensor
        (B, C, npoint) tensor
"""

print(f"========= {model_name} model initial==========")
core = Core()
core.add_extension(ov_extension_lib_path)

features = torch.randn([1, 3, 20000], dtype=torch.float32)
idx = torch.ones([1, 64], dtype=torch.int32)


class SelfModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, idx):
        interpolated_feats = gather_operation(features, idx)
        idx_batch_size, idx_npoint, idx_nsample = interpolated_feats.shape[:3]
        idx_output = interpolated_feats.float()
        matrix_multiplication_result = torch.bmm(idx_output.view(idx_batch_size * idx_npoint, idx_nsample, -1), 
                                                  idx_output.view(idx_batch_size * idx_npoint, -1, idx_nsample))
        # linear_output = self.linear_layer(matrix_multiplication_result)
        # output = F.softmax(matrix_multiplication_result, dim=-1)
        
        return matrix_multiplication_result

# 实例化模型
torch_model = SelfModel()
torch_cpu_output = torch_model(features, idx)
torch_cpu_result = torch_cpu_output.detach().cpu().numpy()
print("====Torch CPU result====", torch_cpu_result.shape, torch_cpu_result.dtype, type(torch_cpu_result))

onnx_model_path = f'test_model/torch_2_onnx_sub_{model_name}.onnx'
onnx_input = (features, idx)
onnx_input_name = ['features', 'idx']

torch.onnx.export(
        torch_model, 
        onnx_input,
        onnx_model_path,
        input_names=onnx_input_name,
        # opset_version=11,
        export_params=True,
        # verbose=True,
        )
print(f"========= {model_name} onnx export success==========")

ov_model_path = f'test_model/torch_2_onnx_sub_{model_name}.xml'
ov_input = {'features':features, 
            'idx':idx,
            }
ov_input_name = {'features':([1, 3, 20000]), 
                 'idx':([1, 64]),
                 }

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
torch_gpu_output = torch_gpu_model(features.to(gpu_device), idx.to(gpu_device))
torch_gpu_result = torch_gpu_output.detach().cpu().numpy()
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

torch_mse = np.mean((torch_cpu_result - torch_gpu_result) ** 2)
torch_max_diff = np.max(np.abs(torch_cpu_result - torch_gpu_result))
print(f"[Torch] Mean Squared Error: {torch_mse}")
print(f"[Torch] Max Difference: {torch_max_diff}")