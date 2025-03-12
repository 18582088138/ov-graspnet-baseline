import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import sys
import os
import numpy as np
# import pyopencl as cl


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import openvino as ov
from openvino.runtime import Core
from openvino import serialize

from models.modules import ApproachNet, CloudCrop, OperationNet, ToleranceNet
from pointnet2.pointnet2_utils import  grouping_operation, cylinder_query

model_name = 'GroupingOperation'
device = "GPU"
ov_extension_lib_path = 'build/libopenvino_operation_extension.so'
gpu_extension_xml = 'grouping_operation.xml'
cl_kernel_path = 'grouping_operation.cl'

print(f"========= {model_name} model initial==========")
b = 1
c = 8
n = 20000
npoints = 256
nsample = 64

core = Core()
core.add_extension(ov_extension_lib_path)
core.set_property("GPU", {"CONFIG_FILE": gpu_extension_xml})

# print(f"========= {model_name} model initial==========")
# with open(cl_kernel_path, 'r') as f:
#     kernel_src = f.read()

# # 创建上下文和命令队列
# ctx = cl.create_some_context()
# queue = cl.CommandQueue(ctx)

# # 编译内核
# prg = cl.Program(ctx, kernel_src).build()


# features = np.random.rand(b * c * n).astype(np.float32)
# idx = np.zeros((b * npoint * nsample), dtype=np.int32)
# output = np.zeros((b * c * npoint, nsample), dtype=np.float32)

# # 创建内存对象
# mf = cl.mem_flags
# features_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=features)
# idx_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=idx)
# output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)

# global_size=(b*c* npoint*nsample,)
# local_size=(256,)

# # 执行内核
# prg.groupingOperationCLKernel(queue, global_size, local_size, 

#                         features_buf, idx_buf,  
#                         output_buf,
#                         np.int32(b), np.int32(c), np.int32(n), 
#                         np.int32(npoint), np.int32(nsample),
#                         )
# # 读取结果
# cl.enqueue_copy(queue, output, output_buf)

# print("======= OpenCL kernel test Success=======",output_buf)

features = torch.randn([b, c, n], dtype=torch.float32)
idx = torch.ones([b, npoints, nsample], dtype=torch.int32)

class SelfModel(nn.Module):
    def __init__(self, input_dim=128, output_dim=192):
        super().__init__()
        self.linear_layer = nn.Linear(input_dim, output_dim)
        self.grouping_operation = grouping_operation

    def forward(self, features, idx):
        grouped_xyz = self.grouping_operation(features, idx)
        # features_add = features + torch.randn_like(features)
        # idx_times_two = idx * 2
        # grouped_xyz = self.grouping_operation(features_add, idx_times_two)
        batch_size, channels, num_points = grouped_xyz.shape[:3]
        matrix_multiplication_result = torch.bmm(grouped_xyz.view(batch_size * channels, num_points, -1), 
                                                  grouped_xyz.view(batch_size * channels, -1, num_points))
        # linear_output = self.linear_layer(matrix_multiplication_result)
        # output = F.softmax(linear_output, dim=-1)

        return grouped_xyz

# 实例化模型
torch_model = SelfModel()
torch_cpu_output = torch_model(features, idx)
torch_cpu_result = torch_cpu_output.detach().cpu().numpy()
print("====Torch CPU result====", torch_cpu_result.shape, torch_cpu_result.dtype, type(torch_cpu_result))

onnx_model_path = f'test_model/torch_2_onnx_sub_{model_name}.onnx'
onnx_input = (features, idx)
onnx_input_name = ['features', 'idx']

dynamic_axes = {'features': {0: 'batch_size', 2: 'num_points'},
                'idx': {0: 'batch_size', 1: 'num_points', 2: 'num_sample'}}

torch.onnx.export(
        torch_model, 
        onnx_input,
        onnx_model_path,
        input_names=onnx_input_name,
        # opset_version=11,
        export_params=True,
        dynamic_axes=dynamic_axes,
        # verbose=True,
        )
print(f"========= {model_name} onnx export success==========")

ov_model_path = f'test_model/torch_2_onnx_sub_{model_name}.xml'

ov_input ={'features':features, 
            'idx':idx}

ov_input_name =  {'features': ([b, -1, -1]),
                  'idx': ([b, -1, -1])}

# ov_input_name =  {'features': ([b, c, n]),
#                   'idx': ([b, npoints, nsample])}

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

print("=======OpenVINO CPU inference========")
ov_model = core.read_model(ov_model_path)
ov_compiled_model = core.compile_model(ov_model, "CPU")
ov_infer_request = ov_compiled_model.create_infer_request()
ov_output = ov_infer_request.infer(ov_input)
ov_cpu_results = ov_output[0]
print("==== OV CPU result====", ov_cpu_results.shape, ov_cpu_results.dtype, type(ov_cpu_results))

#==========================================================================================

print("=======OpenVINO GPU inference========")
ov_model = core.read_model(ov_model_path)
ov_compiled_model = core.compile_model(ov_model, "GPU")
ov_infer_request = ov_compiled_model.create_infer_request()
ov_output = ov_infer_request.infer(ov_input)
ov_gpu_results = ov_output[0]
print("==== OV GPU result====", ov_gpu_results.shape, ov_gpu_results.dtype, type(ov_gpu_results))

print("=======OpenVINO inference Success========")


print("========= pytorch & openvino inference result compare ==========")
gpu_device = torch.device("cuda:0")
torch_gpu_model = torch_model.to(gpu_device)
torch_gpu_output = torch_gpu_model(features.to(gpu_device), idx.to(gpu_device))
torch_gpu_result = torch_gpu_output.detach().cpu().numpy()
print("====Torch GPU result====", torch_gpu_result.shape, torch_gpu_result.dtype, type(torch_gpu_result))

gpu_mse = np.mean((torch_gpu_result - ov_cpu_results) ** 2)
gpu_max_diff = np.max(np.abs(torch_gpu_result - ov_cpu_results))
gpu_summary = np.array_equal(torch_gpu_result, ov_cpu_results)
print(f"[Torch GPU & OV CPU] Mean Squared Error: {gpu_mse}")
print(f"[Torch GPU & OV CPU] Max Difference: {gpu_max_diff}")

cpu_mse = np.mean((torch_cpu_result - ov_cpu_results) ** 2)
cpu_max_diff = np.max(np.abs(torch_cpu_result - ov_cpu_results))
print(f"[Torch CPU & OV CPU] Mean Squared Error: {cpu_mse}")
print(f"[Torch CPU & OV CPU] Max Difference: {cpu_max_diff}")



gpu_mse = np.mean((torch_gpu_result - ov_gpu_results) ** 2)
gpu_max_diff = np.max(np.abs(torch_gpu_result - ov_gpu_results))
gpu_summary = np.array_equal(torch_gpu_result, ov_gpu_results)
print(f"[Torch GPU & OV GPU] Mean Squared Error: {gpu_mse}")
print(f"[Torch GPU & OV GPU] Max Difference: {gpu_max_diff}")

cpu_mse = np.mean((torch_cpu_result - ov_gpu_results) ** 2)
cpu_max_diff = np.max(np.abs(torch_cpu_result - ov_gpu_results))
print(f"[Torch CPU & OV GPU] Mean Squared Error: {cpu_mse}")
print(f"[Torch CPU & OV GPU] Max Difference: {cpu_max_diff}")