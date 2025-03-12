import pyopencl as cl
import numpy as np

import torch
import openvino as ov
from openvino.runtime import Core

model_name = 'GroupingOperation'
device = "GPU"
ov_extension_lib_path = 'build/libopenvino_operation_extension.so'
cl_interface_path = "grouping_operation.xml"
cl_kernel_path = "grouping_operation.cl"

# 设置输入参数
b = 1
c = 3
n = 20000
npoint = 64
nsample = 8


print(f"========= {model_name} model initial==========")
with open(cl_kernel_path, 'r') as f:
    kernel_src = f.read()

features = torch.randn([b, c, n], dtype=torch.float32)  # 示例数据
idx = torch.ones([b, npoint, nsample], dtype=torch.int32)  # 示例数据
ov_model_path = f'test_model/torch_2_onnx_sub_{model_name}.xml'

ov_input = {'features':features,
            'idx':idx,
            # 'b':np.int32(b), 
            # 'c':np.int32(c), 
            # 'n':np.int32(n), 
            # 'npoint':np.int32(npoint), 
            # 'nsample':np.int32(nsample),
            }

# ov_input_name = {'features': ([b, c, n]),
#                 'idx': ([b, npoint, nsample])}

ov_input_name = {'features': ([b, -1, -1]),
                'idx': ([b, -1, -1])}

core = Core()
core.add_extension(ov_extension_lib_path)
print("=======OpenVINO CPU inference========")
ov_model = core.read_model(ov_model_path)
ov_compiled_model = core.compile_model(ov_model, 'CPU')
ov_infer_request = ov_compiled_model.create_infer_request()
ov_output = ov_infer_request.infer(ov_input)
ov_results = ov_output[0]
print("==== OV CPU result====", ov_results.shape, ov_results.dtype, type(ov_results))

print("=======OpenVINO GPU inference========")
core.set_property("GPU", {"CONFIG_FILE": cl_interface_path})
ov_compiled_model = core.compile_model(ov_model, 'GPU')
ov_infer_request = ov_compiled_model.create_infer_request()
ov_output = ov_infer_request.infer(ov_input)
ov_results = ov_output[0]
print("==== OV GPU result====", ov_results.shape, ov_results.dtype, type(ov_results))
