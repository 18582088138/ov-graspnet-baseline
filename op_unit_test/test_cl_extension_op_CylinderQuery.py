import pyopencl as cl
import numpy as np

import torch
import openvino as ov
from openvino.runtime import Core

model_name = 'CylinderQuery'
device = "GPU"
ov_extension_lib_path = '../ov_custom_op/build/libopenvino_operation_extension.so'
cl_cylinder_query_path = "../ov_custom_op/cylinder_query.xml"
cl_kernel_path = "../ov_custom_op/cylinder_query.cl"

print(f"========= {model_name} model initial==========")
with open(cl_kernel_path, 'r') as f:
    kernel_src = f.read()

# 创建上下文和命令队列
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# 编译内核
prg = cl.Program(ctx, kernel_src).build()

# 设置输入参数
b = 1
n = 512
m = 512
radius = np.float32(1.0)
hmin = np.float32(-2.0)
hmax = np.float32(2.0)
nsample = np.int32(32)

new_xyz = np.random.rand(b * m * 3).astype(np.float32)
xyz = np.random.rand(b * n * 3).astype(np.float32)
rot = np.random.rand(b * m * 9).astype(np.float32)
idx = np.zeros((b * m * nsample), dtype=np.int32)

# 创建内存对象
mf = cl.mem_flags
new_xyz_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=new_xyz)
xyz_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xyz)
rot_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rot)
idx_buf = cl.Buffer(ctx, mf.WRITE_ONLY, idx.nbytes)

global_size=(b*m,)
local_size=(256,)

# 执行内核
prg.cylinderQueryCLKernel(queue, global_size, local_size, 
                              new_xyz_buf, xyz_buf, rot_buf, 
                              radius, hmin, hmax, nsample, 
                              np.int32(b), np.int32(n), np.int32(m), 
                              idx_buf)

# 读取结果
cl.enqueue_copy(queue, idx, idx_buf)


print("======= OpenCL kernel test Success=======",idx)

#========================================================

new_xyz = torch.randn([1, 512, 3], dtype=torch.float32)  # 示例数据
xyz = torch.randn([1, 512, 3], dtype=torch.float32)  # 示例数据
rot = torch.randn([1, 512, 9], dtype=torch.float32)
radius = torch.tensor(1.0)
hmin = torch.tensor(-2.0)
hmax = torch.tensor(2.0)
nsample = torch.tensor(32)
b = torch.tensor(1)
n = torch.tensor(512)
m = torch.tensor(512)


ov_model_path = f'test_model/torch_2_onnx_sub_{model_name}.xml'

ov_input = {'new_xyz':new_xyz,
            'xyz':xyz,
            'rot':rot,
            'radius':radius, 
            'hmin':hmin,
            'hmax':hmax,
            'nsample':nsample,
            }
ov_input_name = {'new_xyz': ([1, 512, 3]),
                'xyz': ([1, 512, 3]),
                'rot': ([1, 512, 9]),
                'radius': ([1]),
                'hmin': ([1]),
                'hmax': ([1]),
                'nsample': ([1]),
                }

ov_gpu_input = {'new_xyz':new_xyz,
            'xyz':xyz,
            'rot':rot,
            'radius':radius, 
            'hmin':hmin,
            'hmax':hmax,
            'nsample':nsample,
            'b': b,
            'n': n,
            'm': m
            }
ov_gpu_input_name = {'new_xyz': ([1, 512, 3]),
                'xyz': ([1, 512, 3]),
                'rot': ([1, 512, 9]),
                'radius': ([1]),
                'hmin': ([1]),
                'hmax': ([1]),
                'nsample': ([1]),
                'b': ([1]),
                'n': ([1]),
                'm': ([1])
                }


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
core.set_property("GPU", {"CONFIG_FILE": cl_cylinder_query_path})
ov_compiled_model = core.compile_model(ov_model, 'GPU')
ov_infer_request = ov_compiled_model.create_infer_request()
ov_output = ov_infer_request.infer(ov_input)
ov_results = ov_output[0]
print("==== OV GPU result====", ov_results.shape, ov_results.dtype, type(ov_results))