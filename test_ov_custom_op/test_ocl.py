import pyopencl as cl
import numpy as np

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

# 创建上下文和命令队列
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# 编译内核
prg = cl.Program(ctx, kernel_src).build()


features = np.random.rand(b * c * n).astype(np.float32)
idx = np.zeros((b * npoint * nsample), dtype=np.int32)
output = np.zeros((b * c * npoint, nsample), dtype=np.float32)

# 创建内存对象
mf = cl.mem_flags
features_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=features)
idx_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=idx)
output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)

global_size=(b*c* npoint*nsample,)
local_size=(256,)

# 执行内核
prg.groupingOperationCLKernel(queue, global_size, local_size, 
                              np.int32(b), np.int32(c), np.int32(n), 
                        np.int32(npoint), np.int32(nsample),
                        features_buf, idx_buf,  
                        output_buf,
                        )
# 读取结果
cl.enqueue_copy(queue, output, output_buf)

print("======= OpenCL kernel test Success=======",output_buf)
