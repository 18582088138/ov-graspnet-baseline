import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import openvino as ov
from openvino.runtime import Core
from openvino import serialize

from models.modules import ApproachNet, CloudCrop, OperationNet, ToleranceNet
from pointnet2.pointnet2_utils import QueryAndGroup, furthest_point_sample, gather_operation
import pointnet2.pytorch_utils as pt_utils


model_name = 'QueryAndGroup'
device = "CPU"
ov_extension_lib_path = './ov_custom_op/build/libopenvino_operation_extension.so'

print(f"========= {model_name} model initial==========")
core = Core()
core.add_extension(ov_extension_lib_path)

mlp=[128, 128, 128, 256]
npoint=1024
radius = 0.1
nsample = 32
xyz = torch.randn([1, 2048, 3], dtype=torch.float32)
new_xyz = torch.randn([1, 1024, 3], dtype=torch.float32)
features = torch.randn([1, 128, 2048], dtype=torch.float32)

class SelfModel(nn.Module):
    def __init__(self, mlp, npoint, radius, nsample, bn=True, use_xyz=True, ret_grouped_xyz=True, 
                 normalize_xyz=True, sample_uniformly=False, ret_unique_cnt=False):
        super().__init__()
        self.npoint = npoint
        self.grouping_operation = QueryAndGroup(radius, nsample, use_xyz=use_xyz, 
                                                ret_grouped_xyz=ret_grouped_xyz, normalize_xyz=normalize_xyz, 
                                                sample_uniformly=sample_uniformly, ret_unique_cnt=ret_unique_cnt)
        mlp_spec = mlp
        if use_xyz and len(mlp_spec)>0:
            mlp_spec[0] += 3
        self.mlp_module = pt_utils.SharedMLP(mlp_spec, bn=bn)
       
    def forward(self, xyz, features):
        xyz_flipped = xyz.transpose(1, 2).contiguous()

        inds = furthest_point_sample(xyz, torch.tensor(self.npoint))
        print("========inds========",inds, inds.shape, inds.dtype, type(inds))

        new_xyz = gather_operation(
            xyz_flipped, inds
        ).transpose(1, 2).contiguous() if self.npoint is not None else None

        # gather_xyz = gather_operation(
        #     xyz_flipped, inds
        # ).transpose(1, 2).contiguous() if self.npoint is not None else None

        grouped_features, grouped_xyz, unique_cnt  = self.grouping_operation(xyz, new_xyz, features=features)
        
        new_features = self.mlp_module(
            grouped_features
        )

        kernel_size=[1, int(new_features.size(3))]
        new_features = F.max_pool2d(new_features, kernel_size=kernel_size)
        print("========new_features========",new_features.shape, new_features.dtype, type(new_features))
        # new_features = new_features.squeeze(-1)
        target_shape = list(new_features.shape[:-1])  # 获取除了最后一个维度之外的所有维度
        new_features = new_features.view(*target_shape)
        return new_xyz, new_features, inds, unique_cnt


torch_model = SelfModel(mlp, npoint, radius, nsample)
torch_cpu_output = torch_model(xyz, features)
torch_cpu_result = torch_cpu_output[0].detach().cpu().numpy()
print("====Torch CPU result====", torch_cpu_result.shape, torch_cpu_result.dtype, type(torch_cpu_result))

onnx_model_path = f'torch_2_onnx_sub_{model_name}.onnx'
onnx_input = (xyz, features)
onnx_input_name = ['xyz', 'features']

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

ov_model_path = f'torch_2_onnx_sub_{model_name}.xml'
# 
ov_input = {'xyz': xyz, 
            'features': features}
ov_input_name =  {'xyz': ([1, 2048, 3]), 
                  'features': ([1, 128, 2048])}
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
ov_compiled_model = core.compile_model(ov_model, "CPU")
ov_infer_request = ov_compiled_model.create_infer_request()
ov_output = ov_infer_request.infer(ov_input)

ov_results = ov_output[0]
print("==== OV CPU result====", ov_results.shape, ov_results.dtype, type(ov_results))

print("=======OpenVINO inference Success========")