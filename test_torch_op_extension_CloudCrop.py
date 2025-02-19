import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import openvino as ov
from openvino.runtime import Core
from openvino import serialize

from models.modules import ApproachNet, CloudCrop, OperationNet, ToleranceNet


core = Core
ov_extension_lib_path = 'ov_custom_op/build/libopenvino_operation_extension.so'

fp2_xyz = torch.randn([1, 1024, 3], dtype=torch.float32)
input_xyz = torch.randn([1, 1024, 3], dtype=torch.float32)
grasp_top_view_rot = torch.randn([1, 1024, 3, 3], dtype=torch.float32)

class CloudCropCustomModel(nn.Module):
    def __init__(self, num_angle=12, num_depth=4, cylinder_radius=0.05, 
                 hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04]):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.crop = CloudCrop(64, 3, cylinder_radius, hmin, hmax_list)

    def forward(self, input_xyz, fp2_xyz, grasp_top_view_rot):
        # 定义前向传播
        pointcloud = input_xyz
        seed_xyz = fp2_xyz
        grasp_top_views_rot = grasp_top_view_rot
        vp_features = self.crop(seed_xyz, pointcloud, grasp_top_views_rot)
        return vp_features

# 实例化模型
CloudCrop_model = CloudCropCustomModel()

output = CloudCrop_model(fp2_xyz, input_xyz, grasp_top_view_rot)
print(output.size())

torch.save(CloudCrop_model, "IR_model/torch_extension_CloudCrop.pth")
torch.save(CloudCrop_model.state_dict(), "IR_model/torch_extension_CloudCrop_state_dict.pth")

print("=========CloudCrop torch save success==========")




grasp_generator_onnx_input = (input_xyz, fp2_xyz, grasp_top_view_rot)
torch.onnx.export(
        CloudCrop_model, 
        grasp_generator_onnx_input,
        'IR_model/onnx_extension_sub_CloudCrop.onnx',
        input_names=['input_xyz', 'fp2_xyz', 'grasp_top_view_rot'],
        # opset_version=11,
        export_params=True,
        # verbose=True,
        )
print("=========CloudCrop onnx export success==========")

grasp_generator_ov_input = {'input_xyz':input_xyz, 
                            'fp2_xyz':fp2_xyz, 
                            'grasp_top_view_rot':grasp_top_view_rot}

grasp_generator_input_name = {'input_xyz': ([1, 1024, 3]),
                            'fp2_xyz': ([1, 1024, 3]),
                            'grasp_top_view_rot': ([1, 1024, 3, 3])}

ov_grasp_generator = ov.convert_model(
                    # CloudCrop_model, 
                    'IR_model/onnx_extension_sub_CloudCrop.onnx',
                    input=grasp_generator_input_name, 
                    example_input=grasp_generator_ov_input, 
                    extension=ov_extension_lib_path,
                    # verbose=True,
                    )

# ov_grasp_generator_compile = core.compile_model(ov_grasp_generator, 'CPU')
ov.save_model(ov_grasp_generator, 'IR_model/ov_extension_sub_CloudCrop.xml')
print("=========CloudCrop ov compile success==========")





from pointnet2.pointnet2_utils import  grouping_operation, cylinder_query

features = torch.randn([1, 3, 20000], dtype=torch.float32)
idx = torch.ones([1, 64, 8], dtype=torch.int32)

class GroupingOperationModel(nn.Module):
    def __init__(self, input_dim=64, output_dim=192):
        super().__init__()
        self.linear_layer = nn.Linear(input_dim, output_dim)
        # self.grouping_operation = GroupingOperation()

    def forward(self, features, idx):
        features_add = features + torch.randn_like(features)
        idx_times_two = idx * 2

        grouped_xyz = grouping_operation(features_add, idx_times_two)
        batch_size, channels, num_points = grouped_xyz.shape[:3]
        matrix_multiplication_result = torch.bmm(grouped_xyz.view(batch_size * channels, num_points, -1), 
                                                  grouped_xyz.view(batch_size * channels, -1, num_points))
        linear_output = self.linear_layer(matrix_multiplication_result)
        output = F.softmax(linear_output, dim=-1)
        
        return output

# 实例化模型
GroupingOperation_model = GroupingOperationModel()
output = GroupingOperation_model(features, idx)
print(output.size())

GroupingOperation_onnx_input = (features, idx)
torch.onnx.export(
        GroupingOperation_model, 
        GroupingOperation_onnx_input,
        'IR_model/onnx_extension_sub_GroupingOperation.onnx',
        input_names=['features', 'idx'],
        # opset_version=11,
        export_params=True,
        # verbose=True,
        )
print("========= GroupingOperation_model onnx export success==========")

GroupingOperation_ov_input = {'features':features, 
                            'idx':idx}
GroupingOperation_input_name = {'features': ([1, 3, 20000]),
                            'idx': ([1, 64, 8])}
ov_GroupingOperation = ov.convert_model(
                    # GroupingOperation_model,        #含custom op的torch model 无法导出graph
                    'IR_model/onnx_extension_sub_GroupingOperation.onnx',  
                    input=GroupingOperation_input_name, 
                    example_input=GroupingOperation_ov_input, 
                    extension=ov_extension_lib_path,
                    verbose=True,
                    )
serialize(ov_GroupingOperation, 'IR_model/onnx_extension_sub_GroupingOperation.xml')
print("========= GroupingOperation_model openvino export success==========")







print("========= CylinderQuery model initial==========")

radius = torch.tensor(1.0)
hmin = torch.tensor(-2.0)
hmax = torch.tensor(2.0)
nsample = torch.tensor(32)
xyz = torch.randn([1, 1024, 3], dtype=torch.float32)  # 示例数据
new_xyz = torch.randn([1, 512, 3], dtype=torch.float32)  # 示例数据
rot = torch.randn([1, 512, 9], dtype=torch.float32)

class CylinderQueryModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.grouping_operation = GroupingOperation()

    def forward(self, new_xyz, xyz, rot, radius, hmin, hmax, nsample):
        
        new_xyz_add = new_xyz * 2 + torch.randn_like(new_xyz)
        xyz_add = xyz + torch.randn_like(xyz)
        rot_add = rot*0.5 + torch.randn_like(rot)
        idx_output = cylinder_query(new_xyz_add, xyz_add, rot_add, radius, hmin, hmax, nsample)

        # idx_output = cylinder_query(new_xyz, xyz, rot, radius, hmin, hmax, nsample)
        idx_batch_size, idx_npoint, idx_nsample = idx_output.shape[:3]

        matrix_multiplication_result = torch.bmm(idx_output.view(idx_batch_size * idx_npoint, idx_nsample, -1), 
                                                  idx_output.view(idx_batch_size * idx_npoint, -1, idx_nsample))
        # linear_output = self.linear_layer(matrix_multiplication_result)
        # output = F.softmax(matrix_multiplication_result, dim=-1)
        
        return matrix_multiplication_result

# 实例化模型
cylinder_query_model = CylinderQueryModel()
output = cylinder_query_model(new_xyz, xyz, rot, radius, hmin, hmax, nsample)
print(output.size())

cylinder_query_onnx_input = (new_xyz, xyz, rot, radius, hmin, hmax, nsample)
torch.onnx.export(
        cylinder_query_model, 
        cylinder_query_onnx_input,
        'IR_model/onnx_extension_sub_CylinderQuery.onnx',
        input_names=['new_xyz', 'xyz', 'rot', 'radius', 'hmin', 'hmax', 'nsample'],
        # opset_version=11,
        export_params=True,
        # verbose=True,
        )
print("========= CylinderQuery onnx export success==========")

CylinderQuery_ov_input = {'new_xyz':new_xyz,
                          'xyz':xyz,
                          'rot':rot,
                          'radius':radius, 
                          'hmin':hmin,
                          'hmax':hmax,
                          'sample':nsample,
                          }
CylinderQuery_input_name = {'new_xyz': ([1, 512, 3]),
                            'xyz': ([1, 512, 3]),
                            'rot': ([1, 512, 9]),
                            'radius': ([1]),
                            'hmin': ([1]),
                            'hmax': ([1]),
                            'nsample': ([1]),
                            }

ov_CylinderQuery = ov.convert_model(
                    # cylinder_query_model,       #含custom op的torch model 无法导出graph
                    'IR_model/onnx_extension_sub_CylinderQuery.onnx',
                    input=CylinderQuery_input_name, 
                    example_input=CylinderQuery_ov_input, 
                    extension=ov_extension_lib_path,
                    verbose=True,
                    )
serialize(ov_CylinderQuery, 'IR_model/onnx_extension_sub_CylinderQuery.xml')
print("========= CylinderQuery openvino export success==========")
