import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
# from graspnetAPI import GraspGroup

import openvino as ov
from openvino import serialize

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

device = "cuda"

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

from openvino.runtime import Core
from openvino.frontend.onnx import OpExtension
from openvino.frontend import ConversionExtension, NodeContext

parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--checkpoint_path', type=str, default="logs/log_kn/checkpoint.tar", help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
ov_extension_lib_path = 'ov_custom_op/build/libopenvino_operation_extension.so'
core = Core()
core.add_extension(ov_extension_lib_path)
compare_idx = 0

def get_net(device):
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(data_dir,device):
    # load data
    color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    # generate cloud
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    color_sampled = torch.from_numpy(color_sampled)

    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud


def export_single_graspnet(data_dir, device):
    net = get_net(device)
    view_estimator = net.view_estimator
    grasp_generator = net.grasp_generator

    end_points, cloud = get_and_process_data(data_dir, device)
    view_estimator_input = end_points['point_clouds']
    view_estimator_input_name = ['point_clouds']
    view_estimator_onnx_path = 'torch_sub_model_view_estimator.onnx'

    torch.onnx.export(
            view_estimator, 
            view_estimator_input,
            view_estimator_onnx_path,
            input_names=view_estimator_input_name,
            # opset_version=11,
            export_params=True,
            # verbose=True,
            )
    print(f"========= view_estimator onnx export success==========")

    ov_view_estimator_path = f'torch_sub_model_view_estimator.xml'
    ov_view_estimator_input = {'point_clouds': view_estimator_input}
    ov_view_estimator_input_name = {'point_clouds': ([1, 20000, 3])}

    ov_model = ov.convert_model(
                # torch_model,       #含custom op的torch model 无法导出graph
                view_estimator_onnx_path,
                input=ov_view_estimator_input_name,
                example_input=ov_view_estimator_input,
                extension=ov_extension_lib_path,
                verbose=True,
                )
    ov_model = core.read_model(view_estimator_onnx_path)
    ov_compiled_model = core.compile_model(ov_model, 'CPU')
    serialize(ov_model, ov_view_estimator_path)
    print(f"========= view_estimator openvino export success==========")


    with torch.no_grad():
        input_xyz, fp2_xyz, objectness_score, grasp_top_view_xyz, grasp_top_view_rot = view_estimator(view_estimator_input)
        grasp_generator_input = (input_xyz, fp2_xyz, grasp_top_view_rot)
    grasp_generator_input_name = ['input_xyz', 'fp2_xyz', 'grasp_top_view_rot']
    onnx_grasp_generator_path = 'torch_sub_model_grasp_generator.onnx'
    print(f"========= input_xyz: ==========", input_xyz.shape, input_xyz.dtype)
    print(f"========= fp2_xyz: ==========", fp2_xyz.shape, fp2_xyz.dtype)
    print(f"========= grasp_top_view_rot: ==========", grasp_top_view_rot.shape, grasp_top_view_rot.dtype)

    torch.onnx.export(
            grasp_generator, 
            grasp_generator_input,
            onnx_grasp_generator_path,
            input_names=grasp_generator_input_name,
            # opset_version=11,
            export_params=True,
            # verbose=True,
            )
    print(f"========= grasp_generator onnx export success==========")

    ov_grasp_generator_path = f'torch_sub_model_grasp_generator.xml'
    # fp2_xyz = torch.randn([1, 1024, 3], dtype=torch.float32)
    # input_xyz = torch.randn([1, 1024, 3], dtype=torch.float32)
    # grasp_top_view_rot = torch.randn([1, 1024, 3, 3], dtype=torch.float32)
    ov_grasp_generator_input = {'input_xyz': input_xyz, 
                                'fp2_xyz': fp2_xyz, 
                                'grasp_top_view_rot': grasp_top_view_rot}
    ov_grasp_generator_input_name = {'input_xyz': ([1, 20000, 3]),
                                    'fp2_xyz': ([1, 1024, 3]),
                                    'grasp_top_view_rot': ([1, 1024, 3, 3])}

    ov_model = ov.convert_model(
                # torch_model,       #含custom op的torch model 无法导出graph
                onnx_grasp_generator_path,
                input=ov_grasp_generator_input_name, 
                example_input=ov_grasp_generator_input, 
                extension=ov_extension_lib_path,
                verbose=True,
                )
    ov_model = core.read_model(onnx_grasp_generator_path)
    ov_compiled_model = core.compile_model(ov_model, 'CPU')
    serialize(ov_model, ov_grasp_generator_path)
    print(f"========= grasp_generator openvino export success==========")


def export_whole_graspnet(data_dir, device):
    net = get_net(device)
    view_estimator = net.view_estimator
    grasp_generator = net.grasp_generator

    end_points, cloud = get_and_process_data(data_dir, device)
    view_estimator_input = end_points['point_clouds']
    view_estimator_input_name = ['point_clouds']
    view_estimator_onnx_path = 'torch_model_graspnet.onnx'

    torch_cpu_output = net(view_estimator_input)
    torch_cpu_result = torch_cpu_output[compare_idx].detach().cpu().numpy()
    print("====Torch CPU result====", torch_cpu_result.shape, torch_cpu_result.dtype, type(torch_cpu_result))
    torch.onnx.export(
            net, 
            view_estimator_input,
            view_estimator_onnx_path,
            input_names=view_estimator_input_name,
            # opset_version=11,
            export_params=True,
            # verbose=True,
            )
    print(f"========= whole graspnet onnx export success==========")

    ov_view_estimator_path = f'torch_model_graspnet.xml'
    ov_view_estimator_input = {'point_clouds': view_estimator_input}
    ov_view_estimator_input_name = {'point_clouds': ([1, -1, 3])}

    ov_model = ov.convert_model(
                # torch_model,       #含custom op的torch model 无法导出graph
                view_estimator_onnx_path,
                input=ov_view_estimator_input_name,
                example_input=ov_view_estimator_input,
                extension=ov_extension_lib_path,
                verbose=True,
                )
    ov_model = core.read_model(view_estimator_onnx_path)
    ov_compiled_model = core.compile_model(ov_model, 'CPU')
    serialize(ov_model, ov_view_estimator_path)
    print(f"========= whole graspnet openvino export success==========")

    print("=======OpenVINO inference========")
    ov_model = core.read_model(ov_view_estimator_path)
    ov_compiled_model = core.compile_model(ov_model, "CPU")
    ov_infer_request = ov_compiled_model.create_infer_request()
    ov_output = ov_infer_request.infer(ov_view_estimator_input)

    ov_results = ov_output[compare_idx]
    print("==== OV CPU result====", ov_results.shape, ov_results.dtype, type(ov_results))

    print("=======OpenVINO inference Success========")


    print("========= pytorch & openvino inference result compare ==========")
    gpu_device = torch.device("cuda:0")
    torch_gpu_model = net.to(gpu_device)
    torch_gpu_output = torch_gpu_model(view_estimator_input.to(gpu_device))
    torch_gpu_result = torch_gpu_output[compare_idx].detach().cpu().numpy()
    print("====Torch GPU result====", torch_gpu_result.shape, torch_gpu_result.dtype, type(torch_gpu_result))

    gpu_mse = np.mean((torch_gpu_result - ov_results) ** 2)
    gpu_max_diff = np.max(np.abs(torch_gpu_result - ov_results))
    gpu_summary = np.array_equal(torch_gpu_result, ov_results)
    print(f"[GPU] Mean Squared Error: {gpu_mse}")
    print(f"[GPU] Max Difference: {gpu_max_diff}")

    # cpu_mse = np.mean((torch_cpu_result - ov_results) ** 2)
    # cpu_max_diff = np.max(np.abs(torch_cpu_result - ov_results))
    # print(f"[CPU] Mean Squared Error: {cpu_mse}")
    # print(f"[CPU] Max Difference: {cpu_max_diff}")




if __name__ == '__main__':
    data_dir = 'doc/example_data'
    export_whole_graspnet(data_dir, device)