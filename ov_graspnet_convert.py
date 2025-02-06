import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup

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

parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--checkpoint_path', type=str, default="logs/log_kn/checkpoint.tar", help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()

def ov_convert():

    return 0

def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(data_dir):
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def export_grasp_generator(data_dir):
    net = get_net()
    end_points, cloud = get_and_process_data(data_dir)
    view_estimator = net.view_estimator
    grasp_generator = net.grasp_generator
    print("== end_points ==",end_points)
    view_estimator_input = end_points['point_clouds']
    with torch.no_grad():
        input_xyz, fp2_xyz, objectness_score, grasp_top_view_xyz, grasp_top_view_rot = view_estimator(view_estimator_input)
        grasp_generator_input = (input_xyz, fp2_xyz, grasp_top_view_rot)
        print("== grasp_generator_input ==",grasp_generator(input_xyz, fp2_xyz, grasp_top_view_rot))

    torch.onnx.export(
        grasp_generator, 
        grasp_generator_input,
        'IR_model/grasp_generator.onnx',
        input_names=['input_xyz', 'fp2_xyz', 'grasp_top_view_rot'],
        opset_version=11,
        do_constant_folding=True,
        # export_params=True,
        verbose=True,)
    print("== export grasp_generator.onnx success ==")

    # ov_model = ov.convert_model(net, example_input=end_points)
    # serialize(ov_model, 'IR_model/ov_graspnet.xml')
    # print("== export ov_graspnet IR success ==")


def export_view_estimator(data_dir):
    net = get_net()
    end_points, cloud = get_and_process_data(data_dir)
    view_estimator = net.view_estimator
    grasp_generator = net.grasp_generator
    print("== end_points ==",end_points)
    view_estimator_input = end_points['point_clouds']
    torch.onnx.export(
        view_estimator, 
        view_estimator_input,
        'IR_model/view_estimator.onnx',
        input_names=['point_clouds'],
        opset_version=11,
        do_constant_folding=True,
        # export_params=True,
        verbose=True,)
    print("== export view_estimator.onnx success ==")


if __name__ == '__main__':
    data_dir = 'doc/example_data'
    export_view_estimator(data_dir)
    export_grasp_generator(data_dir)