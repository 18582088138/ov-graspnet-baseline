import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
import time
import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode, pred_decode_np
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()
device = torch.device("cpu")

from openvino.runtime import Core
from openvino import AsyncInferQueue
ov_extension_lib_path = 'ov_custom_op/build/libopenvino_operation_extension.so'
core = Core()
core.add_extension(ov_extension_lib_path)

def get_ov_net(ov_model_path, device="CPU"):
    ov_model = core.read_model(ov_model_path)
    ov_compiled_model = core.compile_model(ov_model, device)
    return ov_compiled_model
    ov_infer_request = ov_compiled_model.create_infer_request()
    return ov_infer_request


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
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def get_ov_grasps(ov_compiled_model, end_points):
    ov_input_data = end_points['point_clouds']
    ov_outputs = ov_compiled_model(ov_input_data)

    # print("============",len(ov_outputs))
    # print(ov_outputs.keys())
    # end_points['objectness_score'] = ov_outputs[0]
    # end_points['grasp_score_pred'] = ov_outputs[1]
    # end_points['fp2_xyz'] = ov_outputs[2]
    # end_points['grasp_top_view_xyz'] = ov_outputs[3]
    # end_points['grasp_angle_cls_pred'] = ov_outputs[4]
    # end_points['grasp_width_pred'] = ov_outputs[5]
    # end_points['grasp_tolerance_pred'] = ov_outputs[6]
    # grasp_preds = pred_decode_np(end_points)
    end_points['objectness_score'] = torch.from_numpy(ov_outputs[0])
    end_points['grasp_score_pred'] = torch.from_numpy(ov_outputs[1])
    end_points['fp2_xyz'] = torch.from_numpy(ov_outputs[2])
    end_points['grasp_top_view_xyz'] = torch.from_numpy(ov_outputs[3])
    end_points['grasp_angle_cls_pred'] = torch.from_numpy(ov_outputs[4])
    end_points['grasp_width_pred'] = torch.from_numpy(ov_outputs[5])
    end_points['grasp_tolerance_pred'] = torch.from_numpy(ov_outputs[6])
    grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    # print("======gg_array======",gg_array, gg_array.shape)
    gg = GraspGroup(gg_array)
    return gg

def ov_sync_infer(ov_compiled_model, end_points):
    ov_input_data = end_points['point_clouds']
    ov_outputs = ov_compiled_model(ov_input_data)
    print("============",len(ov_outputs))
    

def ov_async_infer(ov_compile_model, end_points, num_request=4, jobs = 8) :
    infer_queue = AsyncInferQueue(ov_compile_model, num_request)
    jobs_done = [{"finished": False, "latency": 0} for _ in range(jobs)]

    def callback(request, job_id):
        jobs_done[job_id]["finished"] = True
        jobs_done[job_id]["latency"] = request.latency
    
    infer_queue.set_callback(callback)
    for i in range(jobs):
        infer_queue.start_async(end_points['point_clouds'],i)
    infer_queue.wait_all()
    print("Jobs Done: ", jobs_done)

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def demo(data_dir,infer_count=10):
    ov_model_path = "torch_model_graspnet.xml"
    ov_compiled_model = get_ov_net(ov_model_path)
    end_points, cloud = get_and_process_data(data_dir)
    ov_input_data = end_points['point_clouds']
    gg = get_ov_grasps(ov_compiled_model, end_points)

    start_time = time.time()
    for i in range(infer_count):
        ov_sync_infer(ov_compiled_model, end_points)
    print(f"[OV CPU Sync Infer] total {infer_count} count Inference time: ", time.time()-start_time)
    print("[OV CPU Sync Infer] Avg Inference time", (time.time()-start_time)/infer_count)

    num_request=8
    jobs = infer_count
    start_time = time.time()
    ov_async_infer(ov_compiled_model, end_points, num_request, jobs)
    print(f"[OV CPU Async Infer] Totale {infer_count} count Inference time: ", time.time()-start_time)
    print("[OV CPU Async Infer] Avg Inference time", (time.time()-start_time)/infer_count)

    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    vis_grasps(gg, cloud)

if __name__=='__main__':
    data_dir = 'doc/example_data'
    demo(data_dir)
