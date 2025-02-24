""" GraspNet baseline model definition.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from backbone import Pointnet2Backbone
from modules import ApproachNet, CloudCrop, OperationNet, ToleranceNet
from loss import get_loss
from loss_utils import GRASP_MAX_WIDTH, GRASP_MAX_TOLERANCE, batch_viewpoint_params_to_matrix_np
from label_generation import process_grasp_labels, match_grasp_view_and_label, batch_viewpoint_params_to_matrix


class GraspNetStage1(nn.Module):
    def __init__(self, input_feature_dim=0, num_view=300):
        super().__init__()
        self.backbone = Pointnet2Backbone(input_feature_dim)
        self.vpmodule = ApproachNet(num_view, 256)

    def forward(self, point_clouds):
        # breakpoint()
        fp2_features, fp2_xyz, input_xyz = self.backbone(point_clouds)
        # return input_xyz, fp2_xyz, fp2_features
        objectness_score, grasp_top_view_xyz, grasp_top_view_rot = self.vpmodule(fp2_xyz, fp2_features)
        return input_xyz, fp2_xyz, objectness_score, grasp_top_view_xyz, grasp_top_view_rot

class GraspNetStage2(nn.Module):
    def __init__(self, num_angle=12, num_depth=4, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=True):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.is_training = is_training
        self.crop = CloudCrop(64, 3, cylinder_radius, hmin, hmax_list)
        self.operation = OperationNet(num_angle, num_depth)
        self.tolerance = ToleranceNet(num_angle, num_depth)
    
    def forward(self, input_xyz, fp2_xyz, grasp_top_view_rot, end_points=None):
        pointcloud = input_xyz
        if self.is_training:
            grasp_top_views_rot, _, _, _, end_points = match_grasp_view_and_label(end_points)
            seed_xyz = end_points['batch_grasp_point']
        else:
            grasp_top_views_rot = grasp_top_view_rot
            seed_xyz = fp2_xyz
        vp_features = self.crop(seed_xyz, pointcloud, grasp_top_views_rot)
        grasp_score_pred, grasp_angle_cls_pred, grasp_width_pred = self.operation(vp_features)
        grasp_tolerance_pred = self.tolerance(vp_features)

        return grasp_score_pred, grasp_angle_cls_pred, grasp_width_pred, grasp_tolerance_pred

class GraspNet(nn.Module):
    def __init__(self, input_feature_dim=0, num_view=300, num_angle=12, num_depth=4, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=True):
        super().__init__()
        self.is_training = is_training
        self.view_estimator = GraspNetStage1(input_feature_dim, num_view)
        self.grasp_generator = GraspNetStage2(num_angle, num_depth, cylinder_radius, hmin, hmax_list, is_training)

    def forward(self, point_clouds, end_points=None):
        input_xyz, fp2_xyz, objectness_score, grasp_top_view_xyz, grasp_top_view_rot = self.view_estimator(point_clouds)
        if self.is_training:
            end_points = process_grasp_labels(end_points)
        grasp_score_pred, grasp_angle_cls_pred, grasp_width_pred, grasp_tolerance_pred = self.grasp_generator(input_xyz, fp2_xyz, grasp_top_view_rot)
        return objectness_score, grasp_score_pred, fp2_xyz, grasp_top_view_xyz, grasp_angle_cls_pred, grasp_width_pred, grasp_tolerance_pred
        # return fp2_xyz, objectness_score, grasp_top_view_xyz, grasp_score_pred, grasp_angle_cls_pred, grasp_width_pred, grasp_tolerance_pred

def pred_decode(end_points):
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    for i in range(batch_size):
        ## load predictions
        objectness_score = end_points['objectness_score'][i].float()
        grasp_score = end_points['grasp_score_pred'][i].float()
        grasp_center = end_points['fp2_xyz'][i].float()
        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_angle_class_score = end_points['grasp_angle_cls_pred'][i]
        grasp_width = 1.2 * end_points['grasp_width_pred'][i]
        grasp_width = torch.clamp(grasp_width, min=0, max=GRASP_MAX_WIDTH)
        grasp_tolerance = end_points['grasp_tolerance_pred'][i]

        # print("=== objectness_score ===",objectness_score.size())
        # print("=== grasp_score ===",grasp_score.size())
        # print("=== grasp_center ===",grasp_center.size())
        # print("=== approaching ===",approaching.size())
        # print("=== grasp_angle_class_score ===",grasp_angle_class_score.size())
        # print("=== grasp_width ===",grasp_width.size())
        # print("=== grasp_tolerance ===",grasp_tolerance.size())

        ## slice preds by angle
        # grasp angle
        # breakpoint()
        grasp_angle_class = torch.argmax(grasp_angle_class_score, 0)
        grasp_angle = grasp_angle_class.float() / 12 * np.pi
        # grasp score & width & tolerance
        grasp_angle_class_ = grasp_angle_class.unsqueeze(0)
        grasp_score = torch.gather(grasp_score, 0, grasp_angle_class_).squeeze(0)
        grasp_width = torch.gather(grasp_width, 0, grasp_angle_class_).squeeze(0)
        grasp_tolerance = torch.gather(grasp_tolerance, 0, grasp_angle_class_).squeeze(0)

        ## slice preds by score/depth
        # grasp depth
        grasp_depth_class = torch.argmax(grasp_score, 1, keepdims=True)
        grasp_depth = (grasp_depth_class.float()+1) * 0.01
        # grasp score & angle & width & tolerance
        grasp_score = torch.gather(grasp_score, 1, grasp_depth_class)
        grasp_angle = torch.gather(grasp_angle, 1, grasp_depth_class)
        grasp_width = torch.gather(grasp_width, 1, grasp_depth_class)
        grasp_tolerance = torch.gather(grasp_tolerance, 1, grasp_depth_class)

        ## slice preds by objectness
        objectness_pred = torch.argmax(objectness_score, 0)
        objectness_mask = (objectness_pred==1)
        grasp_score = grasp_score[objectness_mask]
        grasp_width = grasp_width[objectness_mask]
        grasp_depth = grasp_depth[objectness_mask]
        approaching = approaching[objectness_mask]
        grasp_angle = grasp_angle[objectness_mask]
        grasp_center = grasp_center[objectness_mask]
        grasp_tolerance = grasp_tolerance[objectness_mask]
        grasp_score = grasp_score * grasp_tolerance / GRASP_MAX_TOLERANCE

        ## convert to rotation matrix
        Ns = grasp_angle.size(0)
        approaching_ = approaching.view(Ns, 3)
        grasp_angle_ = grasp_angle.view(Ns)
        rotation_matrix = batch_viewpoint_params_to_matrix(approaching_, grasp_angle_)
        rotation_matrix = rotation_matrix.view(Ns, 9)

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, rotation_matrix, grasp_center, obj_ids], axis=-1))
    return grasp_preds


import numpy as np

def pred_decode_np(end_points):
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    for i in range(batch_size):
        ## load predictions
        objectness_score = end_points['objectness_score'][i]
        grasp_score = end_points['grasp_score_pred'][i]
        grasp_center = end_points['fp2_xyz'][i]
        approaching = -end_points['grasp_top_view_xyz'][i]
        grasp_angle_class_score = end_points['grasp_angle_cls_pred'][i]
        grasp_width = 1.2 * end_points['grasp_width_pred'][i]
        grasp_width = np.clip(grasp_width, a_min=0, a_max=GRASP_MAX_WIDTH)  # 假设GRASP_MAX_WIDTH已定义
        grasp_tolerance = end_points['grasp_tolerance_pred'][i]

        # print("=== objectness_score ===",objectness_score.shape)
        # print("=== grasp_score ===",grasp_score.shape)
        # print("=== grasp_center ===",grasp_center.shape)
        # print("=== approaching ===",approaching.shape)
        # print("=== grasp_angle_class_score ===",grasp_angle_class_score.shape)
        # print("=== grasp_width ===",grasp_width.shape)
        # print("=== grasp_tolerance ===",grasp_tolerance.shape)

        ## slice preds by angle
        grasp_angle_class = np.argmax(grasp_angle_class_score, axis=0)
        grasp_angle = grasp_angle_class.astype(np.float32) / 12 * np.pi
        grasp_angle_class_ = np.expand_dims(grasp_angle_class, axis=0)
        grasp_score = np.take_along_axis(grasp_score, grasp_angle_class_, axis=0).squeeze(0)
        grasp_width = np.take_along_axis(grasp_width, grasp_angle_class_, axis=0).squeeze(0)
        grasp_tolerance = np.take_along_axis(grasp_tolerance, grasp_angle_class_, axis=0).squeeze(0)

        ## slice preds by score/depth (简化了这一部分的逻辑以适应NumPy)
        grasp_depth_class = np.argmax(grasp_score, axis=1, keepdims=True)
        grasp_depth = (grasp_depth_class.astype(np.float32)+1) * 0.01
        grasp_score = np.take_along_axis(grasp_score, grasp_depth_class, axis=1).squeeze(1)
        grasp_angle = np.take_along_axis(grasp_angle, grasp_depth_class, axis=1).squeeze(1)
        grasp_width = np.take_along_axis(grasp_width, grasp_depth_class, axis=1).squeeze(1)
        grasp_tolerance = np.take_along_axis(grasp_tolerance, grasp_depth_class, axis=1).squeeze(1)

        ## slice preds by objectness
        objectness_pred = np.argmax(objectness_score, axis=0)
        objectness_mask = (objectness_pred == 1)
        grasp_score = grasp_score[objectness_mask]
        grasp_width = grasp_width[objectness_mask]
        grasp_depth = grasp_depth[objectness_mask]
        approaching = approaching[objectness_mask]
        grasp_angle = grasp_angle[objectness_mask]
        grasp_center = grasp_center[objectness_mask]
        grasp_tolerance = grasp_tolerance[objectness_mask]
        grasp_score = grasp_score * grasp_tolerance / GRASP_MAX_TOLERANCE  # 假设GRASP_MAX_TOLERANCE已定义

        ## convert to rotation matrix
        Ns = grasp_angle.size
        approaching_ = approaching.reshape(Ns, 3)
        grasp_angle_ = grasp_angle.reshape(Ns)
        rotation_matrix = batch_viewpoint_params_to_matrix_np(approaching_, grasp_angle_)  # 需要提供此函数的具体实现
        rotation_matrix = rotation_matrix.reshape(Ns, 9)

        # merge preds
        grasp_height = 0.02 * np.ones_like(grasp_score)
        obj_ids = -1 * np.ones_like(grasp_score)
        grasp_preds.append(np.concatenate([np.expand_dims(grasp_score, axis=-1),
                                           np.expand_dims(grasp_width, axis=-1),
                                           np.expand_dims(grasp_height, axis=-1),
                                           np.expand_dims(grasp_depth, axis=-1),
                                           rotation_matrix,
                                           grasp_center,
                                           np.expand_dims(obj_ids, axis=-1)], axis=-1))
    return grasp_preds