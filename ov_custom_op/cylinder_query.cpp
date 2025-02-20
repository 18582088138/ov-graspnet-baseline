#include "cylinder_query.hpp"
#include "openvino/op/constant.hpp"
#include <iostream>

using namespace TemplateExtension;

//! [op:ctor]
CylinderQuery::CylinderQuery(const ov::Output<ov::Node>& new_xyz, const ov::Output<ov::Node>& xyz, 
                const ov::Output<ov::Node>& rot, const ov::Output<ov::Node>& radius, 
                const ov::Output<ov::Node>& hmin, const ov::Output<ov::Node>& hmax, 
                const ov::Output<ov::Node>& nsample) : Op({new_xyz, xyz, rot, radius, hmin, hmax, nsample}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void CylinderQuery::validate_and_infer_types() {
    /*
    Parameters
    ----------
    radius : float
        radius of the cylinders
    hmin, hmax : float
        endpoints of cylinder height in x-rotation axis
    nsample : int
        maximum number of features in the cylinders
    xyz : torch.Tensor
        (B, N, 3) xyz coordinates of the features
    new_xyz : torch.Tensor
        (B, npoint, 3) centers of the cylinder query
    rot: torch.Tensor
        (B, npoint, 9) flatten rotation matrices from
                        cylinder frame to world frame
    Returns
    -------
    torch.Tensor
        (B, npoint, nsample) tensor with the indicies of the features that form the query balls
    */
    const auto& new_xyz = input(0);
    const auto& xyz = input(1);
    const auto& rot = input(2);
    const auto& radius = input(3);
    const auto& hmin = input(4);
    const auto& hmax = input(5);
    const auto& nsample = input(6); // There is no way to get the value of the input parameter during the initialization phase.

    // auto nsample_const = ov::as_type_ptr<ov::op::v0::Constant>(nsample.get_source_output().get_node_shared_ptr());
    // if (!nsample_const) {
    //     std::cout << "Input is not a Constant node." << std::endl;
    // }
    auto new_xyz_shape = new_xyz.get_partial_shape();
    ov::PartialShape output_shape = {new_xyz_shape[0], new_xyz_shape[1], -1}; //64 is only a temporary setting. The value of output shape needs to be updated during inference.

    set_output_type(0, ov::element::i32, output_shape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> CylinderQuery::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    // OPENVINO_ASSERT(new_args.size() == 2, "Incorrect number of new arguments");

    return std::make_shared<CylinderQuery>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), new_args.at(4), new_args.at(5), new_args.at(6));
}
//! [op:copy]

//! [op:visit_attributes]
bool CylinderQuery::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool CylinderQuery::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const float* new_xyz = inputs[0].data<const float>();
    const float* xyz = inputs[1].data<const float>();
    const float* rot = inputs[2].data<const float>();
    const float radius = *inputs[3].data<const float>();
    const float hmin = *inputs[4].data<const float>();
    const float hmax = *inputs[5].data<const float>();
    const int nsample = *inputs[6].data<const int>();


    int b = inputs[1].get_shape()[0]; // batch size
    int n = inputs[1].get_shape()[1]; // number of points in xyz
    int npoint = inputs[0].get_shape()[1]; // number of points in new_xyz
    // int m = inputs[0].get_shape()[1];

    ov::PartialShape output_shape = {b, npoint, nsample};
    outputs[0].set_shape(output_shape.to_shape());

    auto& out_tensor = outputs[0];
    // std::cout<<"==== out_tensor shape: ===="<<out_tensor.get_shape()<<std::endl;
    int* idx = out_tensor.data<int>();

    // 预先计算半径的平方
    float radius2 = radius * radius;

    // 遍历每个批次
    for (int batch_index = 0; batch_index < b; ++batch_index) {
        // 计算当前批次中xyz, new_xyz, rot 和 idx 的起始位置
        const float* current_xyz = xyz + batch_index * n * 3;
        const float* current_new_xyz = new_xyz + batch_index * npoint * 3;
        const float* current_rot = rot + batch_index * npoint * 9;
        int* current_idx = idx + batch_index * npoint * nsample;

        // 遍历每个新点
        for (int j = 0; j < npoint; ++j) {
            // 获取当前点坐标和旋转矩阵
            float new_x = current_new_xyz[j * 3 + 0];
            float new_y = current_new_xyz[j * 3 + 1];
            float new_z = current_new_xyz[j * 3 + 2];
            float r[9] = {
                current_rot[j * 9 + 0], current_rot[j * 9 + 1], current_rot[j * 9 + 2],
                current_rot[j * 9 + 3], current_rot[j * 9 + 4], current_rot[j * 9 + 5],
                current_rot[j * 9 + 6], current_rot[j * 9 + 7], current_rot[j * 9 + 8]
            };

            int cnt = 0;
            // 遍历每个原始点
            for (int k = 0; k < n && cnt < nsample; ++k) {
                // 计算点相对于新点的位置，并应用旋转
                float x = current_xyz[k * 3 + 0] - new_x;
                float y = current_xyz[k * 3 + 1] - new_y;
                float z = current_xyz[k * 3 + 2] - new_z;
                float x_rot = r[0] * x + r[3] * y + r[6] * z;
                float y_rot = r[1] * x + r[4] * y + r[7] * z;
                float z_rot = r[2] * x + r[5] * y + r[8] * z;

                // 判断是否在圆柱体内
                float d2 = y_rot * y_rot + z_rot * z_rot;
                if (d2 < radius2 && x_rot > hmin && x_rot < hmax) {
                    // 如果这是第一个找到的点，则填充所有索引为当前点的索引
                    if (cnt == 0) {
                        for (int l = 0; l < nsample; ++l) {
                            current_idx[j * nsample + l] = k;
                        }
                    }
                    current_idx[j * nsample + cnt] = k;
                    ++cnt;
                }
            }

            // 如果找到的点少于nsample，则填充剩余索引为最后一个有效索引或初始时设置的所有索引为同一个值
            // while (cnt < nsample) {
            //     current_idx[j * nsample + cnt] = (cnt == 0) ? -1 : current_idx[j * nsample + cnt - 1];
            //     ++cnt;
            // }
        }
    }
    // out.set_shape(in.get_shape());  //TODO : need to implememtation the output shape update, reference dynamic shape.
    // memcpy(out.data(), in.data(), in.get_byte_size());
    return true;
}

bool CylinderQuery::has_evaluate() const {
    return true;
}