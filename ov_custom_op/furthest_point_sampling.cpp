#include "furthest_point_sampling.hpp"
#include "openvino/op/constant.hpp"

using namespace TemplateExtension;

//! [op:ctor]
FurthestPointSampling::FurthestPointSampling(const ov::Output<ov::Node>& xyz, const ov::Output<ov::Node>& npoint) 
: Op({xyz, npoint}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void FurthestPointSampling::validate_and_infer_types() {
    // Operation doesn't change shapes end element type
    /*
    Parameters
    ----------
    xyz : torch.Tensor
        (B, N, 3) tensor where N > npoint
    npoint : int32
        number of features in the sampled set

    Returns
    -------
    torch.Tensor
        (B, npoint) tensor containing the set
    */
    const auto& xyz = input(0);
    int npoint = -1;   //dynamic shape
    auto xyz_shape = xyz.get_partial_shape();
    ov::PartialShape output_shape = {xyz_shape[0], npoint};
    set_output_type(0, ov::element::i32, output_shape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> FurthestPointSampling::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    // OPENVINO_ASSERT(new_args.size() == 2, "Incorrect number of new arguments");
    return std::make_shared<FurthestPointSampling>(new_args.at(0), new_args.at(1));
}
//! [op:copy]

//! [op:visit_attributes]
bool FurthestPointSampling::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool FurthestPointSampling::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const float* xyz = inputs[0].data<const float>();
    const int npoint = *inputs[1].data<const int>();

    int b = inputs[0].get_shape()[0]; // batch size
    int n = inputs[0].get_shape()[1]; // number of points in xyz

    ov::PartialShape output_shape = {b, npoint};
    outputs[0].set_shape(output_shape.to_shape());

    auto& out_tensor = outputs[0];
    int *out_data = out_tensor.data<int>();

    if (npoint <= 0) {
        return false;
    }

    for (int batch_index = 0; batch_index < b; ++batch_index) {
        // 每个batch中的起始位置
        const float *current_dataset = xyz + batch_index * n * 3;
        int *current_idxs = out_data + batch_index * npoint;

        // 初始化temp数组为最大值，表示初始时每个点到已选点集的距离未知或无限大
        std::vector<float> temp(n, std::numeric_limits<float>::max());

        // 初始化第一个点
        current_idxs[0] = 0;
        for (int j = 1; j < npoint; ++j) {
            int besti = 0;
            float best = -std::numeric_limits<float>::max();
            float x1 = current_dataset[current_idxs[j - 1] * 3 + 0];
            float y1 = current_dataset[current_idxs[j - 1] * 3 + 1];
            float z1 = current_dataset[current_idxs[j - 1] * 3 + 2];

            // 计算每个点的距离，并找到最远的点
            for (int k = 0; k < n; ++k) {
                float x2 = current_dataset[k * 3 + 0];
                float y2 = current_dataset[k * 3 + 1];
                float z2 = current_dataset[k * 3 + 2];
                float mag = x2 * x2 + y2 * y2 + z2 * z2;
                if (mag <= 1e-3) continue;

                float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
                float d2 = std::min(d, temp[k]);
                temp[k] = d2;
                if (d2 > best) {
                    best = d2;
                    besti = k;
                }
            }

            // 更新下一个选择的点
            current_idxs[j] = besti;
        }
    }

    // out.set_shape(in.get_shape());
    // memcpy(out.data(), in.data(), in.get_byte_size());
    return true;
}

bool FurthestPointSampling::has_evaluate() const {
    return true;
}
//! [op:evaluate]