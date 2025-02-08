#include "furthest_point_sampling.hpp"

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
    const auto& xyz_input = input(0);
    const auto& npoint_input = input(1);

    auto npoint_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(npoint_input.get_source_output().get_node_shared_ptr());
    int64_t npoint = npoint_const->cast_vector<int64_t>()[0];

    auto xyz_shape = xyz_input.get_partial_shape();
    ov::PartialShape output_shape = {xyz_shape[0], npoint};

    set_output_type(0, xyz_input.get_element_type(), output_shape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> FurthestPointSampling::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 2, "Incorrect number of new arguments");
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
    const auto& in_xyz = inputs[0];
    const auto& in_npoint = inputs[1];
    auto& out = outputs[0];

    size_t batch_size = in_xyz.get_shape()[0];
    size_t n_points = in_xyz.get_shape()[1];
    
    for (size_t b = 0; b < batch_size; ++b) {
        // get currebnt batch data
        const float* dataset = in_xyz.data<const float>() + b * n_points * 3;
        int npoint = *in_npoint.data<const int>() + b; // Get the npoint value of the current batch
        int* resultIndices = out.data<int>() + b * npoint;

        // Add the first point to the result set
        resultIndices[0] = 0;
        if (npoint == 1) continue;

        std::vector<float> distances(n_points, std::numeric_limits<float>::max());

        for (int i = 1; i < npoint; ++i) {
            float maxDist = 0;
            int farthest = 0;
            for (int j = 0; j < n_points; ++j) {
                float dist = std::pow(dataset[resultIndices[i-1] * 3] - dataset[j * 3], 2) +
                                std::pow(dataset[resultIndices[i-1] * 3 + 1] - dataset[j * 3 + 1], 2) +
                                std::pow(dataset[resultIndices[i-1] * 3 + 2] - dataset[j * 3 + 2], 2);
                distances[j] = std::min(dist, distances[j]);
                if (distances[j] > maxDist) {
                    maxDist = distances[j];
                    farthest = j;
                }
            }
            resultIndices[i] = farthest;
            distances[farthest] = 0; // Prevent the point from being selected again
        }
    }
    return true;
}

bool FurthestPointSampling::has_evaluate() const {
    return true;
}
//! [op:evaluate]