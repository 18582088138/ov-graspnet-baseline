#pragma once

//! [op:common_include]
#include <openvino/op/op.hpp>
#include "openvino/op/constant.hpp"
#include <cstring>
#include <vector>
#include <cmath>
#include <limits>
//! [op:common_include]

//! [op:header]
namespace TemplateExtension {

class CylinderQuery : public ov::op::Op {
public:
    OPENVINO_OP("CylinderQuery");

    CylinderQuery() = default;
    CylinderQuery(const ov::Output<ov::Node>& new_xyz, const ov::Output<ov::Node>& xyz, 
                const ov::Output<ov::Node>& rot, const ov::Output<ov::Node>& radius, 
                const ov::Output<ov::Node>& hmin, const ov::Output<ov::Node>& hmax, 
                const ov::Output<ov::Node>& nsample);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
//! [op:header]

}  // namespace TemplateExtension