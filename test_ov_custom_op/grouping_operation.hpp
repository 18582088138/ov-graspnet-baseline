#pragma once

//! [op:common_include]
#include <openvino/op/op.hpp>
#include "openvino/op/constant.hpp"
//! [op:common_include]

//! [op:header]
namespace TemplateExtension {

class GroupingOperation : public ov::op::Op {
    private:
    int32_t  m_b;
    int32_t  m_c;
    int32_t  m_n;
    int32_t  m_npoints;
    int32_t  m_nsample;
public:
    OPENVINO_OP("GroupingOperation");

    GroupingOperation() = default;
    GroupingOperation(const ov::Output<ov::Node>& features, const ov::Output<ov::Node>& idx, 
                                    int32_t bx, int32_t  c, int32_t  n, int32_t  npoints, int32_t  nsample) ;
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
// private:
//     bool evaluateCPU(ov::TensorVector& outputs, const ov::TensorVector& inputs) const;
//     bool evaluateGPU(ov::TensorVector& outputs, const ov::TensorVector& inputs) const;
//     bool use_gpu = false;
};
//! [op:header]

}  // namespace TemplateExtension