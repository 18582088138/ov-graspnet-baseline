#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>

#include "furthest_point_sampling.hpp"
#include "gather_operation.hpp"
#include "three_nn.hpp"
#include "three_interpolate.hpp"
#include "cylinder_query.hpp"
#include "ball_query.hpp"
#include "grouping_operation.hpp"
#include "custom_add.hpp"

// clang-format off
//! [ov_extension:entry_point]
OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        // Register operation itself, required to be read from IR
        std::make_shared<ov::OpExtension<TemplateExtension::FurthestPointSampling>>(),
        // Register operaton mapping, required when converted from framework model format
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::FurthestPointSampling>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::GatherOperation>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::GatherOperation>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::ThreeNN>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::ThreeNN>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::ThreeInterpolate>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::ThreeInterpolate>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::GroupingOperation>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::GroupingOperation>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::BallQuery>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::BallQuery>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::CylinderQuery>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::CylinderQuery>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::CustomAdd>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::CustomAdd>>(),
        
    }));
//! [ov_extension:entry_point]
// clang-format on