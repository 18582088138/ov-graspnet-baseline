import openvino as ov
from openvino import serialize
from openvino.runtime import Core
from openvino.frontend.onnx import OpExtension
from openvino.frontend import ConversionExtension, NodeContext

onnx_model_path = "IR_model/grasp_generator.onnx"
ov_model_path = "IR_model/grasp_generator.xml"
ov_extension_lib_path = "ov_custom_op/build/libopenvino_operation_extension.so"


import onnx
import numpy as np
from onnx.helper import make_graph, make_model, make_tensor_value_info

from openvino.frontend import FrontEndManager

fem = FrontEndManager()

def create_onnx_model_extension_with_custom_domain():
    add = onnx.helper.make_node("CustomAdd", inputs=["x", "y"], outputs=["z"], domain="custom_domain")
    const_tensor = onnx.helper.make_tensor("const_tensor",
                                           onnx.TensorProto.FLOAT,
                                           (2, 2),
                                           [0.5, 1, 1.5, 2.0])
    const_node = onnx.helper.make_node("Constant", [], outputs=["const_node"],
                                       value=const_tensor, name="const_node")
    mul = onnx.helper.make_node("Mul", inputs=["z", "const_node"], outputs=["out"])
    input_tensors = [
        make_tensor_value_info("x", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("y", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [make_tensor_value_info("out", onnx.TensorProto.FLOAT, (2, 2))]
    graph = make_graph([add, const_node, mul], "graph", input_tensors, output_tensors)
    return make_model(graph, producer_name="ONNX Frontend")

onnx_model_extension_with_custom_domain = "model_extension_custom_domain.onnx"
onnx.save_model(create_onnx_model_extension_with_custom_domain(), onnx_model_extension_with_custom_domain)

def test_onnx_op_extension_with_custom_domain():
    # use specific (openvino.frontend.onnx) import here
    from openvino.frontend.onnx import OpExtension

    fe = fem.load_by_model(onnx_model_extension_with_custom_domain)
    assert fe
    assert fe.get_name() == "onnx"

    fe.add_extension(OpExtension("opset1.Add", "CustomAdd", "custom_domain", {}, {"auto_broadcast": "numpy"}))
    input_model = fe.load(onnx_model_extension_with_custom_domain)
    assert input_model
    model = fe.convert(input_model)
    assert model
    print("=========Test op_extension Pass==========")

test_onnx_op_extension_with_custom_domain()


def test_onnx_conversion_extension_with_custom_domain():
    # use specific (openvino.frontend.onnx) import here
    from openvino.frontend.onnx import ConversionExtension
    from openvino.frontend import NodeContext
    import openvino.runtime.opset8 as ops

    fe = fem.load_by_model(onnx_model_extension_with_custom_domain)
    assert fe
    assert fe.get_name() == "onnx"

    invoked = False

    def custom_converter(node: NodeContext):
        nonlocal invoked
        invoked = True
        input_1 = node.get_input(0)
        input_2 = node.get_input(1)
        add = ops.add(input_1, input_2)
        return [add.output(0)]

    fe.add_extension(ConversionExtension("CustomAdd", "custom_domain", custom_converter))
    input_model = fe.load(onnx_model_extension_with_custom_domain)
    assert input_model
    model = fe.convert(input_model)
    assert model
    assert invoked
    print("=========Test conversion_extension Pass==========")

test_onnx_conversion_extension_with_custom_domain()

# core = Core()
# print("===== Loading extension lib ======")
# core.add_extension(ov_extension_lib_path)
# core.add_extension(OpExtension("Identity", "CylinderQuery", "custom_domain"))
# # core.add_extension(OpExtension("CylinderQuery", "CylinderQuery", "custom_domain"))
# # core.add_extension(OpExtension("GroupingOperation", "GroupingOperation", "custom_domain"))

# print("===== Loading OV model ======")
# ov_model = core.read_model(onnx_model_path)
# print("===== compile OV model ======")
# ov_compiled_model = core.compile_model(ov_model, 'CPU')
# print("===== serialize OV model ======")
# serialize(ov_compiled_model, 'IR_model/ov_grasp_generator.xml')
# print("== export ov_graspnet IR success ==")