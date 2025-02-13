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
    add = onnx.helper.make_node("CustomAdd", inputs=["x", "y"], outputs=["z"])
    # add = onnx.helper.make_node("CustomAdd", inputs=["x", "y"], outputs=["z"], domain="custom_domain")
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
    # fe.add_extension(OpExtension("opset1.Add", "CustomAdd", "custom_domain", {}, {"auto_broadcast": "numpy"}))
    fe.add_extension(OpExtension("opset1.Add", "CustomAdd", {}, {"auto_broadcast": "numpy"}))
    input_model = fe.load(onnx_model_extension_with_custom_domain)
    assert input_model
    model = fe.convert(input_model)
    assert model
    print("=========Test op_extension Pass==========")

# test_onnx_op_extension_with_custom_domain()


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
    # fe.add_extension(ConversionExtension("CustomAdd", "custom_domain", custom_converter))
    fe.add_extension(ConversionExtension("CustomAdd", custom_converter))
    input_model = fe.load(onnx_model_extension_with_custom_domain)
    assert input_model
    model = fe.convert(input_model)
    assert model
    assert invoked
    print("=========Test conversion_extension Pass==========")

# test_onnx_conversion_extension_with_custom_domain()

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



onnx_grasp_generator_path = "IR_model/grasp_generator.onnx"

def print_model_info(model_path):
    # 加载模型
    model = onnx.load(model_path)
    
    # 遍历模型中的每个节点
    for node in model.graph.node:
        print(f"Node name: {node.name}")
        print(f"Op type: {node.op_type}")
        
        # 打印每个输入的名称和形状
        for input_name in node.input:
            tensor_shape, tensor_type = get_tensor_info(model, input_name)
            print(f"  Input: {input_name}, Shape: {tensor_shape}, Type: {tensor_type}")
        
        # 打印每个输出的名称和形状
        for output_name in node.output:
            tensor_shape, tensor_type = get_tensor_info(model, output_name)
            print(f"  Output: {output_name}, Shape: {tensor_shape}, Type: {tensor_type}")
        
        print("-" * 40)

def get_tensor_info(model, tensor_name):
    """
    根据张量名称获取其形状和数据类型。
    :param model: ONNX 模型对象
    :param tensor_name: 张量名称
    :return: 形状和数据类型的元组
    """
    tensor_shape = "Unknown"
    tensor_type = "Unknown"
    
    # 查找对应张量的值信息(ValueInfoProto)或初始化器(Initializer)
    for value_info in model.graph.value_info:
        if value_info.name == tensor_name:
            tensor_shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            tensor_type = onnx.TensorProto.DataType.Name(value_info.type.tensor_type.elem_type)
            return tensor_shape, tensor_type
    
    for initializer in model.graph.initializer:
        if initializer.name == tensor_name:
            tensor_shape = initializer.dims
            tensor_type = onnx.TensorProto.DataType.Name(initializer.data_type)
            return tensor_shape, tensor_type
    
    return tensor_shape, tensor_type

# 使用示例
print_model_info(onnx_grasp_generator_path)

model = onnx.load(onnx_grasp_generator_path)
graph =model.graph

# 遍历图中的每个节点(Node)，即每个操作(op)
for node in model.graph.node:
    print("Node name: ", node.name)
    print("Node op type: ", node.op_type)
    
    # 打印输入输出信息
    for idx, input_name in enumerate(node.input):
        print(f"Input {idx} name: {input_name}, data type: To be determined")
    
    for idx, output_name in enumerate(node.output):
        print(f"Output {idx} name: {output_name}, data type: To be determined")
    
    print("-"*30)

# 如果需要获取详细的shape和数据类型，可以检查graph的initializer和value_info
# 下面是如何找到对应输入输出的shape和数据类型的示例
for value_info in model.graph.value_info:
    print("Value info name: ", value_info.name)
    print("Shape: ", [dim.dim_value for dim in value_info.type.tensor_type.shape.dim])
    print("Data type: ", value_info.type.tensor_type.elem_type)  # 这个是整数形式的数据类型

for initializer in model.graph.initializer:
    print("Initializer name: ", initializer.name)
    print("Shape: ", initializer.dims)
    print("Data type: ", initializer.data_type)