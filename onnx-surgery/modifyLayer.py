import argparse
import numpy as np
from onnx import numpy_helper
from surgery import Surgery

def add_conv_layer(onnxsu, target_node_name):
    target_node = onnxsu.get_node_by_name(target_node_name)
    weight_dict = {
         # W_from_a_new_conv_op:权重的name
         # B_from_a_new_conv_op:偏置的name
        "W_from_a_new_conv_op": np.random.normal(0, 1, (3, 3, 3, 3)).astype(np.float32),
        "B_from_a_new_conv_op": np.random.normal(0, 1, (3,)).astype(np.float32)
    }
    attr_dict = {
        # 取决于所添加层的故有属性
        "kernel_shape": [3, 3],
        "pads": [0, 0, 0, 0],
         "strides":[1, 1], 
         "dilations":[1, 1], 
         "groups":1
    }
    onnxsu.insert_op_before(
        node_name="new_conv_op", #节点output名
        target_node=target_node, #插入节点内容
        op_name="Conv",          #插入层的类型
        weight_dict=weight_dict, #卷积层的权重
        attr_dict=attr_dict      #卷积层的属性
        )

def add_relu6_layer(onnxsu, target_node_name):
    target_node = onnxsu.get_node_by_name(target_node_name)
    attr_dict = {
    }
    onnxsu.insert_op_before(
        node_name="new_conv_op", #节点output名
        target_node=target_node, #插入节点内容
        op_name="Relu6",         #插入层的类型
        attr_dict=attr_dict      #relu的属性
    )    
 
def add_slice_layer(onnxsu, target_node_name):
    target_node = onnxsu.get_node_by_name(target_node_name)
    weight_dict = {
        # inputs=['x', 'starts', 'ends', 'axes', 'steps']
        'starts': np.array([0, 0], dtype=np.int64),
        'ends': np.array([3, 10], dtype=np.int64) ,
        'axes': np.array([0, 1], dtype=np.int64),
        'steps': np.array([1, 1], dtype=np.int64)
    }
    attr_dict = {
    }
    onnxsu.insert_op_before(
        node_name="new_conv_op", #节点output名
        target_node=target_node, #插入节点内容
        op_name="Slice",          #插入层的类型
        weight_dict=weight_dict, 
        attr_dict=attr_dict      #relu的属性
    )  

def add_prelu_layer(onnxsu, target_node_name):
    target_node = onnxsu.get_node_by_name(target_node_name)
    weight_dict = {
        # inputs=['x', 'slope']
        'slope': np.random.randn(3, 4, 5).astype(np.float32),
    }
    attr_dict = {
    }
    onnxsu.insert_op_before(
        node_name="prelu-test",  #节点output名
        target_node=target_node, #插入节点内容
        op_name="PRelu",         #插入层的类型
        weight_dict=weight_dict, #输入参数
        attr_dict=attr_dict      #层的属性
    )  
               
def remove_node_layer(onnxsu, target_node_name):
    target_node = onnxsu.get_node_by_name(target_node_name)
    onnxsu.remove_node(target_node)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="onnx test")
    parser.add_argument("--input", default="../models/onnx/RFB-test-sim.onnx", type=str)
    parser.add_argument("--output", default="../models/onnx/RFB-prelu.onnx", type=str)
    args = parser.parse_args()
    onnxsu = Surgery(args.input)
    #add_conv_layer(onnxsu, "")
    #remove_node_layer(onnxsu, "288")
    add_prelu_layer(onnxsu, "246")
    #add_slice_layer(onnxsu, "287")
    onnxsu.export(args.output)
