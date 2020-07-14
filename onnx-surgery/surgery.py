import functools
import onnx
import numpy as np
from onnx import helper
from onnx import numpy_helper

class Surgery(object):
    def __init__(self, onnx_model_path):
        self.model = onnx.load(onnx_model_path)

    def export(self, file_name, infer_shapes=False):
        if infer_shapes:
            self.model = onnx.shape_inference.infer_shapes(self.model)
        onnx.checker.check_model(self.model)
        onnx.save(self.model, file_name)
        
    def get_node_by_name(self, name):
        for node in self.model.graph.node:
            if node.output[0] == name:
                return node 
    #def get_node_by_name(self, name):
    #    for node in self.model.graph.node:
    #        if node.name == name:
    #            return node
                
    def insert_op_before(self, node_name, target_node, input_idx=0, *args, **kwargs):
        # get target_node inputs
        node_input = target_node.input[input_idx]
        weight_input = []
        weight_input_vi = []
        weight_initializer = []
        if "weight_dict" in kwargs:
            for weight_name, weight_numpy in kwargs["weight_dict"].items():
                weight_input.append(weight_name)
                weight_input_vi.append(
                        helper.make_tensor_value_info(
                            name=weight_name,
                            elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[weight_numpy.dtype],
                            shape=weight_numpy.shape
                        )
                )
                weight_initializer.append(
                    helper.make_tensor(
                            name=weight_name,
                            data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[weight_numpy.dtype],
                            dims=weight_numpy.shape,
                            vals=weight_numpy.tobytes(),
                            raw=True
                    )
                )
        # create new node
        new_op_node = helper.make_node(
                                kwargs["op_name"],
                                inputs=[node_input, *weight_input],
                                outputs=[node_name],
                                name=node_name,
                                **kwargs["attr_dict"]
                            )
        # set target_node input to new node outputs
        target_node.input[input_idx] = node_name
        # TODO: change other nodes input into the new node?
        # iterator all the nodes in the graph and find
        # which node's input equals the original target_node input
        # ...
        # add new node and weight input into the graph
        for target_node_index, _target_node in enumerate(self.model.graph.node):
            if _target_node == target_node:
                self.model.graph.node.insert(target_node_index, new_op_node)
                break
        self.model.graph.input.extend(weight_input_vi)
        self.model.graph.initializer.extend(weight_initializer) 
          
    def remove_node(self, target_node):
        node_input = target_node.input[0]
        node_output = target_node.output[0]
        # set input of successor node to predecessor node of target node
        for node in self.model.graph.node:
            for i, n in enumerate(node.input):
                if n == node_output:
                    node.input[i] = node_input
        target_names = set(target_node.input) & set([weight.name for weight in self.model.graph.initializer])
        self.remove_weights(target_names)
        target_names.add(node_output)
        self.remove_inputs(target_names)
        self.remove_value_infos(target_names)
        self.model.graph.node.remove(target_node)
          
    def remove_weights(self, name_list):
        rm_list = []
        for weight in self.model.graph.initializer:
            if weight.name in name_list:
                rm_list.append(weight)
        for weight in rm_list:
            self.model.graph.initializer.remove(weight)

    def remove_inputs(self, name_list):
        rm_list = []
        for input_t in self.model.graph.input:
            if input_t.name in name_list:
                rm_list.append(input_t)
        for input_t in rm_list:
            self.model.graph.input.remove(input_t)

    def remove_value_infos(self, name_list):
        rm_list = []
        for value_info in self.model.graph.value_info:
            if value_info.name in name_list:
                rm_list.append(value_info)
        for value_info in rm_list:
            self.model.graph.value_info.remove(value_info)      
