from onnx_tf.backend import prepare
import onnx

if __name__ == '__main__':
    onnx_model = onnx.load("RFB-test-sim.onnx")
    tf_rep = prepare(onnx_model, strict=False)
    tf_rep.export_graph("RFB-test-sim.pb")
