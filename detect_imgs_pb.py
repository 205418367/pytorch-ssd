"""
This code uses the onnx model to detect faces from live video or cameras.
requirtment:
   1.tensorflow
   2.torch
"""
import os
import time
import tensorflow as tf
from tensorflow.python.platform import gfile
import cv2
import numpy as np
import onnx
import vision.utils.box_utils_numpy as box_utils


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


label_path = "models/voc-model-labels.txt"
model_path = "models/onnx/RFB-test-sim.pb"
class_names = [name.strip() for name in open(label_path).readlines()]

with gfile.FastGFile(model_path,'rb') as f:
         graph_def = tf.GraphDef()
         graph_def.ParseFromString(f.read())

configer = tf.ConfigProto()
_ = tf.import_graph_def(graph_def, name="")
result_path = "./"

threshold = 0.7
path = "imgs"
sum = 0
if not os.path.exists(result_path):
    os.makedirs(result_path)
listdir = os.listdir(path)
sum = 0

with tf.Session() as sess:
     sess.run(tf.global_variables_initializer())
     input_ = sess.graph.get_tensor_by_name('input:0')
     scores_ = sess.graph.get_tensor_by_name('scores:0')
     boxes_ = sess.graph.get_tensor_by_name('boxes:0')
     for file_path in listdir:
        img_path = os.path.join(path, file_path)
        orig_image = cv2.imread(img_path)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        # image = cv2.resize(image, (640, 480))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        # confidences, boxes = predictor.run(image)
        time_time = time.time()
        feed_dict = {input_: image}
        confidences, boxes = sess.run([scores_, boxes_], feed_dict=feed_dict)
        print("cost time:{}".format(time.time() - time_time))
        boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            cv2.imwrite(os.path.join(result_path, file_path), orig_image)
        sum += boxes.shape[0]
print("sum:{}".format(sum))
