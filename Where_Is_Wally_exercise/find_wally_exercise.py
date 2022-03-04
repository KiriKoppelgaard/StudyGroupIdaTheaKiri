from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
import matplotlib
from PIL import Image
import matplotlib.patches as patches
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import argparse
import glob

# import model
model_path = './trained_model/frozen_inference_graph.pb'

# import images
imgs = glob.glob("images/*.jpg")
imgs.sort()

# path idx
idx = len("images/")

# prepare dataframe
df = pd.DataFrame(columns = ["image", "result", "score", "position"])

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(model_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

label_map = label_map_util.load_labelmap('./trained_model/labels.txt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

for img in imgs:
    with detection_graph.as_default():
      with tf.compat.v1.Session(graph=detection_graph) as sess:
        parser = argparse.ArgumentParser()
        #parser.add_argument('image_path')
        #args = parser.parse_args()
        image_np = load_image_into_numpy_array(Image.open(img)) #args.image_path))
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: np.expand_dims(image_np, axis=0)})

        print(img)

        bboxes = boxes[scores > 0.2] #min_score_thresh
        width, height = image_np.shape[:2]
        pos = []
        for box in bboxes:
            ymin, xmin, ymax, xmax = box
            pos = [np.mean([int(xmin * height), int(xmax * height)]), np.mean([int(ymin * width), int(ymax * width)])]#[int(xmin * height), int(ymin * width), int(xmax * height), int(ymax * width)]

        if scores[0][0] < 0.1:
            result = 'Wally not found'
        else: 
            result = 'Wally found'
        
        df = df.append({
            "image": img[idx:-4],
            "result": result,
            "score": scores[0][0],
            "position": pos}, ignore_index = True)
        
print(df)

    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     image_np,
    #     np.squeeze(boxes),
    #     np.squeeze(classes).astype(np.int32),
    #     np.squeeze(scores),
    #     category_index,
    #     use_normalized_coordinates=True,
    #     line_thickness=8)
    # plt.figure(figsize=(12, 8))
    # plt.imshow(image_np)
    # plt.show()
