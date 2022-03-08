import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import glob
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import argparse
import sys
from matplotlib import pyplot as plt
import matplotlib
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#---STREAMLIT INTERFACE
st.markdown("<h1 style='text-align: center; color: red;'>Where is Wally?</h1>", unsafe_allow_html=True)
st.write('Some explanation text goes here.')

#---SIDEBAR
#realtime_update = st.checkbox("Update in realtime", True)

#--ADD SOMETHING HERE TO GET THE IMAGE+GUESS FROM THE MODEL

#---GET IMAGE
imgs = glob.glob("images/*.jpg") #just the base images right now

# Save model path
model_path = './trained_model/frozen_inference_graph.pb'

#Create functions
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

#Save variables 
label_map = label_map_util.load_labelmap('./trained_model/labels.txt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

index = int()
if not index:
    index = 0


with detection_graph.as_default():
  with tf.compat.v1.Session(graph=detection_graph) as sess:
    parser = argparse.ArgumentParser()
    #parser.add_argument('image_path')
    #args = parser.parse_args()
    image_np = load_image_into_numpy_array(Image.open(imgs[index]))
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: np.expand_dims(image_np, axis=0)})

    #print(scores[0][0])

    if scores[0][0] < 0.1:
        sys.exit('Wally not found :(')

    #print('Wally found')
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=5)
    plt.figure(figsize=(12, 8))
    plt.imshow(image_np)
    plt.show()
    plt.savefig(f'{imgs[index]}.png')


#Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",     # Fixed fill color with some opacity
    background_image=Image.open(f'{imgs[index]}.png') if f'{imgs[index]}.png' else None,
    update_streamlit=False,
    height=500,
    width=700,
    drawing_mode='rect',
    point_display_radius= 0,
    key="canvas",
    stroke_width= 1,
)


if st.button('Save'):
    index+=1

<<<<<<< HEAD
=======
image = Image.open(imgs[index])
st.image(image, use_column_width=True)


# Create a canvas component
<<<<<<< HEAD
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    #background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=500,
    width=700,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    st.dataframe(objects)

def save_corrected_data(dataframe):
    dataframe.to_csv('data/hejehj.csv')

if st.button('save'):
    save_corrected_data(objects)
=======
#canvas_result = st_canvas(
#    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
#    background_image=Image.open(img) if img else None,
#    update_streamlit=realtime_update,
#    height=500,
#    width=700,
#    drawing_mode='rect',
#    point_display_radius= 0,
#    key="canvas",
#)

>>>>>>> db387aee3ff9b3816e626c81345faae12df3d51a

# Do something interesting with the image data and paths
>>>>>>> 88df59f71a339d1156046287062ac6688e35ad54
