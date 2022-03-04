import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import glob
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


#---STREAMLIT INTERFACE
st.markdown("<h1 style='text-align: center; color: red;'>Where is Wally?</h1>", unsafe_allow_html=True)
st.write('Some explanation text goes here.')

#---SIDEBAR
realtime_update = st.sidebar.checkbox("Update in realtime", True)

#--ADD SOMETHING HERE TO GET THE IMAGE+GUESS FROM THE MODEL

#---GET IMAGE
imgs = glob.glob("images/*.jpg") #just the base images right now
iterator_imgs = iter(imgs)
#bg_image = imgs[0] #just the first image

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
'''
n = 0
for p in range(0, len(imgs)): 
    while p == p:
        st.write(n)
        st.image(imgs[n])
        if st.button(imgs[n], key = p): 
            continue
'''
#images = glob.glob("/path/to/images/")
index= 0 #st.number_input('Index')

if st.button('Save'):
    index+=1

image = Image.open(imgs[index])
st.image(image, use_column_width=True)


# Create a canvas component
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


# Do something interesting with the image data and paths
