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
st.markdown("<h1 style='text-align: center; color: #F78181;'>Where is Wally?</h1>", unsafe_allow_html=True)
st.write('The green square displays where the algorithm located Wally. ')

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
    st_canvas()

image = Image.open(imgs[index])
st.image(image, use_column_width=True)

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])


# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=1,
    stroke_color=stroke_color,
    #background_color=bg_color,
    background_image=Image.open("images/1.jpg"),#  if bg_image else None,
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
