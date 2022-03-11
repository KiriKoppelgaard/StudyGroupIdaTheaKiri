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
st.markdown("<h1 style='text-align: center; color: black;'>Where is Wally?</h1>", unsafe_allow_html=True)

#Add instructions
with st.expander('Help! This is my first time', expanded=False): 
    st.write("The green box indicates where the algorithm located Wally. If you agree, you can move on to the next image. If you disagree, you can correct the location by placing a square around Wally's face. Click and drag the mouse to make a square.")

#---SIDEBAR
#realtime_update = st.checkbox("Update in realtime", True)

#--ADD SOMETHING HERE TO GET THE IMAGE+GUESS FROM THE MODEL

#---GET IMAGE
imgs = glob.glob("images/*.jpg") #just the base images right now

# Save model path
model_path = './trained_model/frozen_inference_graph.pb'

#plots in columns:
left_column_upper, mid_column_upper,midr_column_upper, right_column_upper = st.columns(4)

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

#Initialise index
if 'index' not in st.session_state:
	st.session_state.index = 0

#Create save button 
with right_column_upper:
    save = st.button('Next image', help='Saving the image and loading the next. You can always go back and make corrections.')
if save:
    st.session_state.index += 1
with left_column_upper:
    previous = st.button('Previous image', help='Go back to the previous image')
if previous:
    st.session_state.index -= 1


with detection_graph.as_default():
  with tf.compat.v1.Session(graph=detection_graph) as sess:
    parser = argparse.ArgumentParser()
    #parser.add_argument('image_path')
    #args = parser.parse_args()
    image_np = load_image_into_numpy_array(Image.open(imgs[st.session_state.index]))
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
    plt.savefig(f'{imgs[st.session_state.index]}.png')

current_score = round(100 * float(scores[0][0]), 1)
if current_score > 90: 
    st.subheader(f'The algorithm is {current_score} % certain!✅')
    
elif current_score < 90 and current_score > 30:
    st.subheader(f'The algorithm is {current_score} % certain!🤔')
else: 
    st.subheader(f'The algorithm is {current_score} % certain! ⛔')
    
#Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)", # Fixed fill color with some opacity
    background_image=Image.open(f'{imgs[st.session_state.index]}.png') if f'{imgs[st.session_state.index]}.png' else None,
    update_streamlit=False,
    height=500,
    width=750,
    drawing_mode='rect',
    point_display_radius= 0,
    key="canvas",
    stroke_width= 2,
)

# Do something interesting with the image data and paths
#if canvas_result.image_data is not None:
#    st.image(canvas_result.image_data)
#if canvas_result.json_data is not None:
#    objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
#    for col in objects.select_dtypes(include=['object']).columns:
#        objects[col] = objects[col].astype("str")
#    st.dataframe(objects)


#Create function to save positions of drawing
def save_corrected_data(dataframe):
    dataframe.to_csv('data/hybrid_data.csv')
left_column_lower, mid_column_lower, right_column_lower = st.columns(3)
with mid_column_lower:
    if st.button('Export hybrid intelligent data', help='Exports the updated coordinates as a .csv and stores it in the data folder'):
        save_corrected_data(objects)

#Add instructions
with st.expander('Learn more about the design', expanded=False): 
    st.image('flowchart.png')
    st.markdown("<h1 style='text-align: center; color: black;'>HOW OUR SOLUTION RELATES TO THE FOUR AKATA CRITERIA</h1>", unsafe_allow_html=True)

    text = '''
    \n**Collaborative**
    \n Our collaborative solution combines the strengths of:
    \n- _the AI (the neural network):_ quickly identify the point in a big image which is most likely to contain Wally + report certainty
    \n- _the human_: quickly and intuitively check doubt cases, and correct the AI where it’s wrong. Human task is smaller since the big hurdle of scanning a full detailed image has been outsourced to the computer, and the human can focus on doubt cases.
    This human-computer collaboration can quickly and accurately locate Wally in a large selection of images.
    \n**Adaptive**
    \n Our solution could be extended to adapt to the user input.
    \n - _Synergy_: the feedback from the human (where was Wally actually?) could ideally be fed back into the network (as a ‘true label’) to make it perform better over time!
    Potentially, it could also utilize transfer learning and adapt to find other targets - for example Wanda - by using user input!
    \n**Responsible**
    \nLegal and moral values aren’t really relevant for this specific case, and no ethics are involved.
    \n**Explainable**
    \n Our solution could be expanded to be more explainable. Ideally, the NN should be able to say WHY it thinks Wally is in a particular place.
    For example by highlighting the pixels of the guess that makes it most confident that this is Wally.
        e.g. maybe it’s the striped shirt, and not the face.

    '''
    st.write(text)

