import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import glob


#---STREAMLIT INTERFACE
st.markdown("<h1 style='text-align: center; color: red;'>Where is Wally?</h1>", unsafe_allow_html=True)
st.write('Some explanation text goes here.')

#---SIDEBAR
realtime_update = st.sidebar.checkbox("Update in realtime", True)

#--ADD SOMETHING HERE TO GET THE IMAGE+GUESS FROM THE MODEL

#---GET IMAGE
imgs = glob.glob("images/*.jpg") #just the base images right now
bg_image = imgs[0] #just the first image

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.25)",  # Fixed fill color with some opacity
    stroke_width=1.5,
    stroke_color = 'yellow',
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=500,
    width=700,
    drawing_mode='rect',
    point_display_radius= 0,
    key="canvas",
)

# Do something interesting with the image data and paths
