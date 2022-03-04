#from asyncio.windows_events import NULL
import streamlit as st
import pandas as pd
import plotly.express as px
import plotnine
import numpy as np
import random
import os

#KRISTIAN: MINIMUM 3-4 PLOTS/DATA VISUALISATIONS

#configuration
st.set_page_config(layout = 'wide') #wide mode

#load data
df = pd.read_csv('Eisenberg_2019_data_compiled.csv')

#rename variables
df["Sex"].replace({0: "Male", 1: "Female"}, inplace=True)

#make lists of unique options for later use in selectboxes
races = df['Race'].unique()
races = np.insert(races, 0, 'All') #add choose all option to front of list
sexes = df['Sex'].unique()
sexes = np.insert(sexes, 0, 'All') #add choose all option to front of list

#SIDEBAR STUFF:
st.sidebar.write('Choose which data subsets to explore:')
sex = st.sidebar.selectbox('Sex', sexes)
race = st.sidebar.selectbox('Race', races)

st.sidebar.write('Choose how to color the plots:')
color_by = st.sidebar.selectbox('Color by', options=['Sex', 'Race', 'HighestEducation', 'Nothing'])

st.markdown("<h1 style='text-align: center; color: blue;'>How do you become the most mindful person? 💆</h1>", unsafe_allow_html=True)
st.markdown("<h8 style='text-align: center; color: black;'>Explore the different variables and see if the answer lies in the data</h8>", unsafe_allow_html=True)



#plots in columns:
left_column, right_column = st.columns(2)

#customise what's plotted?:
dfplot = df
#Sex
if sex != "All":
    dfplot = df[df["Sex"] == sex]
#Race
if race != "All":
    dfplot = dfplot[dfplot["Race"] == race]



yplot = 'five_facet_mindfulness_survey.act_with_awareness'
xplot = 'HouseholdIncome'

with left_column:
    st.markdown("<h5 style='text-align: center; color: blue;'>Household Income 🤑</h1>", unsafe_allow_html=True)
    #line plot
    fig = px.scatter(dfplot, x= xplot, y= yplot, title='How does income affect mindfulness?', trendline="ols", labels= {"five_facet_mindfulness_survey.act_with_awareness": "Mindfulness"})        
    if color_by!='Nothing':
        fig = px.scatter(dfplot, x= xplot, y= yplot, color=color_by, title='How does income affect mindfulness?', trendline="ols", labels= {"five_facet_mindfulness_survey.act_with_awareness": "Mindfulness"})
    st.plotly_chart(fig, use_container_width=True) #display the plot

yplot = 'five_facet_mindfulness_survey.act_with_awareness'
xplot = 'CoffeeCupsPerDay'

with right_column:
    st.markdown("<h5 style='text-align: center; color: blue;'>Coffee ☕️</h1>", unsafe_allow_html=True)
    #scatter plot
    fig = px.box(dfplot, x= xplot, y= yplot, title='How does coffee intake affect mindfulness?', labels= {"five_facet_mindfulness_survey.act_with_awareness": "Mindfulness"})        
    if color_by!='Nothing':
        fig = px.box(dfplot, x= xplot, y= yplot, color=color_by,title='How does coffee intake affect mindfulness?', labels= {"five_facet_mindfulness_survey.act_with_awareness": "Mindfulness"})
    st.plotly_chart(fig, use_container_width=True) #display the plot

st.write('')
st.markdown("<h5 style='text-align: center; color: blue;'>👈 Use the sidebar to choose what to color by</h1>", unsafe_allow_html=True)


#childrr
#child = 'ChildrenNumber'
#with st.expander("Children tab"):
#    st.markdown("<h5 style='text-align: center; color: blue;'>Number of children</h1>", unsafe_allow_html=True)
#    no = st.slider("slide for children number", min_value=0, max_value=5, step=1)
#    st.markdown(no)
#    #scatter plot
#    fig = px.box(dfplot, x= child, y= yplot, title='How does number of children affect mindfulness?', labels= {"five_facet_mindfulness_survey.act_with_awareness": "Mindfulness"})        
#    if color_by!='Nothing':
#        fig = px.box(dfplot, x= child, y= yplot, color=color_by,title='How does number of children affect mindfulness?', labels= {"five_facet_mindfulness_survey.act_with_awareness": "Mindfulness"})
#    st.plotly_chart(fig, use_container_width=True) #display the plot


title = st.text_input('What is your conclusion?', 'welcome')
os.system('say ' +title)


st.markdown("<h8 style='text-align: left; color: purple;'>Check out the full data set here:</h1>", unsafe_allow_html=True)
with st.expander("Show full data set"):
    st.write(df)


#add some more cool plots, make them interactive - maybe with slider