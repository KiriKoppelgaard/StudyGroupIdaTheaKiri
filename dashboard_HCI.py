import streamlit as st
import pandas as pd
import plotly.express as px
import plotnine
import numpy as np
import random

#KRISTIAN: MINIMUM 3-4 PLOTS/DATA VISUALISATIONS

#configuration
st.set_page_config(layout = 'wide') #wide mode

#load data
df = pd.read_csv('Eisenberg_2019_data_compiled.csv')

#rename variables
df["Sex"].replace({0: "Male", 1: "Female"}, inplace=True)

#make lists of unique options for later use in selectboxes
races = df['Race'].unique()
sexes = df['Sex'].unique()

#SIDEBAR STUFF:
st.sidebar.write('Choose which data subsets to explore (NOT IMPLEMENTED):')
st.sidebar.selectbox('Sex', sexes)
st.sidebar.selectbox('Race', races)

st.sidebar.write('Choose how to color the plots:')
color_by = st.sidebar.selectbox('Color by', options=['Sex', 'Race', 'HighestEducation', 'Nothing'])

#INITIAL TEXT:
st.title('Welcome!')
st.write('##### Here you can explore the Eisenberg data set. Use the sidebar to blablabla.')

with st.expander("Show full data set"):
    st.write(df)



st.write('##### Here are some cool plots. Use the sidebar to choose what to color by:')
#plots in columns:
left_column, right_column = st.columns(2)

#customise what's plotted?:
dfplot = df #could make it a subset of their choosing?
xplot = 'five_facet_mindfulness_survey.act_with_awareness' #can make them select it maybe?
yplot = 'HouseholdIncome'

with left_column:
    #line plot
    fig = px.line(dfplot, x= xplot, y= yplot, title='Some plot')        
    if color_by!='Nothing':
        fig = px.line(dfplot, x= xplot, y= yplot, color=color_by, title='Some plot')
    st.plotly_chart(fig) #display the plot

with right_column:
    #scatter plot
    fig = px.line(dfplot, x= xplot, y= yplot, title='Another plot')        
    if color_by!='Nothing':
        fig = px.scatter(dfplot, x= xplot, y= yplot, color=color_by,title='Another plot')
    st.plotly_chart(fig) #display the plot


#add some more cool plots, make them interactive - maybe with slider