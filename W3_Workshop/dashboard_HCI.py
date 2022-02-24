from asyncio.windows_events import NULL
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
races = np.insert(races, 0, 'All') #add choose all option to front of list
sexes = df['Sex'].unique()
sexes = np.insert(sexes, 0, 'All') #add choose all option to front of list

#SIDEBAR STUFF:
st.sidebar.write('Choose which data subsets to explore:')
sex = st.sidebar.selectbox('Sex', sexes)
race = st.sidebar.selectbox('Race', races)

st.sidebar.write('Choose how to color the plots:')
color_by = st.sidebar.selectbox('Color by', options=['Sex', 'Race', 'HighestEducation', 'Nothing'])

st.markdown("<h1 style='text-align: center; color: green;'>How do you become the most mindful person?</h1>", unsafe_allow_html=True)

with st.expander("Show full data set"):
    st.write(df)


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
    #line plot
    fig = px.scatter(dfplot, x= xplot, y= yplot, title='How does your income affect your mindfulness?', trendline="ols")        
    if color_by!='Nothing':
        fig = px.scatter(dfplot, x= xplot, y= yplot, color=color_by, title='How does your income affect your mindfulness?', trendline="ols")
    st.plotly_chart(fig, use_container_width=True) #display the plot

yplot = 'five_facet_mindfulness_survey.act_with_awareness'
xplot = 'CoffeeCupsPerDay'

with right_column:
    #scatter plot
    fig = px.box(dfplot, x= xplot, y= yplot, title='How does your coffee intake affect you mindfulness?')        
    if color_by!='Nothing':
        fig = px.box(dfplot, x= xplot, y= yplot, color=color_by,title='How does your coffee intake affect you mindfulness?')
    st.plotly_chart(fig, use_container_width=True) #display the plot

st.write('')
st.markdown("<h5 style='text-align: center; color: blue;'>Use the sidebar to choose what to color by</h1>", unsafe_allow_html=True)


#add some more cool plots, make them interactive - maybe with slider