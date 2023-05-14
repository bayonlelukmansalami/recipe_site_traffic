#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
import scipy.optimize as opt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix
%matplotlib inline

df = pd.read_csv('https://github.com/bayonlelukmansalami/recipe_site_traffic/raw/main/recipe_site_traffic_2212.csv')

df["high_traffic"] = df["high_traffic"].fillna(value='Low')

df["calories"] = df["calories"].fillna(df["calories"].median())
df["carbohydrate"] = df["carbohydrate"].fillna(df["carbohydrate"].median())
df["sugar"] = df["sugar"].fillna(df["sugar"].median())
df["protein"] = df["protein"].fillna(df["protein"].median())

df['servings'] = df['servings'].str.replace(' as a snack','')
df['servings'] = df['servings'].astype(int)

df['category'] = df['category'].str.replace('Chicken Breast','Chicken')

df = pd.get_dummies(df, columns= ["category"], drop_first=False)

#df["high_traffic"] = df["high_traffic"].map({"High": 1, "Low": 0})

X = df[df.columns.drop(["recipe","high_traffic"])]
y = df[['high_traffic']]


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, stratify=y, random_state=4)

model = LogisticRegression(random_state=4)
model.fit(X_train, y_train)

social_acc = ['About', 'Kaggle', 'Medium', 'LinkedIn']
social_acc_nav = st.sidebar.selectbox('About', social_acc)
if social_acc_nav == 'About':
    st.sidebar.markdown("<h2 style='text-align: center;'> Salami Lukman Bayonle</h2> ", unsafe_allow_html=True)
    st.sidebar.markdown('''---''')
    st.sidebar.markdown('''
    • Data Analytics/Scientist (Python/R/SQL/Tableau) \n 
    • Maintenance Specialist (Nigerian National Petroleum Company Limited) \n 
    • IBM/GOOGLE/DATACAMP Certified Data Analyst and Data Scientist''')
    st.sidebar.markdown("[ Visit Github](https://github.com/bayonlelukmansalami)")

elif social_acc_nav == 'Kaggle':
    st.sidebar.image('kaggle.jpg')
    st.sidebar.markdown("[Kaggle](https://www.kaggle.com/bayonlesalami)")

elif social_acc_nav == 'Medium':
    st.sidebar.image('medium.jpg')
    st.sidebar.markdown("[Click to read my blogs](https://medium.com/@bayonlelukmansalami/)")

elif social_acc_nav == 'LinkedIn':
    st.sidebar.image('linkedin.jpg')
    st.sidebar.markdown("[Visit LinkedIn account](https://www.linkedin.com/in/salamibayonlelukman/)")
    



st.title('Recipe Traffic Prediction Web App')
st.write('Predict which recipes will lead to high traffic')
st.write("It helps banks and credit card companies immediately to issue loans to customers with good creditworthiness")   
st.write("There are two recipe traffic that Tasty Bytes used :1. High, 0. Low")
st.write("A recipe with a high score will lead to more revenue.")


calories = st.number_input('calories')

carbohydrate = st.number_input('carbohydrate')

sugar = st.number_input('sugar')

protein = st.number_input('protein')

servings = st.number_input('servings', min_value=1, max_value=6, step=1)

category_Beverages = st.number_input('category_Beverages', min_value=0, max_value=1, step=1)

category_Breakfast = st.number_input('category_Breakfast', min_value=0, max_value=1, step=1)

category_Chicken = st.number_input('category_Chicken', min_value=0, max_value=1, step=1)

category_Dessert = st.number_input('category_Dessert', min_value=0, max_value=1, step=1)

category_Lunch_Snacks = st.number_input('category_Lunch/Snacks', min_value=0, max_value=1, step=1)

category_Meat = st.number_input('category_Meat', min_value=0, max_value=1, step=1)

category_One_Dish_Meal = st.number_input('category_One Dish Meal', min_value=0, max_value=1, step=1)

category_Pork = st.number_input('category_Pork', min_value=0, max_value=1, step=1)

category_Potato = st.number_input('category_Potato', min_value=0, max_value=1, step=1)

category_Vegetable = st.number_input('category_Vegetable', min_value=0, max_value=1, step=1)




features = [calories, carbohydrate, sugar, protein, servings,
           category_Beverages, category_Breakfast, category_Chicken, category_Dessert, category_Lunch_Snacks,
           category_Meat, category_One_Dish_Meal, category_Pork, category_Potato, category_Vegetable]
            
features_np  = np.array([features])

st.table(features_np)


if st.button('Predict'):
    prediction = model.predict(features_np)
    st.write('Predicted Recipe Traffic = ', model.predict(features_np))


# In[ ]:




