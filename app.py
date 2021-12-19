import streamlit as st
import numpy as np
from sklearn.datasets import load_iris # import dataset
from sklearn.neighbors import KNeighborsClassifier #import algorithm

st.title("IRIS_FLOWER CLASSIFICATION")
var = load_iris() # load the dataset into a variable

#Split the data into input and output
x = var.data # extract all the input columns
y = var.target # extract all the output column

#Use the algorithm (Kneighbors)

model = KNeighborsClassifier(n_neighbors=13)
model.fit(x,y)

# Find min,max values using numpy
xmin = np.min(x,axis=0)
xmax = np.max(x,axis=0)

#create a slider using these min,max values
sepal_length = st.slider('sepal_length',float(xmin[0]),float(xmax[0]))
sepal_width  = st.slider('sepal_width',float(xmin[1]),float(xmax[1]))
petal_length = st.slider('petal_length',float(xmin[2]),float(xmax[2]))
petal_width  = st.slider('petal_width',float(xmin[3]),float(xmax[3]))

y_pred = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])

#print the output
out = ['Iris_setosa','Iris_versicolor','Iris_virginica']
st.title(out[y_pred[0]]) # Prints which type of Iris flower according to inputs
