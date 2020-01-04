# Hernia Annotation

This work demonstrates the 3D Slicer / PLUS Toolkit as an open source platform for automatic collection of labelled training data for surgical video. 
It uses this platform to collect labelled training data consiting of optical tracking data, and video data. Specifically, we demonstrate this on
in classifying surgical video frames for inguinal hernia repair by classifying what tool is interacting with which tissue on an inguinal herna phantom. 
We then use the extension manager in 3D Slicer to create a custom Slicer module to deploy our model. 

### File Structure
##### CalibratedTrackedTools folder
Contains the Slicer scene and individual components for collecting data, including the tool transformations, and the tool models.
##### HerniaCode
Contains *HerniaExtension** and **Notebooks**.
##### HerniaExtension
Contains the custom 3D Slicer module used to deploy real-time classification.
##### Notebooks
Contains the Jupyter Notebooks used to export the data from 3D Slicer, and to train our deep learning model. 

### Results
- This platform allowed automatic labelling of our dataset during collection with 98% accuracy. 
- We trained a CNN to classify new frames with 98% testing accuracy on real-time data at 30fps. 
