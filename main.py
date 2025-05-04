# Imported All Essential Libraries
import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Loaded Embedding of Images.
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
# Loaded Filesnames
filenames = pickle.load(open('filenames.pkl','rb'))

# Using A Pre - Train Model < ReNet50 > to generate the embedding of images.
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

# Setting the model
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Setting up the Streamlit web app.
st.markdown("<h1 style='text-align: center; color: white;'>Fashion & Accessories Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Upload an image and get similar outfit recommendations!</p>", unsafe_allow_html=True)

# Opening the Uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# Extracting the feature
def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

# Using KNN to generates 5 Recommendations of input image.
def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices


# steps
# file upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.markdown("---")
        st.subheader("📸 Your Uploaded Image")
        st.image(display_image)
        # feature extract
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        #st.text(features)
        # recommendention
        indices = recommend(features,feature_list)
        # show
        st.markdown("---")
        st.subheader("🎯 Based on your upload, here are 5 similar fashion recommendations:")
        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occured in file upload")


#
#
# import streamlit as st
# import os
# from PIL import Image
# import numpy as np
# import pickle
# import tensorflow
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.layers import GlobalMaxPooling2D
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from sklearn.neighbors import NearestNeighbors
# from numpy.linalg import norm
#
# # Load data
# feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
# filenames = pickle.load(open('filenames.pkl', 'rb'))
#
# # Load model
# model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# model.trainable = False
# model = tensorflow.keras.Sequential([
#     model,
#     GlobalMaxPooling2D()
# ])
#
# # Streamlit UI
# st.markdown("<h1 style='text-align: center; color: white;'>🧥 Fashion Recommender</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center; color: grey;'>Upload an image and get similar outfit recommendations!</p>", unsafe_allow_html=True)
#
# # Function to save uploaded file
# def save_uploaded_file(uploaded_file):
#     try:
#         with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
#             f.write(uploaded_file.getbuffer())
#         return 1
#     except:
#         return 0
#
# # Feature extraction
# def feature_extraction(img_path, model):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     expanded_img_array = np.expand_dims(img_array, axis=0)
#     preprocessed_img = preprocess_input(expanded_img_array)
#     result = model.predict(preprocessed_img).flatten()
#     normalized_result = result / norm(result)
#     return normalized_result
#
# # Recommendation
# def recommend(features, feature_list):
#     neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
#     neighbors.fit(feature_list)
#     distances, indices = neighbors.kneighbors([features])
#     return indices
#
# # File upload
# uploaded_file = st.file_uploader("Upload an image of your clothing style", type=['jpg', 'jpeg', 'png'])
#
# if uploaded_file is not None:
#     if save_uploaded_file(uploaded_file):
#         # Display uploaded image
#         st.markdown("---")
#         st.subheader("📸 Your Uploaded Image")
#         display_image = Image.open(uploaded_file)
#         st.image(display_image, width=300)
#
#         # Extract features and recommend
#         features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
#         indices = recommend(features, feature_list)
#
#         # Display recommendations
#         st.markdown("---")
#         st.subheader("🎯 Based on your upload, here are 5 similar fashion recommendations:")
#         cols = st.columns(5)
#
#         # Set a fixed width for the images
#         image_width = 200  # Set the width to your desired value, for example, 200px
#
#         for i, col in enumerate(cols):
#             with col:
#                 st.image(filenames[indices[0][i]], width=image_width)
