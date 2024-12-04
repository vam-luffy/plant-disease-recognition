import streamlit as st
import tensorflow as tf
import numpy as np

# Tensorflow model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr= np.array([input_arr]) # batches
    prediction = model.predict(input_arr)
    result_index=np.argmax(prediction)
    return result_index

#sidebar
st.sidebar.title("dashboard")
app_mode=st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

if app_mode == "Home":
    st.header("🌱 Plant Disease Recognition System")
    image_path = "img.webp"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    ### Welcome to the Plant Disease Recognition System 🌟
    This system leverages the power of **Artificial Intelligence** and **Deep Learning** to identify diseases in plants by analyzing images of their leaves. 

    #### Features:
    - 📸 Upload a leaf image to detect diseases.
    - 🌍 Supports multiple plant species.
    - 🧪 Provides disease details and possible treatments.

    #### Why use this system?
    - Improve crop yield and reduce losses.
    - Get insights for better agricultural practices.
    - Stay eco-friendly by targeting specific treatments.

    ---
    🤖 **Get started** by navigating through the menu and uploading an image. Let's make farming smarter and more sustainable!
    """)
# About page
elif app_mode == "About":
    st.header("📊 Dataset for Plant Disease Recognition")
    st.markdown("""
    ### 🌾 About the Dataset
    The dataset used for this system is a comprehensive collection of high-quality images of healthy and diseased plant leaves. It has been meticulously curated to enable accurate detection of diseases across a wide variety of plant species, contributing to better agricultural practices and informed decision-making.

    #### 📂 Dataset Features:
    - **Diversity**: Images covering multiple plant species, including common crops and exotic plants, along with various diseases.
    - **High Quality**: Crystal-clear and detailed images to ensure the AI model can distinguish even subtle differences.
    - **Structured Data**: Well-organized and labeled images categorized by plant species and disease type.
    - **Scalability**: A dataset large enough to train robust models while maintaining balance across categories.
    - **Applications**: Ideal for training machine learning, deep learning, and computer vision systems for agriculture.

    #### 🌟 Benefits of Using This Dataset:
    - Reduces the need for manual disease identification.
    - Enables precision farming through early disease detection.
    - Supports the development of eco-friendly solutions by targeting diseases effectively.

    ---
    🛠️ **Data Preparation**:
    To ensure optimal model performance, the dataset has been:
    - **Preprocessed**: Images are normalized and resized to maintain consistency.
    - **Split**: Divided into training, validation, and testing sets for reliable evaluation.
    - **Augmented**: Enhanced with transformations like rotations, flips, and brightness adjustments to simulate real-world scenarios.

    #### 🚀 Future Scope:
    - Expanding the dataset to include more rare diseases and plant species.
    - Collaborating with agricultural researchers for field-level validation.
    - Leveraging the data for other AI applications like pest detection and soil quality analysis.

    This dataset is the backbone of the Plant Disease Recognition System, enabling reliable, efficient, and scalable solutions for modern agriculture.
    """)

# prediction page
elif app_mode == "Disease Recognition":
    st.header("🌿 Disease Recognition")
    
    # File uploader for test image
    test_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True, caption="Uploaded Image")
        
        # Predict button functionality
        if st.button("Predict"):
            with st.spinner("Please wait..."):
                try:
                    # Call the model prediction function
                    result_index = model_prediction(test_image)
                    
                    # Define the class labels
                    class_name = [
                        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
                        'Apple___healthy', 'Blueberry___healthy', 
                        'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
                        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 
                        'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 
                        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
                        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
                        'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
                    ]
                    
                    # Display the predicted class
                    predicted_class = class_name[result_index]
                    st.success(f"✅ Model Prediction: {predicted_class}")
                
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload an image to proceed.")

