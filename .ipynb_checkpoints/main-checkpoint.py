import streamlit as st
import tensorflow as tf
import numpy as np

#  Tensorflow Model Prediction
def model_prediction(test_image):
   model = tf.keras.models.load_model('trained_model.keras') 
   image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
   input_arr = tf.keras.preprocessing.image.img_to_array(image) 
   input_arr = np.array([input_arr])  #convert single image to a batch
   prediction = model.predict(input_arr)
   result_index = np.argmax(prediction)
   return result_index

#sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])


# Home page
if(app_mode == "Home"):
   st.header("DISEASE DETECTION FOR CROPS")
   image_path = "home_page.jpg"
   st.image(image_path,use_container_width=True)
   st.markdown("""
 Welcome to the  DISEASE DETECTION FOR CROPS Platform! 🌿🔍
    
    Our mission is to help in identifying crop diseases efficiently. Upload an image of a crop, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the Real Machine Learning !

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.

      
  """)


# about page
elif(app_mode == "About"):
   st.header("About")
   st.markdown("""
About Dataset
This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.
Content
1. Train (70295 images)
2. Valid (17572 image)
3. Test (33 images)
               
The future of crop disease detection systems is very promising, with advancements in technology driving significant improvements in accuracy, efficiency, and accessibility. Here are some key areas of future scope:

1. Enhanced AI and Deep Learning:

Hybrid Models:
               
Combining Convolutional Neural Networks (CNNs) with Vision Transformers (ViTs) to leverage the strengths of both, leading to more accurate and robust disease detection.
Improved Accuracy:
Continued refinement of deep learning algorithms to achieve higher accuracy in detecting subtle disease symptoms and differentiating between similar diseases.
Predictive Analytics:
Developing AI models that can predict disease outbreaks based on environmental factors, historical data, and other relevant information.
               
2. Integration of Advanced Technologies:

Edge AI and IoT:
               
Deploying lightweight AI models on IoT devices for real-time disease detection in the field, especially in remote areas with limited internet connectivity.
Unmanned Aerial Vehicles (UAVs/Drones):
               
Utilizing drones equipped with high-resolution cameras and sensors to capture aerial imagery for large-scale crop monitoring and early disease detection.
    
             Multi-modal Data Fusion:
               
Combining data from various sources, such as visible light, infrared, hyperspectral, and thermal imaging, to provide a more comprehensive assessment of plant health.
Sensor Technology:
               
The use of advanced sensors that can detect volatile organic compounds(VOC's) that plants release when under stress from disease. This would allow for even earlier detection of disease.
    
3. Data and Accessibility:

Federated Learning:
               
Enabling collaborative model training on decentralized datasets, ensuring data privacy and security.
Crowdsourced Data Collection:
               
Utilizing mobile apps to collect images and data from farmers, creating large-scale datasets for model training.
Accessibility for Farmers:
               
Developing user-friendly mobile apps and platforms that provide farmers with real-time disease detection and management recommendations.
    
4. Sustainable Agriculture:

    
Precision Agriculture:
               
Integrating disease detection systems into precision agriculture practices to optimize resource utilization and minimize environmental impact.
    
Early Intervention:

Enabling early disease detection and intervention to reduce crop losses and promote sustainable farming practices.
In essence, the future of crop disease detection systems is geared towards creating more intelligent, efficient, and accessible solutions that empower farmers to protect their crops and ensure food security.          


      
   """)

elif(app_mode == "Disease Recognition"):
   st.header("Disease Detection")
   test_image = st.file_uploader("Choose an Image:")
   if(st.button("Show Image")):
      st.image(test_image,use_container_width=True)
   # predict button
   if(st.button("predict")):
      st.write("Our Prediction")
      result_index = model_prediction(test_image)
      #define class
      class_name = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
      st.success("Model is Predicting it's a {}".format(class_name[result_index]))     
      