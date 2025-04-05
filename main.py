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

      st.title("Measures to Take")

      # Treatments   
      if(class_name[result_index] == class_name[0]):

               st.write("""

                        Cause:
                        
The fungus Venturia inaequalis.  
                         

                        Impact:
                        
Damages leaves and fruit, reducing yield and quality.   
Severe infections weaken trees.   
                        

                        Conditions:
                        
Thrives in cool, wet weather.   

                        Treatment Strategies:

Effective apple scab control involves a combination of cultural practices and, when necessary, fungicide applications.   

                        

                        Cultural Control:
                        


                        Sanitation:
                        
Rake and remove fallen leaves in autumn to reduce overwintering spores. Do not compost infected leaves in home compost piles.   
Prune trees to improve air circulation, which helps leaves dry faster.
                        

                        Resistant Cultivars:
                        
Plant apple varieties bred for scab resistance. This is the best long term solution.   
Proper watering:
Water in the morning or early evening so that foliage has time to dry, also try and avoid overhead watering that keeps the foliage wet for extended periods of time.

                        Chemical Control (Fungicides):
                        

Fungicides are most effective when applied preventively, before infection occurs.   

                        Timing is crucial:
                        
Applications typically begin in spring as buds break and continue at intervals determined by weather conditions and the specific fungicide.   
"Protectant" fungicides, and "post infection" fungicides are the two main types.

                        Types of fungicides:

Always follow product label directions.
                        It is important to rotate the types of fungicides that are used to help prevent the fungus from building up a resistance.
                        
 """)
         
      elif(class_name[result_index] == class_name[1]):
        
         st.write("""
                   Cause:
The fungus Botryosphaeria obtusa.

                  Symptoms:

                  Leaves: 

 "Frog-eye" leaf spots—circular spots with purplish or reddish edges and tan interiors.
Fruit: Rotting areas, often with concentric rings; fruit may mummify.
Branches: Cankers—sunken, reddish-brown areas with rough, cracked bark.

                  Spread:

The fungus overwinters in cankers and mummified fruit.
Spores spread through wind and rain.
Infections occur through wounds or natural openings.

                  Treatment:

Effective management involves a combination of cultural practices and, in some cases, fungicide applications.


                  Cultural Control:

                  Sanitation:
Remove and destroy mummified fruit, fallen leaves, and infected branches.
Prune out cankers during dry weather.

                  Pruning:
Improve air circulation to reduce moisture.
Prune during dormant periods.

                  Tree Health:
Maintain tree vigor through proper watering and fertilization.
Reduce tree stress.

                  Chemical Control (Fungicides):
Fungicides are most effective when applied preventatively.
Consider fungicide use after cultural controls have been implemented, and the disease is still a problem.
Fungicides such as captan, and copper based fungicides can be used. Always follow the product label.
It is important to note that fungicides applied for apple scab control, can also help to control black rot.

""")

      elif(class_name[result_index] == class_name[2]):
         
         st.write(""" 
                  Cause:
The fungus Gymnosporangium juniperi-virginianae.

                  Symptoms:

                  On Apple Trees:
Bright yellow-orange spots on leaves, which may develop reddish rings.
Fruit can also be affected, with spots and deformities.

                  On Cedar Trees:
Brown, globular galls form on branches.
In spring, these galls produce orange, gelatinous "horns" that release spores.

                  Spread:
Spores released from cedar galls are carried by wind to apple trees.
Spores from infected apple leaves then infect cedar trees.

                  Treatment and Control:


                  Managing cedar-apple rust involves a combination of strategies:


                  Cultural Control:

                  Host Separation:
If feasible, separating apple and cedar trees can reduce infection. However, wind can carry spores over considerable distances.

                  Pruning:
Remove galls from cedar trees before they release spores in spring.
Prune infected branches from apple trees.

                  Resistant Varieties:
Plant apple and crabapple varieties that are resistant to cedar-apple rust. This is a very effective long term solution.

                  Chemical Control (Fungicides):
Fungicides are most effective when applied preventively to apple trees.

                  Timing is crucial:
Applications should begin in spring, when apple buds are developing.
Follow fungicide label instructions carefully. """)
      
      elif(class_name[result_index] == class_name[3]):
       
         st.write(""" No signs of fungal infection, or bacterial infection.""")
 
      elif(class_name[result_index] == class_name[4]):
         
         st.write(""" No signs of fungal infection, or bacterial infection. """)
       
      elif(class_name[result_index] == class_name[5]):
       
         st.write("""
                   Cause:
Primarily caused by the fungus Podosphaera clandestina.

                  Symptoms:
White, powdery growth on leaves, shoots, and sometimes fruit.
Leaves may become distorted, curl, or stunt.
In severe cases, it can reduce fruit quality and yield.
Young growth is most susceptible.

                  Conditions:
Favored by warm, dry days and humid nights.
Unlike many other fungal diseases, it doesn't require free water for infection.

                  Treatment and Control:

Effective management of powdery mildew involves a combination of cultural practices and, when necessary, fungicide applications.


                  Cultural Control:

                  Pruning:
Improve air circulation by pruning to open up the canopy. This helps reduce humidity.
Remove infected shoots and leaves.

                  Sanitation:
Remove fallen leaves and debris that can harbor the fungus.

                  Spacing:
Proper spacing of trees allows for better air circulation.

                  Resistant Varieties:
When possible plant cherry variaties that display some resistance to powdery mildew.

                  Chemical Control (Fungicides):
Fungicides are most effective when applied preventatively or at the first sign of infection.
Common fungicides used to control powdery mildew include:
Sulfur-based fungicides.
Potassium bicarbonate.
Other fungicides labelled for powdery mildew control on cherry trees.

 """)

      elif(class_name[result_index] == class_name[6]):
      
         st.write(""" No signs of fungal infection, or bacterial infection.""")

      elif(class_name[result_index] == class_name[7]):
        
         st.write(""" 
                  Cause:
The fungus Cercospora zeae-maydis.

                  Conditions:
Favored by warm, humid weather and prolonged periods of leaf wetness.
Reduced tillage and continuous corn cropping increase risk.

                  Symptoms:
Initially, small, light-colored lesions appear on lower leaves.
Lesions elongate, becoming rectangular and gray to tan, with parallel sides defined by leaf veins.
In severe cases, lesions coalesce, leading to leaf death.
The leaf will take on a grey appearence due to the production of spores.

                  Management and Treatment:

Effective management involves a combination of cultural practices and, when necessary, fungicide applications.


                  Cultural Control:

                  Crop Rotation:
Rotating corn with non-host crops reduces fungal survival in crop residue.

                  Tillage:
Tillage helps bury infected corn residue, reducing inoculum.

                  Resistant Hybrids:
Planting corn hybrids with resistance to gray leaf spot is a key control measure.

                  Proper spacing:
This helps to increase airflow, and reduce the amount of time that foliage remains wet.

                  Chemical Control (Fungicides):
Fungicides are most effective when applied preventively or at early symptom development.
Foliar fungicides are available for gray leaf spot control.
Timing of application is critical, often coinciding with critical growth stages.
It is very important to always follow lable instructions when using fungicides. """)

      elif(class_name[result_index] == class_name[8]):
         
         st.write(""" 
                  Cause:
The fungus Puccinia sorghi.

                  Symptoms:
Small, round to elongated, reddish-brown pustules (uredinia) appear on both upper and lower leaf surfaces.
These pustules contain powdery, rust-colored spores.
In severe infections, leaves may turn yellow and die prematurely.
As the plant matures, the pustules can turn a dark brown to black color.

                  Conditions:
Favored by cool to moderate temperatures and high humidity.
Spore dispersal is primarily by wind.

                  Management and Treatment:

Effective management of common rust involves a combination of strategies:


                  Cultural Control:

                  Resistant Hybrids:
Planting corn hybrids with genetic resistance is the most effective control measure.

                  Crop Monitoring:
Regularly scout fields for early signs of infection.

                  Removal of alternate hosts:
Common rust needs an alternate host, such as certain oxalis species. Removing these weeds near corn fields can help reduce initial infection.

                  Chemical Control (Fungicides):
Foliar fungicides can be used to control common rust, especially in susceptible hybrids or when disease pressure is high.
Timing of application is crucial, often at early symptom development.
Always follow fungicide label instructions carefully.
It is important to note that the use of fungicides is not always economically necessary, because many modern corn hybrids have good resistance. """)

      elif(class_name[result_index] == class_name[9]):
         
         st.write(""" 
                  Cause:
Caused by the fungus Exserohilum turcicum.

                  ymptoms:
Characterized by long, elliptical or "cigar-shaped" lesions on corn leaves.
Lesions are typically grayish-green or tan.
In severe cases, lesions can coalesce, leading to extensive leaf damage.
The disease starts on the lower leaves, and works its way up the plant.

                  Conditions:
Favored by warm, humid weather and prolonged periods of leaf wetness.
The fungus overwinters in corn residue.

                  Management and Treatment:

Effective NCLB management involves a combination of strategies:


                  Cultural Control:

                  Resistant Hybrids:
Planting corn hybrids with genetic resistance is the most effective control measure.

                  Crop Rotation:
Rotating corn with non-host crops helps reduce fungal survival in crop residue.

                  Tillage:
Tillage can help bury infected corn residue, reducing the amount of inoculum available for infection.

                  Residue management:
Promoting the breakdown of corn residue.

                  Chemical Control (Fungicides):
Foliar fungicides can be effective in controlling NCLB, especially in susceptible hybrids or when disease pressure is high.
Timing of fungicide applications is crucial, often around tasseling or early silking.
It is very important to always follow the fungicide product label instructions.
 """)

      elif(class_name[result_index] == class_name[10]):
     
         st.write(""" No signs of fungal infection, or bacterial infection. """)

      elif(class_name[result_index] == class_name[11]):
         
         st.write(""" 
                  Cause:
The fungus Guignardia bidwellii.

                  Symptoms:

                  Leaves:
Small, reddish-brown circular spots with darker margins. As the disease progresses, these spots may merge.

                  Fruit: 
Infected berries initially show light brown spots, which rapidly expand. The entire berry eventually turns black, shrivels, and becomes a hard "mummy."
Shoots and Tendrils: Dark, elongated lesions.

                  Conditions:
Warm, humid weather favors the development and spread of black rot.
Periods of prolonged wetness are particularly conducive to infection.

                  Disease Cycle:
The fungus overwinters in infected mummified fruit, canes, and tendrils.
In spring, spores are released and spread by wind and rain.
Infections occur when plant tissues remain wet for extended periods.

                  Treatment and Control:

Effective management of grape black rot requires a combined approach:


                  Cultural Control:

                  Sanitation:
Remove and destroy mummified fruit, fallen leaves, and infected canes during dormant pruning. This is very important.
Maintain a clean vineyard environment to reduce overwintering inoculum.

                  Pruning and Training:
Promote good air circulation by proper pruning and training of vines.
This helps leaves and fruit dry quickly, reducing the risk of infection.

                  Site Selection:
Plant grapevines in sunny locations with good air drainage.

                  Weed Control:
Control weeds, as this increases air flow.

                  Chemical Control (Fungicides):
Fungicides are most effective when applied preventively.
Timing is critical, especially during early growth stages and bloom.

                  Commonly used fungicides include those containing:
Myclobutanil
Captan
Various other products, always check local extension office recommendations.
Always follow fungicide label instructions carefully.

                  Resistant Cultivars:
Planting grape varieties that are less susceptible to black rot can be a helpful long term strategy.""")

      elif(class_name[result_index] == class_name[12]):
       
         st.write(""" 
                  Complex of Fungi:
Esca isn't caused by a single pathogen, but rather a complex of wood-decaying fungi, including Phaeoacremonium aleophilum and Phaeomoniella chlamydospora.

                  Impact:
It leads to the progressive decline of grapevines, reducing yield and ultimately causing vine death.
It is a "trunk disease" meaning it attacks the woody parts of the vine.

                  Symptoms:

                  Foliar Symptoms:
"Tiger-stripe" patterns on leaves, with yellow or reddish discoloration between veins, which later become necrotic.
Premature leaf drop.

                  Fruit Symptoms:
Small, dark spots on berries, sometimes with a purple ring.
Berry cracking and drying.

                  Wood Symptoms:
Dark, discolored wood in cross-sections of trunks, canes, and spurs.
"White rot" and "brown wood streaking" within the vine's wood.
apoplexy, which is a sudden collapse of the vine.

                  Treatment and Management:

Esca is notoriously difficult to control, and there's no single cure. Management focuses on prevention and minimizing spread:


                  Preventive Measures:

                  Pruning Practices:
Delay pruning to reduce wound susceptibility.
Avoid pruning during wet periods.
Sterilize pruning tools to prevent disease spread.
Protect pruning wounds with wound protectants.

                  Vineyard Hygiene:
Remove and destroy infected wood.
Maintain healthy vineyard conditions to reduce stress on vines.

                  Planting Material:
Use healthy, disease-free planting material.

                  Resistant Varieties:
Research is ongoing into grape varieties with increased resistance.

                  Limited Chemical Control:
Fungicides are generally ineffective against established Esca infections.
Some wound protectants can help prevent new infections.

                  Biological Control:
Some applications of Trichoderma have shown promise in reducing infections.

                  Important Considerations:
Esca develops slowly, making early detection difficult.
Integrated pest management (IPM) strategies are crucial.""")

      elif(class_name[result_index] == class_name[13]):
         
         st.write(""" """)

      elif(class_name[result_index] == class_name[14]):
         
         st.write(""" """)
      
      elif(class_name[result_index] == class_name[15]):
         
         st.write(""" """)

      elif(class_name[result_index] == class_name[16]):
         
         st.write(""" """)

      elif(class_name[result_index] == class_name[17]):
         
         st.write(""" """)

      elif(class_name[result_index] == class_name[18]):
         
         st.write(""" """)

      elif(class_name[result_index] == class_name[19]):
         
         st.write(""" """)

      elif(class_name[result_index] == class_name[20]):
         
         st.write(""" """)

      elif(class_name[result_index] == class_name[21]):
         
         st.write(""" """)

      elif(class_name[result_index] == class_name[22]):
         
         st.write(""" """)

      elif(class_name[result_index] == class_name[23]):
         
         st.write(""" """)

      elif(class_name[result_index] == class_name[24]):
         
         st.write(""" """)

      elif(class_name[result_index] == class_name[25]):
         
         st.write(""" """)

      elif(class_name[result_index] == class_name[26]):
         
         st.write(""" """)

      elif(class_name[result_index] == class_name[27]):
         
         st.write(""" """)

      elif(class_name[result_index] == class_name[28]):
         
         st.write(""" """)

      elif(class_name[result_index] == class_name[29]):
         
         st.write(""" """) 

      elif(class_name[result_index] == class_name[30]):
         
         st.write(""" """)                                            

       
                         