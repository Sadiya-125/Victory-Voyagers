import os
from flask import Flask, jsonify, render_template, request, redirect, url_for, session
import requests
import cohere, markdown, joblib
import numpy as np
import pandas as pd
from utils.fertilizer import fertilizer_dic
from utils.model import ResNet9
import io
import torch
from torchvision import transforms
from PIL import Image
from markupsafe import Markup
from utils.disease import disease_dic

app = Flask(__name__)

app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fallback_secret_key')
co = cohere.Client('9xtb0BFwgzrTgGQIG7QJrdZAUBH8oQoW30EK2Z7I')

crop_recommendation_model = joblib.load('./models/random_forest_model.pkl')
label_encoder = joblib.load('./models/label_encoder.pkl')
disease_classes = ['Apple___Apple_scab',
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/crop_recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    return render_template('crop.html')

@app.route('/fertilizer_recommendation', methods=['GET','POST'])
def fertilizer_recommendation():
    return render_template('fertilizer.html')

@app.route('/disease_prediction', methods=['GET','POST'])
def disease_prediction():
    return render_template('disease.html')

@app.route('/save_location', methods=['POST'])
def save_location():
    data = request.get_json()
    
    latitude = data.get('latitude')
    longitude = data.get('longitude')

    session['latitude'] = latitude
    session['longitude'] = longitude
    return jsonify({'success': True})

def calculate_average_soil_property(soil_data, property_name):
    if property_name in soil_data:
        values = []

        for depth_label, value in soil_data[property_name]:
            if value is not None:  # Only include valid values
                values.append(value)
        
        if values:
            average_value = sum(values) / len(values)
            return average_value
        else:
            return None
    else:
        return None

@app.route('/get_details', methods=['GET', 'POST'])
def get_details():
    latitude = session['latitude']
    longitude = session['longitude']

    # latitude = -9
    # longitude = -72

    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean,wind_speed_10m_max&timezone=auto"

    response = requests.get(weather_url)
    weather_data = response.json()

    if "error" in weather_data:
        print("API Error: ", weather_data["reason"])
    elif "daily" in weather_data:
        daily_data = weather_data["daily"]
        
        temp_max = daily_data["temperature_2m_max"][0]
        temp_min = daily_data["temperature_2m_min"][0]
        rainfall = daily_data["precipitation_sum"][0]
        humidity = daily_data["relative_humidity_2m_mean"][0]
        wind_speed = daily_data["wind_speed_10m_max"][0]

        session['weather'] = {
            'temperature_max': temp_max,
            'temperature_min': temp_min,
            'rainfall': rainfall,
            'humidity': humidity,
            'wind_speed': wind_speed
        }
    else:
        print("Error: Check API Response")

    soil_url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={longitude}&lat={latitude}&property=bdod&property=cec&property=cfvo&property=clay&property=nitrogen&property=ocd&property=ocs&property=phh2o&property=sand&property=silt&property=soc&property=wv0010&property=wv0033&property=wv1500&depth=0-5cm&depth=0-30cm&depth=5-15cm&depth=15-30cm&depth=30-60cm&depth=60-100cm&depth=100-200cm&value=Q0.05&value=Q0.5&value=Q0.95&value=mean&value=uncertainty"

    response = requests.get(soil_url)
    soil_data = response.json()
    
    if "properties" in soil_data:
        selected_soil_data = {}
        selected_layers = ["bdod", "cec", "cfvo", "clay", "nitrogen", "phh2o", "sand", "silt", "soc"]
        layers = soil_data["properties"]["layers"]
        
        for layer in layers:
            name = layer["name"]  # Property name (e.g., Clay, Sand, pH)
            if name in selected_layers:
                selected_soil_data[name] = []
                for depth_info in layer["depths"]:
                    depth_range = depth_info["range"]
                    depth_label = f"{depth_range['top_depth']}-{depth_range['bottom_depth']} cm"
                    value = depth_info["values"]["mean"]
                    selected_soil_data[name].append((depth_label, value))
    else:
        print("Error: No Soil Data Found!")
    session['soil'] = selected_soil_data

    nitrogen_average = calculate_average_soil_property(session['soil'], 'nitrogen')
    ph_average = calculate_average_soil_property(session['soil'], 'phh2o')
    session['soil']['nitrogen'] = nitrogen_average
    session['soil']['ph'] = ph_average

    return jsonify({"status": "success", "weather": session['weather'], "soil": session['soil']})

@app.route('/crop_predict', methods=['POST'])
def crop_predict():
    if request.method == 'POST':
        data = request.get_json()
        nitrogen = float(data['N'])
        phosphorus = float(data['P'])
        potassium = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        input_features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        prediction = crop_recommendation_model.predict(input_features)
        recommended_crops = label_encoder.inverse_transform(prediction)
        recommended_crops = recommended_crops.tolist()

        # Cohere API call
        prompt = f"""
        You are an expert agricultural assistant trained to analyze soil and environmental data to provide detailed crop recommendation reports.
        Based on the given input parameters and the recommended crop provided, generate a comprehensive crop report.

        The report should explain *why* the recommended crop is suitable under the given conditions.  
        Use the parameters provided to justify the recommendation using scientific reasoning, such as nutrient uptake, climate tolerance, and water needs.

        ### Input Parameters:

        - **Nitrogen (N):** {nitrogen} mg/kg  
        - **Phosphorus (P):** {phosphorus} mg/kg  
        - **Potassium (K):** {potassium} mg/kg  
        - **Temperature:** {temperature} °C  
        - **Humidity:** {humidity} %  
        - **pH Level:** {ph}
        - **Rainfall:** {rainfall} mm  

        ### Recommended Crop:  
        {recommended_crops}

        ### Output Format:

        1. **Crop Name:** {{recommended_crops}}

        2. **Why this Crop?**  
        Below is a detailed justification for recommending **{{recommended_crops}}** based on the input parameters:

        #### 🧪 Soil Nutrients

        - **Nitrogen (N):**  
        Explain how the nitrogen level affects the crop’s vegetative growth, leaf development, or yield quality, and whether the current level is optimal.

        - **Phosphorus (P):**  
        Discuss the role of phosphorus in root development, flowering, or fruiting, and how the current value aligns with the crop’s requirement.

        - **Potassium (K):**  
        Describe how potassium supports disease resistance, water regulation, and fruit quality, and how the measured value suits the crop.

        #### 🌡️ Climate Conditions

        - **Temperature:**  
        Analyze whether the current temperature supports the crop’s optimal growth cycle, germination, and productivity.

        - **Humidity:**  
        Evaluate how the humidity level affects transpiration, disease susceptibility, and crop growth, specific to the recommended crop.

        #### 🌍 Soil pH

        - **pH Level:**  
        Justify if the current soil pH allows for proper nutrient uptake and whether it's in the preferred range for this crop.

        #### 🌧️ Water Availability

        - **Rainfall:**  
        Discuss whether the rainfall level meets the crop's water requirements during its growing season and how it affects root and plant health.
        
        ### Notes:
        - Use only the provided template for the output. Don't use any other heading or subheading.
        - Ensure the explanation is customized for the recommended crop based on its agronomic requirements.
        - Avoid generic responses; use the environmental variables precisely to support your reasoning.
        - Keep the tone educational, informative, and suitable for a farmer or agricultural advisor.

        Restructure it with Bullet Points
        """

        response = co.generate(
            model="command-r-plus-08-2024",
            prompt=prompt,
            max_tokens=1000
        )

        crop_report = response.generations[0].text.strip()
        crop_report = markdown.markdown(crop_report)
        return jsonify({
            "recommended_crops": recommended_crops,
            "crop_report": crop_report
        })

@app.route('/fertilizer_predict', methods=['POST'])
def fertilizer_predict():
    if request.method == 'POST':
        data = request.get_json()
        N = float(data['nitrogen'])
        P = float(data['phosphorous'])
        K = float(data['potassium'])
        crop_name = data['cropname']
    
        df = pd.read_csv('data/fertilizer.csv')

        nr = df[df['Crop'] == crop_name]['N'].iloc[0]
        pr = df[df['Crop'] == crop_name]['P'].iloc[0]
        kr = df[df['Crop'] == crop_name]['K'].iloc[0]

        n = nr - N
        p = pr - P
        k = kr - K
        temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
        max_value = temp[max(temp.keys())]
        if max_value == "N":
            if n < 0:
                key = 'NHigh'
            else:
                key = "NLow"
        elif max_value == "P":
            if p < 0:
                key = 'PHigh'
            else:
                key = "PLow"
        else:
            if k < 0:
                key = 'KHigh'
            else:
                key = "KLow"

        # Cohere API call
        prompt = f"""
        You are an expert agricultural advisor trained to recommend fertilizer strategies based on soil nutrient imbalances and the specific crop being cultivated.  
        Given the current values of Nitrogen (N), Phosphorus (P), and Potassium (K) in the soil, along with the crop name, analyze the most critical nutrient imbalance and provide customized fertilizer recommendations to improve soil health and crop productivity.

        ### Input Parameters:
        - **Crop Name:** {crop_name}
        - **Nitrogen (N):** {N} mg/kg
        - **Phosphorus (P):** {P} mg/kg
        - **Potassium (K):** {K} mg/kg
        - **Deficiency/Excess Key:** {key}

        ### Output Format:

        1. **🌱 Crop Name:** {crop_name}

        2. **🔬 Soil Nutrient Imbalance:**  

        - Based on analysis, the key nutrient imbalance is: **{key}**  

        - This indicates that the soil has **{"high" if "High" in key else "low"}** levels of **{"Nitrogen" if "N" in key else "Phosphorus" if "P" in key else "Potassium"}**.

        3. **🧪 Fertilizer Recommendation and Soil Amendment Strategies:**
        Provide the fertilizer recommendation and soil amendment strategies to address the identified imbalance.
        Make sure to include the following recommendations: {fertilizer_dic[key]}

        ### Notes:
        - Focus only on the identified imbalance (**{key}**).
        - Use only the provided template for the output. Do not include any other headings or sections.
        - Avoid generic responses. Base your explanation strictly on soil science and plant nutrient management principles.
        - Maintain an educational tone that is informative and accessible to agricultural workers and advisors.
        """

        response = co.generate(
            model="command-r-plus-08-2024",
            prompt=prompt,
            max_tokens=1000
        )

        fertilizer_report = response.generations[0].text.strip()
        fertilizer_report = markdown.markdown(fertilizer_report)
        return jsonify({
            "fertilizer_report": fertilizer_report
        })

def predict_image(img):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    model = ResNet9(3, len(disease_classes))
    model.load_state_dict(torch.load('models/plant_disease_model.pth', map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        yb = model(img_u)
        _, preds = torch.max(yb, dim=1)
        prediction = disease_classes[preds[0].item()]
    return prediction

@app.route('/disease_predict', methods=['GET', 'POST'])
def disease_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html')
        img = file.read()
        try:
            prediction = predict_image(img)
            result = Markup(str(disease_dic[prediction]))
            return jsonify({"disease_report": result})
        except Exception as e:
            return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=False)