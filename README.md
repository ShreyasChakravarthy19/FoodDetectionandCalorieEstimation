<h1 align="center">🍽️ Food Detection and Calorie Estimation Using Machine Learning 🍽️</h1>

<p align="center">

</p>

## 🚀 Overview
This project focuses on **food detection and calorie estimation** using machine learning techniques. It employs deep learning models to classify food items and estimate their caloric values, aiding in **dietary monitoring and health management**. The system is designed for real-world applications such as **mobile dietary tracking, fitness management, and healthcare monitoring**.

## 🎯 Features
✅ **Food Recognition**: Uses deep learning models to classify food items from images.  
✅ **Calorie Estimation**: Maps detected food items to a nutritional database to estimate caloric content.  
✅ **Portion Size Estimation**: Uses geometric and AI-based techniques for accurate portion size analysis.  
✅ **Real-time Performance**: Optimized for mobile and embedded deployment.  
✅ **Meal Logging**: Users can maintain a history of their food intake and track daily nutrition.  
✅ **User Dashboard**: Provides insights into calorie consumption trends and personalized recommendations.

## 🛠️ Technologies Used
- **Languages**:  
  ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

- **Frameworks & Libraries**:  
  ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
  ![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
  ![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
  ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

- **Deep Learning Models**:  
  ![EfficientNet](https://img.shields.io/badge/EfficientNet-009688?style=for-the-badge)
  ![MobileNetV2](https://img.shields.io/badge/MobileNetV2-4CAF50?style=for-the-badge)

- **Databases**:  
  ![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
  ![USDA API](https://img.shields.io/badge/USDA-Database-4A90E2?style=for-the-badge)

- **Frontend**:  
  ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
  ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
  ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
  ![Jinja2](https://img.shields.io/badge/Jinja2-B41717?style=for-the-badge&logo=jinja&logoColor=white)


## 📂 Dataset
📌 **Food-20 Dataset**: Used for training and testing food classification models.  
📌 **USDA Nutritional Database**: Provides caloric and nutrient values for identified food items.

## 🏗️ Installation
```bash
# Clone the Repository
git clone https://github.com/YOUR-USERNAME/Food-Detection-Calories.git
cd Food-Detection-Calories

# Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt

# Run the Application
python app.py
```

## 📊 Usage
1️⃣ Upload an image of a meal.  
2️⃣ The system classifies the food items.  
3️⃣ Estimated calorie values and portion sizes are displayed.  
4️⃣ Users can log meals and track daily nutritional intake.  
5️⃣ View meal history and analyze dietary patterns through the dashboard.  

## 🔧 System Architecture
📌 **Frontend**: Built using HTML, CSS, and JavaScript, providing an intuitive interface.  
📌 **Backend**: Flask handles API requests and machine learning model inference.  
📌 **Database**: Stores user meal logs and nutritional data.  
📌 **ML Pipeline**: Image preprocessing, food classification, calorie estimation, and result storage.  

## 📈 Results
| Metric  | Value  |
|---------|--------|
| **Food Detection Accuracy** | 92.5% (EfficientNet-B0) |
| **Calorie Estimation Accuracy** | 88% based on food image mapping |
| **Inference Speed** | Optimized for real-time performance on mobile devices |

## 🚀 Future Improvements
✨ **Multi-food Detection**: Identify multiple food items in one image.  
✨ **3D Volume Estimation**: Improve portion size accuracy using depth estimation.  
✨ **Mobile App Integration**: Deploy on Android/iOS for real-time tracking.  
✨ **Voice & Text-Based Food Logging**: Log meals without image input.  
✨ **Personalized Nutrition Recommendations**: AI-based dietary insights.  

## 📜 License
This project is licensed under the **MIT License** - see the LICENSE file for details.

---
<p align="center">Made with ❤️ by Machine Learning Enthusiasts</p>

