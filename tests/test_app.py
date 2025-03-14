import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask import flash
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import cv2
import datetime

app = Flask(__name__)

app.secret_key = os.urandom(24)

# Folder to save uploaded images temporarily
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
model_path = 'E:/Projects/Mini Project Food Detection and Calorie Estimation/MiniProj FDCE/saved models/my_model_eff.keras'
model = load_model(model_path)
print("Model loaded Successfully!")

# Load the nutritional dataset
nutrition_file_path = r'E:\\Projects\\Mini Project Food Detection and Calorie Estimation\\MiniProj FDCE\\food20dataset\\metadata\\dataset .csv'
nutrition_df = pd.read_csv(nutrition_file_path)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mini_proj.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Enhanced User Table
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    password_hash = db.Column(db.String(200), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Float, nullable=False)  # in cm
    weight = db.Column(db.Float, nullable=False)  # in kg
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    # Relationship with MealLog
    meal_logs = db.relationship('MealLog', backref='user', lazy=True)

    # Add a property to calculate BMI
    @property
    def bmi(self):
        height_in_meters = self.height / 100
        return round(self.weight / (height_in_meters ** 2), 2)

    # Add a property to get BMI category
    @property
    def bmi_category(self):
        bmi = self.bmi
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal weight"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"

    def set_password(self, password):
        """Create password hash."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Check if password is correct."""
        return check_password_hash(self.password_hash, password)

# Enhanced Meal Log Table
class MealLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    meal_type = db.Column(db.String(50), nullable=False)  # breakfast, lunch, dinner, snack
    food_detected = db.Column(db.String(100), nullable=False)
    calories = db.Column(db.Float, nullable=False)
    protein = db.Column(db.Float, nullable=False)
    carbs = db.Column(db.Float, nullable=False)
    fat = db.Column(db.Float, nullable=False)
    image_path = db.Column(db.String(255))
    confidence = db.Column(db.Float)
    proportion = db.Column(db.Float)
    date_logged = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# Existing code for image processing and prediction functions remains the same...
# (Include all the previous functions: preprocess_image, estimate_proportion, get_nutritional_info, predict_food)

# Class-to-food mapping
class_to_food = {
    0: "Biryani",
    1: "Bisibelebath",
    2: "Butter Naan",
    3: "Chaat",
    4: "Chapati",
    5: "Dhokla",
    6: "Dosa",
    7: "Gulab Jamun",
    8: "Halwa",
    9: "Idly",
    10: "Kathi Roll",
    11: "Medu Vadai",
    12: "Noodles",
    13: "Paniyaram",
    14: "Poori",
    15: "Samosa",
    16: "Tandoori Chicken",
    17: "Upma",
    18: "Vada Pav",
    19: "Ven Pongal",
}

# Function to preprocess the image for prediction
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to estimate the proportion of the food in the image
def estimate_proportion(image_path):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(original_image, (224, 224))
    gray = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    non_zero_pixels = cv2.countNonZero(binary_mask)
    total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
    proportion = non_zero_pixels / total_pixels
    return proportion * 100

def get_nutritional_info(food_name):
    food_info = nutrition_df[nutrition_df['Food Item'].str.lower() == food_name.lower()]
    if not food_info.empty:
        row = food_info.iloc[0]

        def parse_value(value):
            try:
                # Convert to string and remove 'g' if present
                str_value = str(value).replace('g', '').strip()

                # Handle range values
                if '\u2013' in str_value or '-' in str_value:
                    # Split on dash, convert to float, and take average
                    parts = str_value.replace('\u2013', '-').split('-')
                    nums = [float(p.strip()) for p in parts if p.strip()]
                    return round(sum(nums) / len(nums), 2)

                # Direct conversion for numeric values
                return round(float(str_value), 2)
            except (ValueError, TypeError):
                return 0

        return {
            'Food Item': row.get('Food Item', ''),
            'Calories': parse_value(row.get('Calories', 0)),
            'Protein': parse_value(row.get('Protein', 0)),
            'Carbs': parse_value(row.get('Carbs', 0)),
            'Fat': parse_value(row.get('Fat', 0))
        }
    return {"Calories": 0, "Protein": 0, "Carbs": 0, "Fat": 0}

# Prediction logic
def predict_food(image_path):
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    food_name = class_to_food.get(predicted_class, "Unknown")
    proportion = estimate_proportion(image_path)
    nutrition_info = get_nutritional_info(food_name)
    return food_name, proportion, nutrition_info

# Route for the home page (index.html)
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in
    return render_template('index.html')  # If logged in, return the index page

# Signup Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            # Get form data
            username = request.form.get('username')
            password = request.form.get('password')
            age = request.form.get('age')
            height = request.form.get('height')
            weight = request.form.get('weight')

            # Print received data for debugging
            print(f"Received signup data - Username: {username}, Age: {age}, Height: {height}, Weight: {weight}")

            # Validate required fields
            if not all([username, password, age, height, weight]):
                flash('All fields are required!', 'error')
                return render_template('signup.html', error="All fields are required!")

            # Convert numeric fields
            try:
                age = int(age)
                height = float(height)
                weight = float(weight)
            except ValueError:
                flash('Invalid numeric values provided!', 'error')
                return render_template('signup.html', error="Please enter valid numbers for age, height, and weight!")

            # Check if user already exists
            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                flash('Username already exists!', 'error')
                return render_template('signup.html', error="Username already exists!")

            # Create new user
            new_user = User(
                username=username,
                age=age,
                height=height,
                weight=weight
            )
            new_user.set_password(password)

            # Add to database
            db.session.add(new_user)
            db.session.commit()
            print(f"Successfully created user: {username}")

            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            db.session.rollback()
            print(f"Error during signup: {str(e)}")
            flash('An error occurred during signup!', 'error')
            return render_template('signup.html', error=f"An error occurred: {str(e)}")

    # GET request - display signup form
    return render_template('signup.html')

# Dashboard Route
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    return render_template('dashboard.html', user=user)

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Validate credentials
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id  # Save user ID in session
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid username or password.")

    return render_template('login.html')

# New Route to Log Meal Intake
@app.route('/log_meal', methods=['POST'])
def log_meal():
    if 'user_id' not in session:
        return jsonify({"error": "User not logged in"}), 401

    user_id = session['user_id']
    meal_type = request.json.get('meal_type')
    food_detected = request.json.get('food_detected')
    calories = request.json.get('calories', 0)
    protein = request.json.get('protein', 0)
    carbs = request.json.get('carbs', 0)
    fat = request.json.get('fat', 0)

    print(f"Logging meal: user_id={user_id}, meal_type={meal_type}, food_detected={food_detected}, "
          f"calories={calories}, protein={protein}, carbs={carbs}, fat={fat}")

    if not meal_type or not food_detected:
        return jsonify({"error": "Meal type and food name are required"}), 400

    # Log meal in the database
    new_meal_log = MealLog(
        user_id=user_id,
        meal_type=meal_type,
        food_detected=food_detected,
        calories=calories,
        protein=protein,
        carbs=carbs,
        fat=fat,
        date_logged=datetime.datetime.now()
    )
    db.session.add(new_meal_log)
    db.session.commit()

    print("Meal logged successfully!")
    return jsonify({"success": True, "message": "Meal logged successfully!"})

# Route to Get User's Meal History
@app.route('/meal_history', methods=['GET'])
def meal_history():
    if 'user_id' not in session:
        return jsonify({"error": "User not logged in"}), 401

    user_id = session['user_id']

    # Get optional date range parameters
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # Build the query with user filtering
    query = MealLog.query.filter_by(user_id=user_id)

    # Apply date filtering if provided
    if start_date:
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        query = query.filter(MealLog.date_logged >= start_date)
    if end_date:
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        # Add one day to include the entire end date
        end_date = end_date + datetime.timedelta(days=1)
        query = query.filter(MealLog.date_logged < end_date)

    # Order by date descending
    meal_logs = query.order_by(MealLog.date_logged.desc()).all()

    meal_data = [
        {
            "meal_type": log.meal_type,
            "food_detected": log.food_detected,
            "date_logged": log.date_logged.strftime("%Y-%m-%d %H:%M:%S"),
            "calories": log.calories,
            "protein": log.protein,
            "carbs": log.carbs,
            "fat": log.fat,
        }
        for log in meal_logs  # Use the filtered query results
    ]

    return jsonify(meal_data=meal_data)

@app.route('/debug_meal_logs', methods=['GET'])
def debug_meal_logs():
    all_logs = MealLog.query.all()
    results = []
    for log in all_logs:
        results.append({
            'id': log.id,
            'user_id': log.user_id,
            'meal_type': log.meal_type,
            'food_detected': log.food_detected,
            'calories': log.calories,
            'date_logged': log.date_logged.isoformat()
        })

    return jsonify(results)

@app.route('/meal_history_page', methods=['GET'])
def meal_history_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    return render_template('meal_history.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log request information
        print("Received a request at /predict")

        # Check if files are in the request
        if 'file' not in request.files:
            print("No file part in the request")
            return jsonify({"error": "No file part in the request"}), 400

        files = request.files.getlist('file')
        if not files:
            print("No files provided")
            return jsonify({"error": "No files provided"}), 400

        # Log the files received
        print("Files received:", [file.filename for file in files])

        # Initialize results
        dishes = []
        total_nutrition = {"Calories": 0, "Protein": 0, "Carbs": 0, "Fat": 0}

        # Add two helper functions right in the route
        def allowed_file(filename):
            """Check if the file has an allowed extension."""
            return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

        def save_file(file):
            """Save the uploaded file to the upload folder."""
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            return filename

        def process_image(file_path):
            """Process the image and return prediction results."""
            try:
                food_name, proportion, nutrition = predict_food(file_path)
                return food_name, proportion, nutrition
            except Exception as e:
                print(f"Error processing image {file_path}: {e}")
                raise

        # Process each file
        for file in files:
            if file and allowed_file(file.filename):  # Ensure file is valid
                try:
                    file_path = save_file(file)  # Save the file locally
                    print(f"Saved file: {file_path}")

                    # Process the image
                    food, proportion, nutrition = process_image(file_path)
                    print(f"Predicted {file.filename}: Food={food}, Nutrition={nutrition}")

                    if 'user_id' in session:
                        new_meal_log = MealLog(
                            user_id=session['user_id'],
                            meal_type='lunch',  # You might want to make this dynamic
                            food_detected=food,
                            calories=nutrition.get('Calories', 0),
                            protein=nutrition.get('Protein', 0),
                            carbs=nutrition.get('Carbs', 0),
                            fat=nutrition.get('Fat', 0),
                            image_path=file_path.replace('\\\\', '/'),
                            confidence=proportion,
                            date_logged=datetime.datetime.now()
                        )
                        db.session.add(new_meal_log)
                        db.session.commit()
                        print(f"Meal logged: {food}")

                    # Append the dish data
                    dishes.append({
                        "image_path": file_path.replace('\\\\', '/'),  # Normalize path for web
                        "food": food,
                        "proportion": proportion,
                        "nutrition": nutrition,
                    })

                    # Aggregate total nutrition
                    for key in total_nutrition:
                        total_nutrition[key] += float(nutrition.get(key, 0))

                except Exception as file_process_error:
                    print(f"Error processing file {file.filename}: {file_process_error}")
                    # Optional: Continue processing other files or return an error
                    return jsonify({"error": f"Error processing file {file.filename}"}), 500

            else:
                print(f"Invalid file: {file.filename}")
                return jsonify({"error": f"Invalid file type: {file.filename}"}), 400

        # If no valid files were processed
        if not dishes:
            return jsonify({"error": "No valid images processed"}), 400

        print("Final dishes:", dishes)
        print("Total nutrition:", total_nutrition)

        # Return the results
        return jsonify({"dishes": dishes, "total_nutrition": total_nutrition})

    except Exception as e:
        print(f"Server error: {str(e)}")
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/intake')
def intake():
    # Ensure user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
   