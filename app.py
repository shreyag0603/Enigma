from flask import Flask, request, render_template, jsonify,redirect,url_for,flash,session, make_response
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from deap import base, creator, tools, algorithms
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import StandardScaler
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import math
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import pytz
from datetime import datetime,timezone
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy import JSON
from collections import defaultdict





# Initialize Flask app
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/register'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Set a secret key for your session
app.config['SECRET_KEY'] = '20326d6e7a5a6857da2bcdbb8d92f2c04424d119efe6451519eb4829a67787d6'
db = SQLAlchemy(app)


# Sample data for cooling analysis
data_cooling = {
    'PP Type': ['Coal', 'Coal', 'Coal', 'Natural Gas', 'Natural Gas', 'Natural Gas', 
                'Nuclear', 'Nuclear', 'Nuclear', 'Biomass', 'Biomass', 'Biomass',
                'Concentrated Solar', 'Concentrated Solar', 'Concentrated Solar',
                'Geothermal', 'Geothermal', 'Geothermal'],
    'Type of Water Usage': ['Once Through', 'Recirculating', 'Dry Cooling',
                            'Once Through', 'Recirculating', 'Dry Cooling',
                            'Once Through', 'Recirculating', 'Dry Cooling',
                            'Once Through', 'Recirculating', 'Dry Cooling',
                            'Once Through', 'Recirculating', 'Dry Cooling',
                            'Once Through', 'Recirculating', 'Dry Cooling'],
    'Water Used per mWh': [25000, 600, 0, 15000, 300, 0, 25000, 800, 0, 
                           18000, 500, 0, 14000, 200, 0, 12000, 200, 0],
    'Energy Consumed': [3, 15, 25, 2, 10, 20, 7, 18, 30, 3, 15, 25, 2, 8, 15, 3, 15, 25],
    'Renewablity': ['No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 
                    'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes']
}
df_cooling = pd.DataFrame(data_cooling)

data=pd.read_csv("datasets/Training_set_ccpp (1).csv")


# Load datasets
device_efficiency_df = pd.read_csv('datasets/updated_data.csv')  # Replace with your dataset path
country_price_df = pd.read_csv('datasets/unique_electricity_prices_by_country.csv')  # Replace with your dataset path

# Load the model and scaler for model2
with open('models/energy_model.pkl', 'rb') as model_file:
    model2 = pickle.load(model_file)

with open('models/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the dataset and initialize model1
df = pd.read_csv('datasets/Carbon Emission.csv')

df_model3 = pd.read_csv('datasets/merged_df_final.csv')

# Define columns for energy sources in model3
energy_sources = {
    'Bioenergy': ('Bioenergy Price', 'Bioenergy (% growth)'),
    'Hydropower': ('Hydropower Price', 'Hydro (% growth)'),
    'Wind': ('Wind Price', 'Wind (% growth)'),
    'Solar': ('Solar Price', 'Solar (% growth)'),
    'Crude Oil': ('Crude Oil Price', 'Oil (% growth)')
}

# Initialize label encoders for categorical variables in model1
label_encoders_model1 = {}
categorical_columns_model1 = ['Body Type', 'Sex', 'Diet', 'How Often Shower', 
                              'Heating Energy Source', 'Transport', 'Vehicle Type', 
                              'Frequency of Traveling by Air', 'Energy efficiency']

df_model1 = df.copy()
for column in categorical_columns_model1:
    le = LabelEncoder()
    df_model1[column] = le.fit_transform(df_model1[column])
    label_encoders_model1[column] = le

df_model1 = df_model1.join(df_model1['Cooking_With'].str.get_dummies(sep=','))
df_model1.drop(columns=['Cooking_With'], inplace=True)

features_model1 = ['Body Type', 'Sex', 'Diet', 'How Often Shower', 
                    'Heating Energy Source', 'Transport', 'Vehicle Type', 
                    'Frequency of Traveling by Air', 'Vehicle Monthly Distance Km', 
                    'Waste Bag Weekly Count', 'How Long TV PC Daily Hour', 
                    'How Long Internet Daily Hour', 'Energy efficiency'] + \
                    [col for col in df_model1.columns if col.startswith('[')]

X_model1 = df_model1[features_model1].values
y_model1 = df_model1['CarbonEmission'].values
X_train_model1, X_test_model1, y_train_model1, y_test_model1 = train_test_split(X_model1, y_model1, test_size=0.2, random_state=42)
model1 = LinearRegression()
model1.fit(X_train_model1, y_train_model1)

device_efficiency_df = device_efficiency_df.dropna()  # Drop rows with missing values

# Define features and target
features = ['Brand', 'EnergyConsumption', 'DeviceType', 'UsageHoursPerDay', 'DeviceAgeMonths', 'MalfunctionIncidents']
X = device_efficiency_df[features]
y = device_efficiency_df['SmartHomeEfficiency']

# Separate features into numerical and categorical
numerical_features = ['EnergyConsumption', 'UsageHoursPerDay', 'DeviceAgeMonths', 'MalfunctionIncidents']
categorical_features = ['Brand', 'DeviceType']

# Create the column transformer with one-hot encoder for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Create and train the model pipeline
model4 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model4.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model1')
def model1_page():
    return render_template('model1.html')

@app.route('/model2')
def model2_page():
    return render_template('model2.html')

@app.route('/predict1', methods=['POST'])
def predict1():
    data = request.form

    # Retrieve form data and provide default values where necessary
    try:
        user_input = {
            'Body Type': data.get('Body Type') or 'Unknown',
            'Sex': data.get('Sex') or 'Unknown',
            'Diet': data.get('Diet') or 'Unknown',
            'How Often Shower': data.get('How Often Shower') or 'Unknown',
            'Heating Energy Source': data.get('Heating Energy Source') or 'Unknown',
            'Transport': data.get('Transport') or 'Unknown',
            'Vehicle Type': data.get('Vehicle Type') or 'Unknown',
            'Frequency of Traveling by Air': data.get('Frequency of Traveling by Air') or 'Unknown',
            'Energy efficiency': data.get('Energy efficiency') or 'Unknown',
            'Cooking_With': data.get('Cooking_With') or '',
            'Vehicle Monthly Distance Km': float(data.get('Vehicle Monthly Distance Km', 0) or 0),
            'Waste Bag Weekly Count': float(data.get('Waste Bag Weekly Count', 0) or 0),
            'How Long TV PC Daily Hour': float(data.get('How Long TV PC Daily Hour', 0) or 0),
            'How Long Internet Daily Hour': float(data.get('How Long Internet Daily Hour', 0) or 0)
        }
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    try:
        user_input_df = preprocess_user_input(user_input, label_encoders_model1, features_model1)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except KeyError as e:
        return jsonify({'error': str(e)}), 400

    predicted_emission = predict_carbon_emission(user_input_df, model1)
    return render_template('result1.html', predicted_emission=round(predicted_emission, 2))

def preprocess_user_input(user_input, label_encoders, features):
    processed_input = {}
    for column in label_encoders:
        if column in user_input:
            le = label_encoders[column]
            if user_input[column] in le.classes_:
                processed_input[column] = le.transform([user_input[column]])[0]
            else:
                raise ValueError(f"Unexpected category '{user_input[column]}' for column '{column}'")
        else:
            raise KeyError(f"Missing expected column: '{column}'")

    # Handle 'Cooking_With' field
    for cooking_method in [col for col in df_model1.columns if col.startswith('[')]:
        processed_input[cooking_method] = 1 if cooking_method.replace('Cooking_With_', '') in user_input['Cooking_With'].split(',') else 0

    # Fill any missing feature columns with default values
    feature_columns = features
    for col in feature_columns:
        if col not in processed_input:
            processed_input[col] = 0

    user_input_df = pd.DataFrame([processed_input], columns=feature_columns)
    return user_input_df

def predict_carbon_emission(user_input_df, model):
    user_input_np = user_input_df.values
    predicted_emission = model.predict(user_input_np)
    predicted_emission = np.maximum(predicted_emission, 0)
    return predicted_emission[0]

# Calculate dynamic thresholds using quantiles or means
thresholds = {
    'at': data['AT'].quantile(0.75),  # 75th percentile of Ambient Temperature
    'ev': data['EV'].quantile(0.75),  # 75th percentile of Exhaust Vacuum
    'rh': data['RH'].quantile(0.25),  # 25th percentile for Relative Humidity (lower is better)
    'ap': data['AP'].quantile(0.75)   # 75th percentile of Ambient Pressure
}

def generate_recommendations(at, ev, rh, ap, thresholds):
    # Dynamic thresholds from the dataset
    threshold_at = thresholds['at']
    threshold_ev = thresholds['ev']
    threshold_rh = thresholds['rh']
    threshold_ap = thresholds['ap']

    # Calculate performance metric components dynamically
    f_at = (at - threshold_at) ** 2
    f_ev = math.log(1 + ev)
    f_rh = math.exp(-rh / 10)
    f_ap = (ap - threshold_ap) ** 3

    # Calculate the composite performance score (weight distribution can be adjusted)
    performance_score = 0.3 * f_at + 0.25 * f_ev + 0.2 * f_rh + 0.25 * f_ap

    # Initialize recommendation list
    recommendations = []

    # Generate recommendations based on dynamic performance metrics
    if at > threshold_at:
        recommendations.append("Consider using cooling techniques to bring the ambient temperature closer to optimal levels.")
    
    if ev > threshold_ev:
        recommendations.append("Reduce exhaust vacuum to lower backpressure and improve turbine efficiency.")
    
    if rh < threshold_rh:
        recommendations.append("Optimize cooling tower operations to handle low relative humidity conditions efficiently.")
    
    if ap > threshold_ap:
        recommendations.append("Adjust combustion settings to accommodate changes in ambient pressure and maintain optimal efficiency.")

    # Additional intelligent recommendations
    if performance_score < np.mean([f_at, f_ev, f_rh, f_ap]):  # Compare against mean of performance metrics
        recommendations.append("System performance is near optimal. Monitor and adjust only if conditions change.")
    else:
        recommendations.append("Implement optimization strategies for cooling, pressure management, and efficiency.")

    return recommendations

@app.route('/process', methods=['POST'])
def process():
    data= request.form
    pp_type = request.form.get('pp_type')

    # Handle energy prediction
    try:
        AT = float(request.form.get('AT'))
        EV = float(request.form.get('EV'))
        AP = float(request.form.get('AP'))
        RH = float(request.form.get('RH'))
        ActualEnergyValue = float(data.get('ActualEnergyValue'))
    except (ValueError, TypeError):
        return render_template('result2.html', result="Invalid input. Ensure that all inputs are numerical and correctly named.", images=None, results=None, show_analysis=False)

    # Prepare the input data for prediction
    input_data = pd.DataFrame([[AT, EV, AP, RH]], columns=['AT', 'EV', 'AP', 'RH'])
    input_data_scaled = scaler.transform(input_data)
    PE_pred = model2.predict(input_data_scaled)
    result = f'Predicted Electrical Energy Output (PE): {round(PE_pred[0], 2)} MW'
    
     # Calculate efficiency rating
    if PE_pred == 0:
         efficiency_rating = "Prediction is zero, cannot calculate efficiency."
    else:
         calculated_efficiency = (ActualEnergyValue / PE_pred) * 100
         efficiency_rating = f'Efficiency Rating: {round(calculated_efficiency[0], 2)} %'
         efficiency_rating = f'{efficiency_rating}' if calculated_efficiency <= 100 else "Efficiency exceeds 100%"
         
           # Generate recommendations
    recommendations = generate_recommendations(AT, EV, RH, AP, thresholds)


    # Pass prediction and power plant type to the result page
    return render_template('result2.html', result=result,efficiency_rating=efficiency_rating, pp_type=pp_type, predicted_energy=PE_pred[0], show_analysis=True, recommendations=recommendations)

@app.route('/analyze', methods=['POST'])
def analyze():
    analysis_choice = int(request.form.get('analysis_choice'))
    pp_type = request.form.get('pp_type')
    predicted_energy = float(request.form.get('predicted_energy'))

    # Filter the data for cooling analysis
    filtered_df = df_cooling[df_cooling['PP Type'] == pp_type]

    # Initialize the `images` and `results` lists
    images = []
    results = []

    # Define color mapping
    colors = {'Dry Cooling': 'red', 'Once Through': 'blue', 'Recirculating': 'green'}

    # Generate plots or cooling results based on the choice
    if analysis_choice == 1:
        # Overall Comparison
        plt.figure(figsize=(10, 6))
        plt.bar(filtered_df['Type of Water Usage'], filtered_df['Water Used per mWh'], color=[colors[x] for x in filtered_df['Type of Water Usage']])
        plt.title(f'Water Usage for {pp_type} Power Plants')
        plt.xticks([])
        plt.yticks([])
        plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=type_) for type_, color in colors.items()], title='Type of Water Usage')
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        images.append('data:image/png;base64,' + base64.b64encode(img.getvalue()).decode())
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.bar(filtered_df['Type of Water Usage'], filtered_df['Energy Consumed'], color=[colors[x] for x in filtered_df['Type of Water Usage']])
        plt.title(f'Energy Consumption for {pp_type} Power Plants')
        plt.xticks([])
        plt.yticks([])
        plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=type_) for type_, color in colors.items()], title='Type of Water Usage')
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        images.append('data:image/png;base64,' + base64.b64encode(img.getvalue()).decode())
        plt.close()

        # Generate Scatter Plot
        filtered_df[['Water Used per mWh', 'Energy Consumed']] = StandardScaler().fit_transform(filtered_df[['Water Used per mWh', 'Energy Consumed']])
        plt.figure(figsize=(10, 6))
        for i, (type_, color) in enumerate(colors.items()):
            subset = filtered_df[filtered_df['Type of Water Usage'] == type_]
            plt.scatter(subset['Water Used per mWh'], subset['Energy Consumed'], label=type_, s=100, color=color)
            for j, row in subset.iterrows():
                plt.text(row['Water Used per mWh'] + 0.1, row['Energy Consumed'] + 0.1, row['Type of Water Usage'], fontsize=9)
        plt.plot([-2, 2], [-2, 2], linestyle='--', color='black', label='Ideal Performance')
        plt.title(f'Standardized Water Usage vs Energy Consumed for {pp_type} Power Plant')
        plt.xticks([])
        plt.yticks([])
        plt.legend(title='Type of Water Usage')
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        images.append('data:image/png;base64,' + base64.b64encode(img.getvalue()).decode())
        plt.close()

    elif analysis_choice == 2:
        # Machine Efficient
        once_through_df = filtered_df[filtered_df['Type of Water Usage'] == 'Once Through']
        results = once_through_df.apply(lambda row: f"Type of Cooling: {row['Type of Water Usage']}, Water required per mWh: {row['Water Used per mWh']}, Energy Consumed: {row['Energy Consumed']}%", axis=1).tolist()

    elif analysis_choice == 3:
        # Environment Efficient
        recirculation_df = filtered_df[filtered_df['Type of Water Usage'] == 'Recirculating']
        results = recirculation_df.apply(lambda row: f"Type of Cooling: {row['Type of Water Usage']}, Water required per mWh: {row['Water Used per mWh']}, Energy Consumed: {row['Energy Consumed']}%", axis=1).tolist()

    elif analysis_choice == 4:
        # Minimalistic Water Usage
        dry_cooling_df = filtered_df[filtered_df['Type of Water Usage'] == 'Dry Cooling']
        results = dry_cooling_df.apply(lambda row: f"Type of Cooling: {row['Type of Water Usage']}, Water required per mWh: {row['Water Used per mWh']}, Energy Consumed: {row['Energy Consumed']}%", axis=1).tolist()

    return render_template('result2.html', result=f'Predicted Electrical Energy Output (PE): {round(predicted_energy, 2)} MW', images=images, results=results, show_analysis=False)


@app.route('/model3')
def model3():
    return render_template('model3.html')

@app.route('/predict3', methods=['POST'])
def results():
    budget_constraints = float(request.form['budget_constraints'])
    energy_requirements = float(request.form['energy_requirements'])
    location = request.form['location']

    df_filtered = df_model3[(df_model3['Year'] == 2022) & (df_model3['Entity'] == location)].copy()

    # Prepare DataFrame for calculations
    data = {
        'Energy Source': list(energy_sources.keys()),
        'Price': [df_filtered[col[0]].values[0] for col in energy_sources.values()],
        'Growth Percentage': [df_filtered[col[1]].values[0] for col in energy_sources.values()]
    }

    df_energy = pd.DataFrame(data)
    df_energy['Annual Consumption'] = df_energy['Growth Percentage']

    def objective_function(percentages):
        total_cost = np.sum(df_energy['Price'] * df_energy['Annual Consumption'] * np.array(percentages))
        return total_cost,

    def constraint_energy(percentages):
        total_energy = np.sum(df_energy['Annual Consumption'] * np.array(percentages))
        return total_energy - energy_requirements

    def constraint_budget(percentages):
        total_cost = np.sum(df_energy['Price'] * df_energy['Annual Consumption'] * np.array(percentages))
        return budget_constraints - total_cost

    def valid_solution(individual):
        individual_array = np.array(individual)
        return np.all(individual_array >= 0) and \
               np.isclose(np.sum(individual_array), 1) and \
               constraint_energy(individual) >= 0 and \
               constraint_budget(individual) >= 0

    def normalize(individual):
        individual_array = np.array(individual)
        individual_array = np.maximum(individual_array, 0)
        total = np.sum(individual_array)
        if total > 0:
            return list(individual_array / total)
        else:
            return list(individual_array)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(df_energy))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", objective_function)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def get_penalty(budget):
        if budget <= 200000:
            return 1e4
        elif budget <= 500000:
            return 1e3
        else:
            return 1e2

    penalty_value = get_penalty(budget_constraints)
    toolbox.decorate("evaluate", tools.DeltaPenalty(valid_solution, penalty_value))

    def mate_and_normalize(ind1, ind2):
        tools.cxBlend(ind1, ind2, alpha=0.5)
        ind1[:] = normalize(ind1)
        ind2[:] = normalize(ind2)
        return ind1, ind2

    def mutate_and_normalize(individual):
        tools.mutGaussian(individual, mu=0, sigma=0.2, indpb=0.3)
        individual[:] = normalize(individual)
        return individual,

    toolbox.register("mate", mate_and_normalize)
    toolbox.register("mutate", mutate_and_normalize)

    if budget_constraints <= 200000:
        hof_size = 5
    elif budget_constraints <= 500000:
        hof_size = 10
    else:
        hof_size = 20

    population = toolbox.population(n=200)
    hof = tools.HallOfFame(hof_size)

    algorithms.eaMuPlusLambda(population, toolbox, mu=200, lambda_=400, cxpb=0.6, mutpb=0.3, ngen=100,
                              halloffame=hof, verbose=True)

    def find_closest_combination(individual):
        best_combination = None
        smallest_diff = float('inf')

        for r in range(1, len(df_energy) + 1):
            for combo in combinations(range(len(df_energy)), r):
                combo_sum = np.sum(np.array(individual)[list(combo)])
                diff = abs(1 - combo_sum)
                if diff < smallest_diff:
                    smallest_diff = diff
                    best_combination = combo

        return best_combination, smallest_diff

    def calculate_efficiency_rating(input_percentages):
        energy_data = {
            'Bioenergy': [4.6, 150],
            'Hydropower': [0.04, 11],
            'Wind': [0.02, 53],
            'Solar': [1.3, 24],
            'Crude Oil': [24, 970]
        }
        weighted_deaths = sum(energy_data[energy][0] * (percentage / 100) for energy, percentage in input_percentages.items())
        weighted_emissions = sum(energy_data[energy][1] * (percentage / 100) for energy, percentage in input_percentages.items())
        max_deaths = max(value[0] for value in energy_data.values())
        max_emissions = max(value[1] for value in energy_data.values())
        normalized_deaths = weighted_deaths / max_deaths
        normalized_emissions = weighted_emissions / max_emissions

        efficiency_rating = (1 - (normalized_deaths + normalized_emissions) / 2) * 100
        return weighted_deaths, weighted_emissions, efficiency_rating

    def calculate_environmental_impact(input_percentages):
        energy_data = {
            'Bioenergy': 150,
            'Hydropower': 11,
            'Wind': 53,
            'Solar': 24,
            'Crude Oil': 970
        }
        total_emissions = sum(energy_data[energy] * (percentage / 100) * energy_requirements for energy, percentage in input_percentages.items())
        return total_emissions / 1000

    filtered_solutions = []
    for ind in hof:
        combo_indices, deviation = find_closest_combination(ind)
        total_cost = objective_function(ind)[0]
        potential_savings = budget_constraints - total_cost

        if deviation < 0.05 and potential_savings > 0:
            filtered_solutions.append((ind, combo_indices, potential_savings))

    filtered_solutions = sorted(filtered_solutions, key=lambda x: calculate_efficiency_rating(
        {df_energy['Energy Source'].iloc[idx]: x[0][idx] * 100 for idx in find_closest_combination(x[0])[0]}
    )[2], reverse=True)

    results = []
    if not filtered_solutions:
        results.append("No feasible solutions found.")
    else:
        for i, (best_individual, combo_indices, potential_savings) in enumerate(filtered_solutions):
            total_cost = objective_function(best_individual)[0]
            total_energy = np.sum(df_energy['Annual Consumption'] * np.array(best_individual))

            solution_percentages = {df_energy['Energy Source'].iloc[idx]: best_individual[idx] * 100 for idx in combo_indices}
            _, _, rating = calculate_efficiency_rating(solution_percentages)
            total_emissions = calculate_environmental_impact(solution_percentages)
            main_energy_source = max(solution_percentages, key=solution_percentages.get)

            results.append({
                "solution_index": i + 1,
                "percentages": solution_percentages,
                "estimated_cost": int(total_cost),
                "total_energy": int(total_energy),
                "potential_savings": int(potential_savings),
                "efficiency_rating": int(rating),
                "environmental_impact": f"Reduce carbon emissions by {int(total_emissions)} kg CO2/year",
                "main_energy_source": main_energy_source
            })

    return render_template('result3.html', results=results)

@app.route('/model4' , methods=['GET', 'POST'])
def model4_page():
    """Render the main page with the input form."""
    return render_template('index.html')


@app.route('/predict4', methods=['POST'])
def predict4():
    """Handle form submission, make prediction, and return results."""
    # Extract form data
    brand = request.form.get('Brand')
    energy_consumption = float(request.form.get('EnergyConsumption'))
    device_type = request.form.get('DeviceType')
    usage_hours_per_day = float(request.form.get('UsageHoursPerDay'))
    device_age_months = int(request.form.get('DeviceAgeMonths'))
    malfunction_incidents = int(request.form.get('MalfunctionIncidents'))
    country_name = request.form.get('CountryName')

    # Fetch the price per unit for the specified country
    country_price = country_price_df .loc[country_price_df ['Country Name'] == country_name, 'Price of Electricity (US cents per kWh)']

    if country_price.empty:
        return jsonify({'error': 'Country not found in the dataset. Cannot calculate price.'})

    price_per_unit = country_price.values[0] / 100  # Convert from cents to dollars

    # Calculate the price based on Energy Consumption
    price = energy_consumption * price_per_unit

    # Create DataFrame for the user input
    user_input_df = pd.DataFrame([{
        'Brand': brand,
        'EnergyConsumption': energy_consumption,
        'DeviceType': device_type,
        'UsageHoursPerDay': usage_hours_per_day,
        'DeviceAgeMonths': device_age_months,
        'MalfunctionIncidents': malfunction_incidents
    }])

    # Apply preprocessing to the input
    user_input_transformed = model4.named_steps['preprocessor'].transform(user_input_df)

    # Predict using the model
    user_prediction = model4.named_steps['classifier'].predict(user_input_transformed)
    
    # Output the result
    result = 'The device is running effectively with good efficiency.' if user_prediction[0] == 1 else 'The device is not working as efficiently as it should.'

    # Render the result page
    return render_template('result4.html', result=result, price=f"${price:.2f}")

# Models
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    login_count = db.Column(db.Integer, default=0)
    logins = db.Column(MutableList.as_mutable(JSON), default=[])
    

    def __repr__(self):
        return f'<User {self.email}>'

class TransactionPro(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    cardholder_name = db.Column(db.String(100), nullable=False)
    cardnumber = db.Column(db.String(16), nullable=False)
    email = db.Column(db.String(120), db.ForeignKey('users.email'), nullable=False)
    pro = db.Column(db.Boolean, default=True)

class TransactionPremium(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    cardholder_name = db.Column(db.String(100), nullable=False)
    cardnumber = db.Column(db.String(16), nullable=False)
    email = db.Column(db.String(120), db.ForeignKey('users.email'), nullable=False)
    premium = db.Column(db.Boolean, default=True)
# Create the database tables
with app.app_context():
    db.create_all()


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered!', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(name=name, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful!', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index1'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            user.login_count += 1
            
            ist_timezone = pytz.timezone('Asia/Kolkata')
            current_time = datetime.now(ist_timezone)
            
            if user.logins is None:
                user.logins = []
            user.logins.append(current_time.isoformat())
            
            user.last_login = current_time
            

            db.session.commit()

            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': True})
            else:
                flash('Login successful!', 'success')
                return redirect(url_for('index1'))
        else:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'message': 'Invalid email or password!'})
            else:
                flash('Invalid email or password!', 'danger')
                return redirect(url_for('login'))
          

    return render_template('login.html')

@app.route('/index')
def index1():
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)
        
        pro_subscription = TransactionPro.query.filter_by(email=user.email).first()
        premium_subscription = TransactionPremium.query.filter_by(email=user.email).first()

        response = make_response(render_template('index.html', pro_subscription=pro_subscription, premium_subscription=premium_subscription))
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response

    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index1'))

@app.route('/process_transaction_pro', methods=['POST'])
def process_transaction_pro():
    cardholder_name = request.form.get('cardholder_name')
    cardnumber = request.form.get('cardnumber')
    email = request.form.get('email')

    # Check if the user is logged in
    if 'user_id' not in session:
        flash('Please log in to complete the transaction.', 'danger')
        return redirect(url_for('login'))

    # Get the registered email for the logged-in user
    user_email = User.query.get(session['user_id']).email

    # Validate if the provided email matches the registered email
    if email != user_email:
        flash('The email provided does not match the one used for registration or login. Please use the registered email.', 'danger')
        return redirect(url_for('transaction_pro_page'))

    # Check if the user already has a Pro subscription
    existing_pro_subscription = TransactionPro.query.filter_by(email=email).first()
    if existing_pro_subscription:
        flash('You already have a Pro subscription. You cannot subscribe again with the same email.', 'danger')
        return redirect(url_for('transaction_pro_page'))

    # Process the transaction if the email is valid
    transaction = TransactionPro(cardholder_name=cardholder_name, cardnumber=cardnumber, email=email, pro=True)
    db.session.add(transaction)
    db.session.commit()

    session['subscription_success'] = 'pro'
    return redirect(url_for('pricing_page'))



@app.route('/process_transaction_premium', methods=['POST'])
def process_transaction_premium():
    cardholder_name = request.form.get('cardholder_name')
    cardnumber = request.form.get('cardnumber')
    email = request.form.get('email')

    # Check if the user is logged in
    if 'user_id' not in session:
        flash('Please log in to complete the transaction.', 'danger')
        return redirect(url_for('login'))

    # Get the registered email for the logged-in user
    user_email = User.query.get(session['user_id']).email

    # Validate if the provided email matches the registered email
    if email != user_email:
        flash('The email provided does not match the one used for registration or login. Please use the registered email.', 'danger')
        return redirect(url_for('transaction_premium_page'))

    # Check if the user already has a Premium subscription
    existing_premium_subscription = TransactionPremium.query.filter_by(email=email).first()
    if existing_premium_subscription:
        flash('You already have a Premium subscription. You cannot subscribe again with the same email.', 'danger')
        return redirect(url_for('transaction_premium_page'))

    # Process the transaction if the email is valid
    transaction = TransactionPremium(cardholder_name=cardholder_name, cardnumber=cardnumber, email=email, premium=True)
    db.session.add(transaction)
    db.session.commit()

    session['subscription_success'] = 'premium'
    return redirect(url_for('pricing_page'))

@app.route('/check_login_and_subscription')
def check_login_and_subscription():
    is_logged_in = 'user_id' in session
    subscription_success = session.get('subscription_success')
    return jsonify({
        'is_logged_in': is_logged_in,
        'subscription_success': subscription_success
    })

@app.route('/check_subscription/<subscription_type>')
def check_subscription(subscription_type):
    if 'user_id' not in session:
        return jsonify({'has_subscription': False, 'is_logged_in': False})

    user_email = User.query.get(session['user_id']).email

    if subscription_type == 'pro':
        subscription = TransactionPro.query.filter_by(email=user_email).first()
    elif subscription_type == 'premium':
        subscription = TransactionPremium.query.filter_by(email=user_email).first()
    else:
        return jsonify({'has_subscription': False, 'is_logged_in': True})

    return jsonify({'has_subscription': subscription is not None, 'is_logged_in': True})

@app.route('/clear_subscription_success', methods=['POST'])
def clear_subscription_success():
    session.pop('subscription_success', None)
    return jsonify({'success': True})

@app.route('/pricing')
def pricing_page():
    return render_template('pricing.html')

@app.route('/transactionpro')
def transaction_pro_page():
    return render_template('transactionpro.html')

@app.route('/transactionpremium')
def transaction_premium_page():
    return render_template('transactionpremium.html')

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize the language model
llm = ChatGroq(model="llama3-70b-8192", api_key=api_key, temperature=0)

# Define the prompt templates for generating questions and answers
question_prompt_template = PromptTemplate(
    input_variables=["text"],
    template="From the following text, generate exactly one question:\n\n{text}\n\nQuestion:"
)

answer_prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="Given the context, provide a concise answer to the question.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
)

summary_prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in a concise manner:\n\n{text}\n\nSummary:"
)


# Create chains for QA generation
qa_chain = LLMChain(llm=llm, prompt=question_prompt_template)
answer_chain = LLMChain(llm=llm, prompt=answer_prompt_template)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt_template)


def chunk_text(text, chunk_size=1000):
    # Split the text into chunks of the specified size
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def generate_qa_for_file(filename):
    questions = []
    answers = []

    # Read the content of the text file
    with open(f'static/{filename}', 'r') as file:
        file_content = file.read()

    # Split the file content into chunks
    chunks = chunk_text(file_content)
    
    # Generate questions and answers for up to 5 chunks
    for chunk in chunks:
        if len(questions) >= 5:  # Stop after generating 5 questions
            break
        
        question = qa_chain.run(chunk).strip()
        answer = answer_chain.run({"question": question, "context": chunk}).strip()
        questions.append(question)
        answers.append({"question": question, "answer": answer})

    return questions, answers

def generate_summary_for_file(filename):
    # Read the content of the text file
    with open(f'static/{filename}', 'r') as file:
        file_content = file.read()
    
    # Generate a summary for the file content
    summary = summary_chain.run(file_content).strip()
    
    return summary


@app.route('/blog1')
def blog1():
    return render_template('blog1.html')

@app.route('/blog2')
def blog2():
    return render_template('blog2.html')

@app.route('/blog3')
def blog3():
    return render_template('blog3.html')

@app.route('/blog4')
def blog4():
    return render_template('blog4.html')

@app.route('/generate_faq', methods=['POST'])
def generate_faq():
    blog_file = request.form['blog_file']
    questions, answers = generate_qa_for_file(blog_file)
    return render_template('results_blog.html', questions=questions, answers=answers)

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    file_name = request.form.get('blog_file', 'blog1.txt')  # Default to 'ME.txt' if no file is specified
    
    # Check if the file exists
    file_path = f'static/{file_name}'
    if not os.path.isfile(file_path):
        return f"Error: The file '{file_name}' does not exist.", 404
    
    # Generate summary
    summary = generate_summary_for_file(file_name)
    
    return render_template('summary_blog.html', summary=summary, file_name=file_name)

from datetime import datetime
from collections import defaultdict

@app.route('/trends')
def trends():
    user_id = session.get('user_id')
    if user_id:
        user = User.query.get(user_id)
        if user:
            login_dates = [datetime.fromisoformat(login) for login in user.logins]
            
            # Sort login dates
            login_dates.sort()
            
            # Calculate the date range
            date_range = (login_dates[-1] - login_dates[0]).days
            
            # Group data based on the date range
            if date_range <= 31:  # Within a month
                grouped_data = defaultdict(int)
                for date in login_dates:
                    grouped_data[date.strftime('%Y-%m-%d')] += 1
                time_format = '%Y-%m-%d'
                chart_type = 'day'
            elif date_range <= 365:  # Within a year
                grouped_data = defaultdict(int)
                for date in login_dates:
                    grouped_data[date.strftime('%Y-%m')] += 1
                time_format = '%Y-%m'
                chart_type = 'month'
            else:  # More than a year
                grouped_data = defaultdict(int)
                for date in login_dates:
                    grouped_data[date.strftime('%Y')] += 1
                time_format = '%Y'
                chart_type = 'year'
            
            labels = list(grouped_data.keys())
            counts = list(grouped_data.values())
            
            return render_template('trends.html', labels=labels, counts=counts, chart_type=chart_type, time_format=time_format)
    return redirect(url_for('login'))

@app.route('/viz1')
def viz1():
    return render_template('viz1.html')

@app.route('/viz2')
def viz2():
    return render_template('viz2.html')

@app.route('/viz3')
def viz3():
    return render_template('viz3.html')

@app.route('/services1')
def services1():
    return render_template('services1.html')

@app.route('/services2')
def services2():
    return render_template('services2.html')

@app.route('/services3')
def services3():
    return render_template('services3.html')

@app.route('/services4')
def services4():
    return render_template('services4.html')

@app.route('/services5')
def services5():
    return render_template('services5.html')

@app.route('/services6')
def services6():
    return render_template('services6.html')

@app.route('/forgotpassword')
def forgotpassword():
    return render_template('forgotpassword.html')

if __name__ == '__main__':
    app.run(debug=True)