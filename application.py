from flask import Flask, request, render_template
from src.NO_SHOW_MLPROJECT.pipelines.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for Home Page:
@app.route('/')
def index():
    return render_template('index.html')

# Route for displaying the form
@app.route('/form', methods=['GET'])
def predict_data_form():
    print("Received GET request for form")
    return render_template('home.html')

# Route for handling form submission and making predictions
@app.route('/predict', methods=['POST'])
def predict_datapoint():
    print("Received POST request")

    # Extract data from the form submission
    Gender = request.form.get('Gender')
    Age = int(request.form.get('Age'))
    Alcohol_Consumption = request.form.get('Alcohol_Consumption')
    Hypertension = request.form.get('Hypertension')
    Diabetes = request.form.get('Diabetes')
    Appointment_Date = request.form.get('Appointment_Date')
    Schedule_Date = request.form.get('Schedule_Date')
    Clinic_Location = request.form.get('Clinic_Location')
    Specialty = request.form.get('Specialty')
    Neighborhood = request.form.get('Neighborhood')

    # Print the extracted data for verification
    print(f"Gender: {Gender}")
    print(f"Age: {Age}")
    print(f"Alcohol Consumption: {Alcohol_Consumption}")
    print(f"Hypertension: {Hypertension}")
    print(f"Diabetes: {Diabetes}")
    print(f"Appointment Date: {Appointment_Date}")
    print(f"Schedule Date: {Schedule_Date}")
    print(f"Clinic Location: {Clinic_Location}")
    print(f"Specialty: {Specialty}")
    print(f"Neighborhood: {Neighborhood}")

    # Initialize CustomData with the extracted data
    data = CustomData(
        Gender=Gender,
        Age=Age,
        Alcohol_Consumption=Alcohol_Consumption,
        Hypertension=Hypertension,
        Diabetes=Diabetes,
        Appointment_Date=Appointment_Date,
        Schedule_Date=Schedule_Date,
        Clinic_Location=Clinic_Location,
        Specialty=Specialty,
        Neighborhood=Neighborhood
    )

    # Convert the CustomData to a DataFrame
    pred_df = data.get_data_as_dataframe()

    # Initialize PredictPipeline and make prediction
    predict_pipeline = PredictPipeline()
    result = predict_pipeline.predict(pred_df)

    # Return the result to the user
    return render_template('home.html', result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0")





# Accuracy: 92.46%
# Precision: 95.92%
# Recall: 57.13%
# F1 Score: 71.61%
# ROC AUC: 91.49%

# Confusion Matrix:
# True Positives (TP): 918
# False Positives (FP): 39
# True Negatives (TN): 8015
# False Negatives (FN): 689