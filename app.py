from flask import Flask, render_template, request
from gradientai import Gradient # Assuming you have a Gradient library for handling models
import os
import csv

app = Flask(__name__)

# Set the Gradient environment variables
os.environ['GRADIENT_ACCESS_TOKEN'] = "hChzE2VjNTevcEtv8W02c868Gj2KtkVg"
os.environ['GRADIENT_WORKSPACE_ID'] = "4516658f-2630-4114-af57-6e73c033ab58_workspace"

# Define the Dataset Path
carrer_dataset_path = "truncated_career_recommender_dataset.csv"

# Initialize the Gradient
gradient = Gradient()

# Loading the dataset
formatted_data = []
with open(carrer_dataset_path, encoding='utf-8-sig') as f:
    dataset_data = csv.DictReader(f, delimiter=",")
    for row in dataset_data:
        # user_data = f"Interests: {row['Interests']}, Skills: {row['Skills']}, Degree: {row['Undergraduate Course']}, Working: {row['Employment Status']}"
        user_data = f"Interests: {row['Interests']}, Skills: {row['Skills']}, Degree: {row['Undergraduate Course']}, Working: {row['Employment Status']}, Specialization: {row['UG Specialization']}, Percentage: {row['UG CGPA/Percentage']}, Certifications: {row['Certifications']}"
        carrer_response = row['Career Path']
        formatted_entry = {
            "inputs": f"### User Data:\n{user_data}\n\n### Suggested Carrer Path:",
            "response": carrer_response
        }
        formatted_data.append(formatted_entry)

# Getting the base model from Gradient
base = gradient.get_base_model(base_model_slug="nous-hermes2")
new_model_adapter = base.create_model_adapter(name="ai_carrer_chatbot")

# Fine-tuning the model adapter in chunks to prevent memory issues
chunck_lines = 20
total_chunks = [formatted_data[x:x + chunck_lines] for x in range(0, len(formatted_data), chunck_lines)]
for i, chunck in enumerate(total_chunks):
    try:
        print(f"Fine-tuning chunk {i + 1} of {len(total_chunks)}")
        new_model_adapter.fine_tune(samples=chunck)
    except Exception as error:
        print(f"Error in fine-tuning chunk {i + 1}: {error}")

# @app.route('/')
# def main_page():
#     return render_template('main_page.html')

@app.route('/',methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # user_query = request.form['user_query']
        interests = request.form['interests']
        skills = request.form['skills']
        degree = request.form['degree']
        working = request.form['working']
        specialization = request.form['specialization']
        percentage = request.form['percentage']
        certifications = request.form['certifications']

        user_query = f"Interests: {interests}, Skills: {skills}, Degree: {degree}, Working: {working}, Specialization: {specialization}, Percentage: {percentage}, Certifications: {certifications}"
        formatted_query = f"### User Data:\n{user_query}\n\n### Suggested Carrer Path:"
        response = new_model_adapter.complete(query=formatted_query, max_generated_token_count=50)
        generated_output = response.generated_output
        return render_template('home.html', user_query=user_query, generated_output=generated_output)

    return render_template('home.html', user_query=None, generated_output=None)


if __name__ == '__main__':
    app.run(debug=True)