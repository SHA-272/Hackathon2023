from flask import Flask, render_template, request, jsonify
import predictor

app = Flask(__name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    # Retrieve input data from the form

    # Male Population
    male_population = int(request.form.get('male_population'))

    # Female Population
    female_population = int(request.form.get('female_population'))

    # Number of Unemployed Population
    number_of_unemployed_population = int(request.form.get('number_of_unemployed_population'))

    # Crime Solving Percentage
    crime_solving_percentage = float(request.form.get('crime_solving_percentage'))

    # Number of Law Enforcement Personnel
    number_of_law_enforcement_personnel = int(request.form.get('number_of_law_enforcement_personnel'))

    # Population Mortality Rate
    population_mortality_rate = float(request.form.get('population_mortality_rate'))

    # Number of Prior Crimes in the Field Of
    number_of_prior_crimes_in_the_field_of = int(request.form.get('number_of_prior_crimes_in_the_field_of'))

    # Average Labor Payment
    average_labor_payment = float(request.form.get('average_labor_payment'))

    # Inflation Rate
    inflation_rate = float(request.form.get('inflation_rate'))

    # Standard of Living
    standard_of_living = float(request.form.get('standard_of_living'))

    # Population Digitization Level
    population_digitization_level = float(request.form.get('population_digitization_level'))

    # Number of Companies in the Information Technology Sector
    number_of_companies_in_the_information_technology_sector = int(request.form.get('number_of_companies_in_the_information_technology_sector'))

    # Number of Population Educated in IT
    number_of_population_educated_in_IT = int(request.form.get('number_of_population_educated_in_IT'))

    # Number of Known Hacking Communities
    number_of_known_hacking_communities = int(request.form.get('number_of_known_hacking_communities'))

    # Number of Cybersecurity Incidents
    number_of_cybersecurity_incidents = int(request.form.get('number_of_cybersecurity_incidents'))

    # Number of Investments in Cybersecurity Sector
    number_of_investments_in_cybersecurity_sector = int(request.form.get('number_of_investments_in_cybersecurity_sector'))

    # Perform prediction using the input data
    prediction = predictor.predict(
        male_population=male_population,
        female_population=female_population,
        number_of_unemployed_population=number_of_unemployed_population,
        crime_solving_percentage=crime_solving_percentage,
        number_of_law_enforcement_personnel=number_of_law_enforcement_personnel,
        population_mortality_rate=population_mortality_rate,
        number_of_prior_crimes_in_the_field_of=number_of_prior_crimes_in_the_field_of,
        average_labor_payment=average_labor_payment,
        inflation_rate=inflation_rate,
        standard_of_living=standard_of_living,
        population_digitization_level=population_digitization_level,
        number_of_companies_in_the_information_technology_sector=number_of_companies_in_the_information_technology_sector,
        number_of_population_educated_in_IT=number_of_population_educated_in_IT,
        number_of_known_hacking_communities=number_of_known_hacking_communities,
        number_of_cybersecurity_incidents=number_of_cybersecurity_incidents,
        number_of_investments_in_cybersecurity_sector=number_of_investments_in_cybersecurity_sector
    )

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
