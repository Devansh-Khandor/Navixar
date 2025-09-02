# Import necessary libraries
from flask import Flask, render_template, request
from flask import redirect
import pickle
import numpy as np

# Load the Flask app
app = Flask(__name__)

# Load the machine learning model and dataframe
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

# Extract unique values for Company, Type, CPU, and GPU
brands = sorted(df['Company'].unique())
types = sorted(df['TypeName'].unique())
cpus = sorted(df['Cpu brand'].unique())
gpus = sorted(df['Gpu brand'].unique())
ops=sorted(df['os'].unique())

# Define route for the home page
@app.route('/')
def home():
    return render_template('home.html')

@app.route("/AboutUs_DK.html")
def aboutus_redirect():
    return redirect("https://www.devansh-khandor.in/", code=302)

@app.route('/AboutAK')
def AboutUs_AK():
    return render_template('AboutUs_Kulkarni.html') 

@app.route('/AboutKM')
def AboutUs_KM():
    return render_template('AboutUs_Kashvi.html')     

@app.route('/ContactUs')
def ContactUs():
    return render_template('ContactUs.html')        

# Define route for handling form submission
@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # Retrieve data from the form
        company = request.form['company']
        type = request.form['type']
        ram = int(request.form['ram'])
        weight = float(request.form['weight'])
        touchscreen = 1 if request.form['touchscreen'] == 'Yes' else 0
        ips = 1 if request.form['ips'] == 'Yes' else 0
        screen_size = float(request.form['screen_size'])
        resolution = request.form['resolution']
        cpu = request.form['cpu']
        hdd = int(request.form['hdd'])
        ssd = int(request.form['ssd'])
        gpu = request.form['gpu']
        os = request.form['os']
        age = request.form['age']
        scratches = request.form['scratches']
        battery_life = request.form['battery_life']
        heating_issues = request.form['heating_issues']
        charger = request.form['charger']
        repair_history = request.form['repair_history']
        under_warranty = request.form['under_warranty']
        color_peel_off = request.form['color_peel_off']
        keypad_functionality = request.form['keypad_functionality']
        broken_parts = request.form['broken_parts']
        ram_size = request.form['ram_size']
        rom_size = request.form['rom_size']

        # Calculate PPI
        # X_res = int(resolution.split('x')[0])
        # Y_res = int(resolution.split('x')[1])
        # ppi = ((X_res*2) + (Y_res*2))*0.5/screen_size
        x, y = map(int, resolution.lower().split('x'))
        ppi = np.hypot(x, y) / screen_size
        ppi = round(ppi, 1)


        # Prepare query array
        query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

        query = query.reshape(1,12)
        initial_price = int(np.exp(pipe.predict(query)[0]))

        # Prepare additional features for depreciation
        features = {
            'age': age,
            'scratches': scratches,
            'battery_life': battery_life,
            'heating_issues': heating_issues,
            'charger': charger,
            'repair_history': repair_history,
            'under_warranty': under_warranty,
            'color_peel_off': color_peel_off,
            'keypad_functionality': keypad_functionality,
            'broken_parts': broken_parts,
            'ram': ram_size,
            'rom': rom_size
        }

        # Calculate depreciated price
        depreciated_price = calculate_depreciation(initial_price, features)

        # Render the result template with the predicted prices
        return render_template('result.html', initial_price=initial_price, depreciated_price=depreciated_price)
    else:
        return render_template('predict.html', brands=brands, types=types, cpus=cpus, gpus=gpus, ops=ops)
# Function to calculate depreciation
def calculate_depreciation(actual_value, features):
    depreciation_percentage = 0

    if features['age'] == '0-1 year':
        depreciation_percentage += 5
    elif features['age'] == '1-2 years':
        depreciation_percentage += 10
    elif features['age'] == '2-3 years':
        depreciation_percentage += 15
    else:
        depreciation_percentage += 20

    if features['scratches'] == 'Minor':
        depreciation_percentage += 5
    elif features['scratches'] == 'Moderate':
        depreciation_percentage += 10
    elif features['scratches'] == 'Major':
        depreciation_percentage += 15

    if features['battery_life'] == '4 hours or less':
        depreciation_percentage += 10
    elif features['battery_life'] == '4-6 hours':
        depreciation_percentage += 5

    if features['heating_issues'] == 'Yes':
        depreciation_percentage += 5

    if features['charger'] == 'Fake':
        depreciation_percentage += 10

    if features['repair_history'] == 'Yes':
        depreciation_percentage += 15

    if features['under_warranty'] == 'No':
        depreciation_percentage += 10

    if features['color_peel_off'] == 'Yes':
        depreciation_percentage += 5

    if features['keypad_functionality'] == 'Not working':
        depreciation_percentage += 10

    if features['broken_parts'] == 'Yes':
        depreciation_percentage += 15

    if features['ram'] == 'Less than 8GB' or features['rom'] == 'Less than 256GB':
        depreciation_percentage += 10

    depreciation_amount = (depreciation_percentage / 100) * actual_value
    predicted_value = actual_value - depreciation_amount
   
    return predicted_value


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)