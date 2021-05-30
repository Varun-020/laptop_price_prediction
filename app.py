import flask
from flask import Flask
import pickle
from flask import render_template, request
import numpy as np
app = Flask(__name__)

with open('lap.pkl', 'rb') as f:
    model = pickle.load(f)
@app.route("/")
def home():
   return render_template('index.html')
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':
        f1 = request.form['f1']
        f2 = request.form['f2']
        f3 = request.form['f3']
        f4 = request.form['f4']
        f5 = request.form['f5']
        f6 = request.form['f6']
        f7 = request.form['f7']
        f8 = request.form['f8']
        f9 = request.form['f9']
        f10 = request.form['f10']
        f11 = request.form['f11']
        f12 = request.form['f12']
        feature_array = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12]
        feature = np.asarray(feature_array, dtype='float64').reshape(1,-1)
        #feature = np.array(feature_array).reshape(1,-1)
        #feature_array = request.get_json()['feature_array[]']
        prediction = float(model.predict(feature)) * 88
        return render_template('index.html', prediction='Prediction {}'.format(prediction))
if __name__ == '__main__':
    app.run(debug=True)