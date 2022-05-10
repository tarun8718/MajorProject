from tokenize import Double
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Model/new_model.pkl', 'rb'))

@app.errorhandler(404)
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('404.html'), 404
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    form_vals = [float(x) for x in request.form.values()]
    #int_features=[9.54,334,116,1,4]
    print("Hello *****", form_vals)
    final_features = [np.array(form_vals)]
    prediction = model.predict(final_features)
    output = prediction[0]*100

    return render_template('index.html', prediction_text='Chance of Admition is {}%'.format(round(output[0],2)))

@app.route('/essay')
def essay():
    return render_template('index2.html')

@app.route('/score',methods=['POST'])
def score():
    '''
    For rendering results on HTML GUI
    '''
    essay = [x for x in request.form.values()]
    print("Hello *****", essay)
    final_features = [np.array(essay)]
    prediction = model.predict(final_features)
    output = prediction[0]*100

    return render_template('index2.html', prediction_text='Essay score is{}%'.format(round(output[0],2)))

if __name__ == "__main__":
    app.run()