from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np

import pickle

# load the model from disk
loaded_model=pickle.load(open('random_forest_regression_model1.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    print(request.form)

    T=request.form["T"]
    TM=request.form["TM"]
    Tm=request.form["Tm"]
    H=request.form["H"]
    PP=request.form["PP"]
    VV=request.form["VV"]
    V=request.form["V"]
    VM=request.form["VM"]
    df=pd.DataFrame(np.array([T,TM,Tm,H,PP,VV,V,VM]))
    print("DF",df)
    #df=pd.read_csv('real_2018.csv')

    my_prediction=loaded_model.predict(np.array([T,TM,Tm,H,PP,VV,V,VM]).reshape(1,-1))
    my_prediction=my_prediction.tolist()
    print(my_prediction)
    return render_template('home.html',prediction =str(my_prediction[0]))



if __name__ == '__main__':
	app.run(debug=True)