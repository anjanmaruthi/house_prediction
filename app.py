from flask import Flask,render_template,request
import numpy as np
import joblib
loaded_model = joblib.load('./models/best_model')
poly = joblib.load('./models/polynomial feature')
sc = joblib.load('./models/scaler')

app = Flask(__name__)

@app.route('/',methods = ['GET'])

def home():
    return render_template ('index.html')

@app.route('/predict',methods = ['GET'])

def predict():
    return render_template ('home.html')

@app.route('/results', methods=['GET','POST'])

def results():
    if request.method == 'POST':
        print (request.form)
        l = [request.form['house_age'],request.form['MRT_distance'],request.form['convenience_store'],request.form['latitude'],request.form['longitude']]
        arr = np.asarray([l])
        arr = poly.transform(arr)
        scaled_arr = sc.transform(arr)
        print('Price of the house per unit area: ', round(loaded_model.predict(scaled_arr)[0][0],2))
        data = {'results':round(loaded_model.predict(scaled_arr)[0][0],2),'house_age':request.form['house_age'],'MRT_distance':request.form['MRT_distance'],'convenience_store':request.form['convenience_store'],'latitude':request.form['latitude'],'longitude':request.form['longitude'],}
    return render_template ('results.html', data = data)




if __name__ == "__main__":
    app.run(port=5000,debug=True)