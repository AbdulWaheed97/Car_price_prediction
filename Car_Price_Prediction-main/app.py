import  numpy as np
from  flask import Flask,request,jsonify,render_template
import pickle
import pandas as pd
#------------------------------------------------------------------------------------------------------------------
app =   Flask(__name__)
model = pickle.load(open('testmodel1.pkl','rb'))
#------------------------------------------------------------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')
#------------------------------------------------------------------------------------------------------------------
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = np.array(int_features)
    
    df = pd.DataFrame(data=final_features.reshape(-1, len(final_features)))
    df.columns = ['Kms_driven', 'Power', 'Transmission', 'Fuel_type', 'Vendor','Location','Drive_type', 'features_score', 'age']

#------------------------------------------------------------------------------------------------------------------    
    prediction = model.predict(df)
    
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Predicted Price $ {}'.format(output))
#------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)