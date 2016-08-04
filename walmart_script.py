import flask
app = flask.Flask(__name__)

#-------- MODEL GOES HERE -----------#
import pandas as pd 
import numpy as np
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv('walmart.csv')
df_test = df[df['Weekly_Sales'].isnull() == True]
df_train = df[df['Weekly_Sales'].isnull() != True]
X_train = df_train[['Temperature', 'Fuel_Price', 'CPI',
       'Unemployment', 'IsHoliday', 'Type', 'Size', 'Dept']]
y_train = df_train['Weekly_Sales']
X_test = df_test[['Temperature', 'Fuel_Price', 'CPI',
       'Unemployment', 'IsHoliday', 'Type', 'Size', 'Dept']]
y_test = df_test['Weekly_Sales']
gb = GradientBoostingRegressor()
PREDICTOR = gb.fit(X_train,y_train)

 
#-------- ROUTES GO HERE -----------#

@app.route('/predict', methods=["GET"])
def predict():
    Temperature = float(flask.request.args['Temperature'])
    Fuel_Price = float(flask.request.args['Fuel_Price'])
    CPI = float(flask.request.args['CPI'])
    Unemployment= float(flask.request.args['Unemployment'])
    IsHoliday = flask.request.args['IsHoliday']
    Type = flask.request.args['Type']
    Size = flask.request.args['Size']
    Dept = flask.request.args['Dept']  
    item = np.array([Temperature, Fuel_Price, CPI, Unemployment, IsHoliday, Type, Size, Dept])
    value = PREDICTOR.predict(item)[0]
    value = round(value, 2)
    results = {'weekly_sales': value}
    return flask.jsonify(results)   
      

if __name__ == '__main__':

    HOST = '127.0.0.1'
    PORT = '4000'

    app.run(HOST, PORT)