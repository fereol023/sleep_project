import joblib
import numpy as np
import pandas as pd
import uvicorn
from dstoolbox.pipeline import DataFrameFeatureUnion
from sklearn.pipeline import Pipeline
from fastapi import FastAPI
from _sleeproj.Encoders import OneHotEncoderPandas, OrdinalEncoderPandas, FloatEncoder, IntEncoder

# import pydantic
# import uvicorn
# from fastapi import FastAPI   # requiert internet : add /docs to localhost url the swaggerUI
# add /redoc to see the automatic documentation

"""
Gender                     object
Age                         int64
Occupation                 object
Physical Activity Level     int64
Stress Level                int64
BMI Category               object
High_pressure               int64
Low_pressure                int64
Heart Rate                  int64
Daily Steps                 int64
Sleep Disorder             object
"""

app = FastAPI(debug=True)  # instance de la classe / initialize


@app.get('/')
def home():
    return {'text': 'Sleep duration time Prediction'}


@app.get('/predict sleep duration')
def predict_sleep_duration(gender: object, age: int, occupation: object, physical_act_level: int, stress_lev: int,
                           bmi_cat: object,
                           high_pressure: int, low_pressure: int, heart_rate: int, daily_steps: int,
                           sleep_disorder: object):
    
    new_data = np.reshape([
        gender, age, occupation, physical_act_level, stress_lev, bmi_cat,
        high_pressure, low_pressure, heart_rate, daily_steps, sleep_disorder
    ], (1, -1))
    print("Enrty : ", new_data)
    print("Preprocessing...")
    
    def preprocess_it(data: pd.DataFrame, 
                      to_ohe = ['Gender', 'Sleep Disorder'], 
                      to_orde = ['BMI Category'], 
                      to_cast_in_float = ['Age', 'Physical Activity Level', 'Stress Level', 'High_pressure', 'Low_pressure'], 
                      to_cast_in_int = ['Heart Rate', 'Daily Steps'])-> pd.DataFrame:
        """
        Fonction qui se base sur l'instance du pipeline de preprocessing pour pretraiter les données.
        input : dataframe pandas à 11 colonnes 
            Gender                     object
            Age                         int64
            Occupation                 object
            Physical Activity Level     int64
            Stress Level                int64
            BMI Category               object
            High_pressure               int64
            Low_pressure                int64
            Heart Rate                  int64
            Daily Steps                 int64
            Sleep Disorder             object
        output : dataframe pandas à 13 colonnes
        """   
        data = pd.DataFrame(data, 
                            columns=['Gender', 'Age', 'Occupation', 
                                     'Physical Activity Level', 'Stress Level', 
                                     'BMI Category', 'High_pressure', 'Low_pressure', 
                                     'Heart Rate', 'Daily Steps', 'Sleep Disorder'])
        feature_pipeline = DataFrameFeatureUnion([
            ('OneHotEncoding', OneHotEncoderPandas(columns=to_ohe)),
            ('OrdinalEncoding', OrdinalEncoderPandas(columns=to_orde)),
            ('FloatEnconding', FloatEncoder(columns_to_transform=to_cast_in_float)),
            ('IntEnconding', IntEncoder(columns_to_transform=to_cast_in_int))
        ], verbose=True)

        # instance et exécution du pipeline
        prepPipeline = Pipeline([('prepare', feature_pipeline)])
        data = prepPipeline.fit_transform(data)
        print(data)
        print("feature pipeline ok")
        print(data.shape, end='\n\n')

        # ohe encoding : affecter des 0 au reste des categories après passage dans le pipeline 
        # dans le cas où les donnees de prod ne sont pas exhaustives au niveau des categories
        known_categ_gender = ['Gender_Male', 'Gender_Female']
        known_categ_sleep_disorder = ['Sleep Disorder_None', 'Sleep Disorder_Insomnia', 'Sleep Disorder_Sleep Apnea']
        # Gender
        #print(data.columns.intersection(known_categ_gender))
        data_gender = data[data.columns.intersection(known_categ_gender)]
        data_gender_exhaust = pd.get_dummies(pd.Categorical(data_gender, categories=known_categ_gender))
        # Sleep Disorder
        data_sleep_disorder = data[data.columns.intersection(known_categ_sleep_disorder)]
        data_sleep_disorder_exhaust = pd.get_dummies(pd.Categorical(data_sleep_disorder, categories=known_categ_sleep_disorder))
        print("Gender & Sleep disorder post treatment ok", end='\n\n')
        # other columns
        data_others = data[data.columns.intersection(to_orde+to_cast_in_int+to_cast_in_float)]

        return pd.concat([data_gender_exhaust,
                          data_sleep_disorder_exhaust,
                          data_others], axis=1)

        
    preproc_new_data = preprocess_it(new_data)
    # Feature names must be in the same order as they were in fit.
    features_fitting_order = ['Gender_Female', 'Gender_Male', 'Sleep Disorder_Insomnia',
                              'Sleep Disorder_None', 'Sleep Disorder_Sleep Apnea', 'BMI Category', 'Age',
                              'Physical Activity Level', 'Stress Level', 'High_pressure', 'Low_pressure',
                              'Heart Rate', 'Daily Steps']
    preproc_new_data = preproc_new_data[features_fitting_order]
    assert preproc_new_data.shape == (1, 13)
    print('Preprocessing ok !')
    print(preproc_new_data)
    
    print("Estimating...")
    model = joblib.load("../data/07_model_output/best_model.joblib", 'r')
    makeprediction = model.predict(preproc_new_data)
    output = round(makeprediction[0], 2)

    return {f'Sleep duration time is : {output}'}

if __name__ == '__main__':
    uvicorn.run(app) # just run
    # uvicorn sleepydeploy_fastapi:app --reload   to reload after a modif  
    # np.reshape(array, (1, -1)) : transforme une liste en 2d array pour le model input
    # (-1, 1) transforme en 1 feature

    # LOGS + HISTO
