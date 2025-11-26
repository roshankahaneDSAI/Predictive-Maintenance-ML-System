from flask import Flask, request, render_template, Response
import numpy as np
import pandas as pd
import pymongo
import os, sys
import certifi
ca=certifi.where()

from dotenv import load_dotenv
load_dotenv()

from src.exception.exception import CustomException
from src.logging.logger import logging

from src.pipelines.prediction_pipeline import CustomData, PredictPipeline
from src.pipelines.training_pipeline import TrainingPipeline

# We can use this if we want to apply the ETL pipeline here

# mongo_db_url=os.getenv("MONGO_DB_URL")
# print(mongo_db_url)
# client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

# from src.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
# from src.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME

# database = client[DATA_INGESTION_DATABASE_NAME]
# collection = database[DATA_INGESTION_COLLECTION_NAME]

application=Flask(__name__)

app=application

# Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/train", methods=["POST"])
def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline() 
        logging.info("Training pipeline executed successfully.")
        return "Training is successful"  
    except Exception as e:
        raise CustomException(str(e), sys)

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else: 
        data=CustomData(
            type=request.form.get('type'),
            air_temperature_k = request.form.get('air_temperature_k'),
            process_temperature_k = request.form.get('process_temperature_k'),
            rotational_speed_rpm = request.form.get('rotational_speed_rpm'),
            torque_nm = request.form.get('torque_nm'),
            tool_wear_min = request.form.get('tool_wear_min'),
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")


        predict_pipeline=PredictPipeline()
        print("Mid Prediction")

        results=predict_pipeline.predict(pred_df)
        print("after Prediction")

        return render_template('index.html', results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0", port= 8080, debug= True)

