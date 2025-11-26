import os
import sys
import pandas as pd
from flask import request
from src.exception.exception import CustomException
from src.utils.main_utils.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("final_model","model.pkl")
            preprocessor_path=os.path.join('final_model','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
        type: str, 
        air_temperature_k: float,
        process_temperature_k: float, 
        rotational_speed_rpm: int,
        torque_nm: float, 
        tool_wear_min: int):

        print("Received values:")
        print(f"type: {type}")
        print(f"air_temperature_k: {air_temperature_k}")
        print(f"process_temperature_k: {process_temperature_k}")
        print(f"rotational_speed_rpm: {rotational_speed_rpm}")
        print(f"torque_nm: {torque_nm}")
        print(f"tool_wear_min: {tool_wear_min}")
    
        self.type=type
        self.air_temperature_k = air_temperature_k
        self.process_temperature_k = process_temperature_k
        self.rotational_speed_rpm = rotational_speed_rpm
        self.torque_nm = torque_nm
        self.tool_wear_min = tool_wear_min

    def get_data_as_data_frame(self):
        try:
            # Print raw request data
            print("Request Form Data:", request.form)

            custom_data_input_dict={
                "Type": [self.type],
                "Air temperature K": [self.air_temperature_k],
                "Process temperature K": [self.process_temperature_k],
                "Rotational speed rpm": [self.rotational_speed_rpm],
                "Torque Nm": [self.torque_nm],
                "Tool wear min": [self.tool_wear_min],
            }
            df=pd.DataFrame(custom_data_input_dict)
            print("Final DataFrame:\n", df)

            return df

        except Exception as e:
            raise CustomException(e, sys)
       