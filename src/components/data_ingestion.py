import os
import sys
import pandas as pd
import numpy as np
import pymongo
from src.exception.exception import CustomException
from src.logging.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split

# from src.components.data_transformation import DataTransformation

# from src.components.model_trainer import ModelTrainer

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)
        
    def export_collection_as_dataframe(self):
        """
        Read data from mongodb
        """
        try:
            database_name=self.data_ingestion_config.database_name
            collection_name=self.data_ingestion_config.collection_name
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            collection=self.mongo_client[database_name][collection_name]

            df=pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df=df.drop(columns=["_id"], axis=1)

            df.replace({'na': np.nan}, inplace=True)
            return df

        except Exception as e:
            raise CustomException(e, sys)
        
    def export_data_into_feature_store(self,dataframe: pd.DataFrame):
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            # Creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)

            # Save raw data
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            logging.info("Raw data saved successfully")

            return dataframe
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            # Prepairing X and Y
            X = dataframe.drop(columns=['UDI','Product ID', 'Target', 'Failure Type'], axis=1)
            y = dataframe['Failure Type']
            logging.info("Prepared feature dataframe X and label series y")

            # Combine X and y into a single DataFrame for splitting
            data = X.copy()
            data['Failure Type'] = y
            print(data)

            logging.info("Train/Test Split Initiated")
            train_set, test_set=train_test_split(data, test_size=self.data_ingestion_config.train_test_split_ratio,random_state=42)
            print(train_set)
            print(test_set)

            logging.info("Performed train test split on the dataframe")

            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            
            os.makedirs(dir_path, exist_ok=True)

            # Replace special characters in column names 
            train_set.columns = train_set.columns.str.replace(r'[^\w\s]', '', regex=True)
            test_set.columns = test_set.columns.str.replace(r'[^\w\s]', '', regex=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)

            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info(f"Exported train and test file path.")

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            dataframe=self.export_collection_as_dataframe()
            print(dataframe.shape)

            dataframe=self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            dataingestionartifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path, test_file_path=self.data_ingestion_config.testing_file_path)
            return dataingestionartifact
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    # data_transformation=DataTransformation()
    # train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    # modeltrainer=ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
    pass



