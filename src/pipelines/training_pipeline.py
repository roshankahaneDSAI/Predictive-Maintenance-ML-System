import os
import sys

from src.exception.exception import CustomException
from src.logging.logger import logging

from src.components.data_ingestion import DataIngestion

from src.entity.config_entity import(
    TrainingPipelineConfig,
    DataIngestionConfig,
)

from src.entity.artifact_entity import (
    DataIngestionArtifact,
)

# from src.constant.training_pipeline import TRAINING_BUCKET_NAME
# from src.cloud.S3_syncer import S3Sync

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config=TrainingPipelineConfig()
        # self.s3_sync = S3Sync()
        

    def start_data_ingestion(self):
        try:
            data_ingestion_config=DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start data Ingestion")
            data_ingestion=DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
    
    def run_pipeline(self):
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            pass    
            # return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
    
