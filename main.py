
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig
from src.entity.config_entity import(
    TrainingPipelineConfig,)
from src.constant import training_pipeline
from src.entity.artifact_entity import DataIngestionArtifact
from src.pipelines.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    try:
        training_pipeline = TrainingPipeline()   # ✅ create object
        training_pipeline.run_pipeline()         # ✅ call method on object
    except Exception as e:
        print(e)
