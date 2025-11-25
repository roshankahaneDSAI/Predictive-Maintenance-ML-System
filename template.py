import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

list_of_files=[
    f"src/__init__.py",
    f"src/components/__init__.py",
    f"src/components/data_ingestion.py",
    f"src/components/data_transformation.py",
    f"src/components/data_validation.py",
    f"src/components/model_trainer.py",
    f"src/components/model_monitoring.py",
    f"src/entity/__init__.py",
    f"src/entity/config_entity.py",
    f"src/entity/artifact_entity.py",
    f"src/pipelines/__init__.py",
    f"src/pipelines/training_pipeline.py",
    f"src/pipelines/prediction_pipeline.py",   
    f"src/constant/__init__.py",
    f"src/constant/training_pipeline/__init__.py",
    f"src/exception/__init__.py",   
    f"src/exception/exception.py", 
    f"src/logging/__init__.py",
    f"src/logging/logger.py", 
    f"src/utils/main_utils/__init__.py", 
    f"src/utils/main_utils/utils.py",
    f"src/utils/ml_utils/__init__.py",
    f"src/utils/ml_utils/model/estimator.py",
    f"src/utils/ml_utils/metric/classification_metric.py",
    f"data_schema/schema.yaml",
    f"/notebook/",
    "app.py",
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py"
]

for filepath in list_of_files:
    filepath=Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"creating directory:{filedir} for the file{filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empth file: {filepath}")
    else:
        logging.info(f"{filename} already exists.")

