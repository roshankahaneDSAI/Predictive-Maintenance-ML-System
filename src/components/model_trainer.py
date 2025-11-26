import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from src.exception.exception import CustomException
from src.logging.logger import logging
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

from src.utils.ml_utils.model.estimator import MachineModel
from src.utils.ml_utils.metric.classification_metric import get_classification_score
from src.utils.main_utils.utils import save_object, evaluate_models, load_object, load_numpy_array_data


class ModelTrainer:
    def __init__(self, data_transformation_artifact:DataTransformationArtifact, model_trainer_config:ModelTrainerConfig):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    def train_model(self, X_train, y_train, X_test, y_test):
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "SVC": SVC(),
            "RandomForestClassifier": RandomForestClassifier(),
            "XGBClassifier": XGBClassifier(use_label_encoder=False, objective="multi:softmax", num_class=6, random_state=42, eval_metric="mlogloss"), 
            }

        params = {
                "Logistic Regression": {
                    'C': [0.1, 1, 10],  # Regularization strength. Smaller values apply stronger regularization.

                    'solver': ['lbfgs', 'liblinear'],  # Optimization algorithms. 
                },
                
                "SVC": {
                    'C': [0.1, 1, 10],  # Regularization strength. Larger values make the decision boundary more complex, smaller values create a smoother boundary.

                    'kernel': ['linear', 'rbf'],  # Types of kernels. 
                },

                "RandomForestClassifier": {
                    'n_estimators': [50, 100],  # Number of trees in the forest. More trees generally lead to better performance but increase computation time.

                    'max_depth': [10, 20],  # Maximum depth of trees. 
                },

                "XGBClassifier": {
                    'n_estimators': [50, 100],  # Number of boosting rounds. More rounds improve performance but can lead to overfitting.

                    'learning_rate': [0.01, 0.1],  # Step size for each boosting round. 

                    'max_depth': [3, 6],  # Maximum depth of each tree. 
                }
            }

        logging.info("Entered into evaluate models")
            
        model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

        # To get best model score from dict
        best_model_score = max(sorted(model_report.values()))
        print(best_model_score)

        # To get best model name from dict
        best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

        best_model=models[best_model_name]

        logging.info("Got the best model")

        if best_model_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")
        
        y_train_pred=best_model.predict(X_train)

        classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)

        y_test_pred=best_model.predict(X_test)

        classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        Machine_Model=MachineModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=Machine_Model)

        # Model pusher
        save_object("final_model/model.pkl",best_model)

        # Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )

        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path=self.data_transformation_artifact.transformed_train_file_path
            test_file_path=self.data_transformation_artifact.transformed_test_file_path

            # Loading training array and testing array
            train_array = load_numpy_array_data(train_file_path)
            test_array=load_numpy_array_data(test_file_path)

            logging.info("Splitting train and test input data")
            X_train, y_train, X_test, y_test=(
                train_array[:, :-1], 
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            model_trainer_artifact=self.train_model(X_train,y_train,X_test,y_test)
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e,sys)
        
