from src.pipelines.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    try:
        training_pipeline = TrainingPipeline()   # ✅ create object
        training_pipeline.run_pipeline()         # ✅ call method on object
    except Exception as e:
        print(e)
