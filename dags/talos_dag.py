from airflow import DAG
from airflow.decorators import task
from datetime import datetime

from sympy.physics.units import year

from src.TALOS.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.TALOS.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.TALOS.pipeline.stage_03_data_transformation import DataTransformationPipeline
from src.TALOS.pipeline.stage_04_model_train import ModelTrainTrainingPipeline
from src.TALOS.pipeline.stage_05_model_run import ModelRunTrainingPipeline

with DAG(
    dag_id="training_pipeline_dag",
    start_date=datetime(2026, 2, 20),
    schedule_interval=None,
    catchup=False,
    tags=["talos_pipeline"],
) as dag:

    @task
    def data_ingestion():
        DataIngestionTrainingPipeline().main()

    @task
    def data_validation():
        DataValidationTrainingPipeline().main()

    @task
    def data_transformation():
        DataTransformationPipeline().main()

    @task
    def model_train():
        ModelTrainTrainingPipeline().main()

    @task
    def model_run():
        ModelRunTrainingPipeline().main()

    # Define execution order
    data_ingestion() >> data_validation() >> data_transformation() >> model_train() >> model_run()