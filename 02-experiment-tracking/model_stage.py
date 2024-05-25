import mlflow
import click
from mlflow.tracking import MlflowClient
from datetime import datetime

EXPERIMENT_NAME = "random-forest-best-models"
mlflow.set_tracking_uri("./my_tracking_uri") # type: ignore
mlflow.set_experiment(EXPERIMENT_NAME)

@click.command()
@click.option(
    "--model_name",
    default="model",
    help="Location of tracking url"
)



def check_model_stage(model_name:str):
    client = MlflowClient()
    latest_versions = client.get_latest_versions(name=model_name)
    for version in latest_versions:
        print(f"version = {version.version}\
            stage = {version.current_stage}")
        if version.current_stage is None:
            return version.version,"Staging"
    return None, None

def change_model_stage(model_name:str,
                        version_id:int,
                        model_stage:str):
    client = MlflowClient()
    latest_versions = client.get_latest_versions(name=model_name)
    client.transition_model_version_stage(
        name=model_name,
        version=version_id,
        stage=model_stage,
        archive_existing_versions=False
    )

    date = datetime.today().date()
    client.update_model_version(
    name=model_name,
    version=version_id,
    description=f"The model version {version_id} was transitioned to {model_stage} on {date}"
    )

if __name__=='__main__':
    model_version, stage = check_model_stage()
    if model_version:
        print("chaning model stage")
        change_model_stage(model_version,stage)
        