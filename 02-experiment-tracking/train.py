import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

EXPERIMENT_NAME = 'random-forest-models'

mlflow.set_tracking_uri("./my_tracking_server")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.autolog()

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_train_pred = rf.predict(X_train)
        train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
        y_val_pred = rf.predict(X_val)
        val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
        y_test_pred = rf.predict(X_test)
        test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

if __name__ == '__main__':
    run_train()