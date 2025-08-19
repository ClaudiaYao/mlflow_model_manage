The repo contains a sample MLFlow project. It uses DataBricks Free Edition, and some functionalities could not work correctly

Implemented functionalities:

1. Upload the initial dataset to DataBricks Unity Catalog.
2. Load existing data from Unity Catalog table.
3. Train/Evaluate the model locally.
4. Log and register the model and its metrics/parameters to Unity Catalog.
5. Create a few unit test scripts.
6. (Partially) Deploy the project to DataBricks.<br>
   Encounter some issues. TBD...

Steps to run the project:

1. Git clone.<br>
2. Install Python 3.11<br>
3. Install Databricks CLI (new Go version):<br>

```bash
brew tap databricks/tap
brew install databricks

```

4. Activate virtual environment by `source .venv/bin/activate`
5. add .env file under the project root path. It should include those fields:<br>

Catalog_name, schema_name, databricks_host and databricks_token needs to be set on DataBricks UI

```bash
CATALOG_NAME=
SCHEMA_NAME=
DATABRICKS_HOST=
DATABRICKS_TOKEN=
TABLE_NAME=teen_phone_addiction
MODEL_NAME=teen_phone_addiction
ARTIFACT_NAME=teen_phone_addiction_predictor
ENDPOINT_NAME=teen-phone-addiction
EXPERIMENT_NAME=/Users/<your-databricks-account-email>/teen_phone_addiction_experiment
```

6. Set current working directory to the project root folder.
7. Run `python3 src/data/upload_initial_data.py`
8. Run `python3 scripts/train_pipeline.py`
