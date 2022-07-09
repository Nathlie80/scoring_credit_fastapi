# scoring-credit-fastapi

Implementation of a credit scoring API

Files:

app.py: contains the code to give:
			the probility for the client to be classified in risky or not,
			the classification prediction.

lgbm_model.joblib: contains the .best_estimator_ to be applied in the app

requirements: contains all library needed in the app code

sample_norm: sample with 10% of the data prepared to be applied in the model

Procfile: specifies the commands that are executed by the app
