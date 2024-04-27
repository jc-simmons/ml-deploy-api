This repository contains a simple machine learning program (in progress) that trains a classification model on a common diabetes dataset. 
When a change is detected in the model .pkl file, a Github actions workflow is initialized that automatically:

- builds a Docker image containing the relevant ML files and a simple Flask prediction API (app.py). The image is then pushed to Dockerhub.
- calls a Render webhook that redeploys a webservice using the Docker image
- sends a curl request to the prediction API with a sample entry (test.json) in order to check a successful deployment (HTTP status code 200)





A custom cross-valiated grid search method is implemented using a wrapper around the sklearn GridSearchCV module in order to enable easier searching across different models as well as their hyperparameters. 
Combinations of machine learning models and their hyperparameters, in addition to other configuration variables, are contained in config.yaml to allow for more convenient model testing. The sklearn modules are imported 
dynamically based on selection and the best model is saved (if desired) and used for the API implementation. 
