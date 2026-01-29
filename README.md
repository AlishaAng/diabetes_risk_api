## Project overview

![Diabetes risk prediction app](assets/webpage.png)

This project is a machine learning–powered diabetes risk prediction application, consisting of:

- a FastAPI backend that uses a trained classification model 

- a Streamlit frontend that allows users to input clinical measurements and receive a risk estimate

The goal of the project is to demonstrate end-to-end data science and deployment skills, including data preprocessing, model training, API design, user interface development, and containerisation.

## Dataset
The model was trained using the Pima Indians Diabetes Dataset, publicly available on Kaggle:
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data

Key characteristics of the dataset:

- All patients are female

- All patients are at least 21 years old

- All patients are of Pima Indian heritage

- Each record contains measurements such as glucose, BMI, blood pressure, insulin, and age

- The target variable indicates whether the patient was diagnosed with diabetes

This dataset represents a very specific population and is not demographically generalisable.

## Important disclaimer

⚠️ This project is not a medical tool.

The model is not intended for clinical use.

Predictions should not be interpreted as medical advice or diagnosis.

The application is designed solely for educational and demonstration purposes.


## Running the project locally
### Prerequisites
Make sure you have the following installed:

Docker Desktop (Mac / Windows / Linux)
👉 https://www.docker.com/products/docker-desktop/

### Start the application

From the project root directory, run:
```
docker compose up --build
```
open http://localhost:8501

## Running without Docker
Terminal 1 – start the API
```
uvicorn app.main:app --reload
```
Terminal 2 – start the UI
```
streamlit run streamlit_app.py
```
open http://localhost:8501
#### API docs available at http://localhost:8000/docs