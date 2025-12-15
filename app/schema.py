# defines what kind of data your API accepts (data validation).
# This is especially useful for ensuring that incoming requests have the correct structure and types.

from pydantic import BaseModel, Field, confloat, conint

class PatientData(BaseModel):
    Pregnancies: conint(ge=0) = Field(
        ..., description="Number of times pregnant", example=2
    )
    Glucose: confloat(ge=0) = Field(
        ..., description="Plasma glucose concentration", example=120
    )
    BloodPressure: confloat(ge=0) = Field(
        ..., description="Diastolic blood pressure (mm Hg)", example=70
    )
    SkinThickness: confloat(ge=0) = Field(
        ..., description="Triceps skinfold thickness (mm)", example=25
    )
    Insulin: confloat(ge=0) = Field(
        ..., description="2-Hour serum insulin (mu U/ml)", example=80
    )
    BMI: confloat(ge=0) = Field(
        ..., description="Body mass index", example=28.5
    )
    DiabetesPedigreeFunction: confloat(ge=0) = Field(
        ..., description="Diabetes pedigree function", example=0.5
    )
    Age: conint(ge=0) = Field(
        ..., description="Age in years", example=35
    )
