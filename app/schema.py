# defines what kind of data your API accepts (data validation).
# This is especially useful for ensuring that incoming requests have the correct structure and types.

from pydantic import BaseModel, confloat, conint

class PatientData(BaseModel):
    Pregnancies: conint(ge=0) # ge=0 means greater than or equal to 0
    Glucose: confloat(ge=0) # non-negative floats
    BloodPressure: confloat(ge=0)
    SkinThickness: confloat(ge=0)
    Insulin: confloat(ge=0)
    BMI: confloat(ge=0)
    DiabetesPedigreeFunction: confloat(ge=0)
    Age: conint(ge=0)

