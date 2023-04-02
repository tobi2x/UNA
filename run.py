import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

model = load_model("subject1.h5")

location = input("Enter the location: ")
property_type = input("Enter the property type (e.g. house, apartment, condo): ")
year_of_construction = int(input("Enter the year of construction: "))
renovated = input("Has the property ever been renovated (yes or no)? ")
if renovated.lower() == "yes":
    renovation_year = int(input("Enter the renovation year: "))
else:
    renovation_year = np.nan

data = {'Location': [location],
        'Property Type': [property_type],
        'Year of Construction': [year_of_construction],
        'Renovated': [renovated],
        'Renovation Year': [renovation_year]}
input_df = pd.DataFrame(data)

df = pd.read_csv('dataset.csv')

input_df = input_df[df.drop(['Predicted Price', 'HVAC', 'Plumbing', 'Electrical'], axis=1).columns]

scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_df)

prediction = model.predict(input_scaled)

print(f"The predicted price is: {prediction[0][0]}")
