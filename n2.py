import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
df = pd.read_csv('dataset.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(['predicted_price'], axis=1), df[['predicted_price', 'hvac', 'plumbing', 'electrical']], test_size=0.2, random_state=42)

# Standardize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(4)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=5000, validation_split=0.2)

# Evaluate the model
test_score = model.evaluate(X_test, y_test)
print(f'Testing score: {test_score:.7f}')

# Make predictions for the recommended maintenance intervals and house prices
predictions = model.predict(X_test)

# Print the predictions
print(predictions)

# Save the model
model.save("subject1.h5")
