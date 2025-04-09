import numpy as np
import joblib

# Load encoders and scaler
le_highest = joblib.load("models/le_Highest_Layer.pkl")
le_transport = joblib.load("models/le_Transport_Layer.pkl")
scaler = joblib.load("models/standard_scaler.pkl") 

def preprocess_input(data):
    # Apply label encoding
    highest_encoded = le_highest.transform([data["Highest Layer"]])[0]
    transport_encoded = le_transport.transform([data["Transport Layer"]])[0]

    print("Allowed Highest Layer:", le_highest.classes_)
    print("Allowed Transport Layer:", le_transport.classes_)

    features = np.array([[
        highest_encoded,
        transport_encoded,
        data["Source Port"],
        data["Dest Port"],
        data["Packet Length"],
        data["Packets/Time"]
    ]])
    
    print("Encoded Input Features:", features)

    features_scaled = scaler.transform(features)

    print("Scaled Features:", features_scaled)

    return features_scaled
