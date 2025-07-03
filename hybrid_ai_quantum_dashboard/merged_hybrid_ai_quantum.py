# Merged Hybrid AI + Quantum Dashboard Example

# --- data_ingest.py ---
import cv2

def get_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        raise RuntimeError("Failed to read frame")

# --- ai_model.py ---
try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    raise ImportError("PyTorch is not installed. Please install it with `pip install torch`.")

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 112 * 112, 10)
        )

    def forward(self, x):
        return self.net(x)

# --- quantum_layer.py ---
import pennylane as qml
try:
    import torch
except ModuleNotFoundError:
    raise ImportError("PyTorch is required for the quantum circuit. Please install it with `pip install torch`.")

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs):
    qml.RY(inputs[0], wires=0)
    qml.RY(inputs[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(1))

# --- automation.py ---
def trigger_action(output, threshold=0.8):
    if output > threshold:
        print("\ud83d\udea8 Action Triggered: Anomaly Detected")

# --- dashboard.py (main logic) ---
import streamlit as st
from PIL import Image
import numpy as np
import cv2
try:
    import torch
except ModuleNotFoundError:
    raise ImportError("PyTorch is required for the dashboard. Please install it with `pip install torch`.")

st.title("Hybrid AI + Quantum Dashboard (Merged)")

model = SimpleCNN()

frame = get_frame()
resized = cv2.resize(frame, (224, 224))
tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255

ai_output = model(tensor)
quantum_input = torch.tensor([0.5, 0.6], requires_grad=True)
quantum_output = quantum_circuit(quantum_input)

st.image(frame, channels="BGR", caption="Live Camera Feed")
st.write(f"AI Output (Raw): {ai_output}")
st.write(f"Quantum Output: {quantum_output.item()}")

trigger_action(quantum_output.item())
