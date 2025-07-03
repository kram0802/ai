import streamlit as st
from PIL import Image
import numpy as np
import cv2
try:
    import torch
except ModuleNotFoundError:
    raise ImportError("PyTorch is required for the dashboard. Please install it with `pip install torch`.")

from data_ingest import get_frame
from ai_model import SimpleCNN
from quantum_layer import quantum_circuit
from automation import trigger_action

st.title("Hybrid AI + Quantum Dashboard")

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
