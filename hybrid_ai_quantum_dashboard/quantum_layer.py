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
