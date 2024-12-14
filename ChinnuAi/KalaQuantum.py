import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from termcolor import colored
import pickle
import json

# QuantumState Class
class QuantumState:
    def __init__(self, state_vector):
        self.state_vector = np.array(state_vector, dtype=complex)
        self.normalize()

    def normalize(self):
        norm = np.linalg.norm(self.state_vector)
        if norm == 0:
            raise ValueError(colored("State vector cannot be zero.", "red"))
        self.state_vector /= norm

    def measure(self):
        probabilities = np.abs(self.state_vector) ** 2
        outcome = np.random.choice(len(self.state_vector), p=probabilities)
        new_state = np.zeros_like(self.state_vector)
        new_state[outcome] = 1
        self.state_vector = new_state
        return outcome

    def apply_gate(self, gate):
        if gate.shape[0] != gate.shape[1] or gate.shape[0] != len(self.state_vector):
            raise ValueError(colored("Gate dimensions must match the quantum state vector size.", "red"))
        self.state_vector = np.dot(gate, self.state_vector)

    def compose_gates(self, *gates):
        combined_gate = np.eye(len(self.state_vector))
        for gate in gates:
            if gate.shape[0] != combined_gate.shape[0]:
                raise ValueError(colored("All gates must have dimensions matching the quantum state vector.", "red"))
            combined_gate = np.dot(gate, combined_gate)
        self.apply_gate(combined_gate)

    def convert_to_qubit(self, classical_data):
        if isinstance(classical_data[0], str):
            unique_strings = list(set(classical_data))
            mapping = {string: idx for idx, string in enumerate(unique_strings)}
            num_states = len(unique_strings)
            state_vector = np.ones(num_states, dtype=complex)  # Initialize non-zero vector
            self.state_vector = state_vector / np.linalg.norm(state_vector)
        elif isinstance(classical_data[0], (int, float)):
            if not np.isclose(sum(classical_data), 1):
                raise ValueError(colored("Probabilities must sum to 1.", "red"))
            self.state_vector = np.sqrt(np.array(classical_data, dtype=complex))
        else:
            raise ValueError(colored("Unsupported data type. Provide strings or numerical probabilities.", "red"))

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.state_vector, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            state_vector = pickle.load(f)
        return QuantumState(state_vector)

    def __repr__(self):
        return f"QuantumState({self.state_vector})"

# QuantumTrainer Class
class QuantumTrainer:
    def __init__(self, model, training_data):
        self.model = model
        self.training_data = training_data
        self.loss_history = []

    def align_target_state(self, target_state):
        current_size = len(self.model.state_vector)
        target_size = len(target_state)
        if target_size < current_size:
            target_state = np.pad(target_state, (0, current_size - target_size), mode='constant')
        elif target_size > current_size:
            target_state = target_state[:current_size]
        return target_state

    def compute_loss(self, target_state):
        target_state = self.align_target_state(target_state)
        current_state = self.model.state_vector
        loss = np.linalg.norm(current_state - target_state)
        return loss

    def train(self, target_state, epochs=100, learning_rate=0.1):
        target_state = self.align_target_state(target_state)
        for epoch in tqdm(range(epochs), desc=colored("Training Progress", "green"), colour="green"):
            loss = self.compute_loss(target_state)
            self.loss_history.append(loss)
            perturbation = (target_state - self.model.state_vector) * learning_rate
            self.model.state_vector += perturbation
            self.model.normalize()

# Define a simple neural network using PyTorch
class KalaAI(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(KalaAI, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize Kala AI model
def initialize_chinnu_ai(input_size, hidden_size, output_size):
    model = KalaAI(input_size, hidden_size, output_size)
    return model

# Define training function
def train_chinnu_ai(model, quantum_trainer, target_state, dataloader, epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epochs), desc=colored("Training Kala AI", "blue"), colour="blue"):
        epoch_loss = 0.0
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.float(), labels.float()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        quantum_trainer.train(target_state, epochs=1, learning_rate=learning_rate)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

# Chat function for Kala AI
def chat_with_chinnu_ai(model, quantum_model, json_input):
    """
    Takes JSON data, processes it through Kala AI, and responds based on quantum measurements.
    Args:
        model: Trained Kala AI model.
        quantum_model: Trained QuantumState model.
        json_input: JSON data containing input for the chatbot.
    Returns:
        str: Chatbot response.
    """
    try:
        # Parse JSON input
        data = json.loads(json_input)
        input_data = np.array(data.get("input", []))
        responses = data.get("responses", [])

        # Validate input
        if input_data.size == 0 or not responses:
            return "Invalid input: JSON must contain an 'input' array and 'responses' list."

        # Convert input to tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float)

        # Process through Kala AI model
        output = model(input_tensor)

        # Measure quantum state to determine response index
        response_index = quantum_model.measure()

        # Return response based on measured index
        return responses[response_index % len(responses)]

    except json.JSONDecodeError:
        return "Invalid JSON format. Please provide a valid JSON string."

# Example usage
def main():
    initial_state = [1, 0, 0, 0]
    target_state = [0.5, 0.5, 0.5, 0.5]
    quantum_model = QuantumState(initial_state)
    quantum_trainer = QuantumTrainer(quantum_model, training_data=None)

    input_size = 4
    hidden_size = 8
    output_size = 4
    model = initialize_chinnu_ai(input_size, hidden_size, output_size)

    dataset = [
        (torch.tensor([1, 0, 0, 0]), torch.tensor([0.5, 0.5, 0.5, 0.5])),
        (torch.tensor([0, 1, 0, 0]), torch.tensor([0.5, 0.5, 0.5, 0.5])),
    ]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    epochs = 200
    learning_rate = 0.01
    train_chinnu_ai(model, quantum_trainer, target_state, dataloader, epochs, learning_rate)

    # Example JSON input
    json_input = '{"input": [1, 0, 0, 0], "responses": ["Welcome!", "How can I help you?", "Here is your data.", "Goodbye!"]}'
    response = chat_with_chinnu_ai(model, quantum_model, json_input)
    print("Kala AI response:", response)
    

if __name__ == "__main__":
    main()

