import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from turbulenceModel.utils.Loader import get_combined_dataframe


class SimpleNN(nn.Module):
    def __init__(self, n0, n1, n2):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(n0, n1)
        self.fc2 = nn.Linear(n1, n2)

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = torch.relu(z1)  # ReLU activation
        output = self.fc2(a1)
        return output


# Function to calculate and collect eigenvalues of weight matrices
def collect_eigenvalues(model):
    eigenvalues = []

    for name, param in model.named_parameters():
        if 'weight' in name:  # Only consider the weight matrices
            weights = param.data
            _, s, _ = torch.svd(weights)  # Compute eigenvalues using torch.svd
            eigenvalues.append(s.cpu().numpy())

    # Flatten the list of eigenvalues
    eigenvalues = np.concatenate(eigenvalues)
    return eigenvalues


# Function to compute loss surface
def compute_loss_surface(model, X, y, criterion, perturb_scale=0.1, grid_size=21):
    original_params = [param.data.clone() for param in model.parameters()]
    losses = np.zeros((grid_size, grid_size))
    perturbations = np.linspace(-perturb_scale, perturb_scale, grid_size)

    for i, alpha in enumerate(perturbations):
        for j, beta in enumerate(perturbations):
            for k, param in enumerate(model.parameters()):
                if param.requires_grad:
                    param.data = original_params[k] + alpha * torch.randn_like(param) + beta * torch.randn_like(param)

            with torch.no_grad():
                predictions = model(X)
                loss = criterion(predictions, y).item()
            losses[i, j] = loss

    for k, param in enumerate(model.parameters()):
        param.data = original_params[k]  # Restore original parameters

    return perturbations, losses

def initialize_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight)  # Kaiming Uniform initialization
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)  # Initialize biases to zero


# Function to plot loss surface
def plot_loss_surface(perturbations, losses):
    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid(perturbations, perturbations)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, losses, cmap='viridis')
    ax.set_xlabel('Perturbation Alpha')
    ax.set_ylabel('Perturbation Beta')
    ax.set_zlabel('Loss')
    plt.title('Loss Surface')
    plt.show()

def compute_hessian(model, X, y, criterion):
    model.eval()
    predictions = model(X)
    loss = criterion(predictions, y)

    params = [p for p in model.parameters() if p.requires_grad and p.ndimension() > 1]
    num_params = sum(p.numel() for p in params)

    hessian = torch.zeros((num_params, num_params), device=loss.device)

    grad1 = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    grad1 = torch.cat([g.view(-1) for g in grad1])

    for i in range(num_params):
        grad2 = torch.autograd.grad(grad1[i], params, retain_graph=True)
        grad2 = torch.cat([g.view(-1) if g is not None else torch.zeros_like(p).view(-1)
                           for g, p in zip(grad2, params)])
        hessian[i] = grad2

    # Regularize the Hessian
    hessian += torch.eye(hessian.size(0), device=hessian.device) * 1e-6
    return hessian


# Function to compare spectral density with Wigner semicircle
def plot_wigner_comparison(eigenvalues, n):
    # Normalize eigenvalues by sqrt(n)
    normalized_eigenvalues = eigenvalues / np.sqrt(n)

    # Plot histogram of normalized eigenvalues
    plt.figure(figsize=(8, 6))
    sns.histplot(normalized_eigenvalues, kde=False, bins=50, stat="density", color="blue", label="Eigenvalue Spectrum")

    # Add labels and title
    plt.xlabel('Normalized Eigenvalues')
    plt.ylabel('Density')
    return plt


def run_model(num_trials=1):  # Number of trials
    # Example data

    # data = get_combined_dataframe()

    features = ['drone1_x', 'drone1_y', 'drone1_z', 'drone1_roll', 'drone1_pitch', 'drone1_yaw', 'drone1_thrust',
                 'drone2_x', 'drone2_y', 'drone2_z', 'drone2_roll', 'drone2_pitch', 'drone2_yaw', 'drone2_thrust']
    target = 'downwash_force_at_pos_drone1'
    # columns = features + [target]

    data = np.random.randn(200, 200)

    # Convert the numpy array to a pandas DataFrame
    df = pd.DataFrame(data)

    # X = data[features].values
    # y = data[target].values
    #
    # X_tensor = torch.tensor(X, dtype=torch.float32)
    # y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    #
    # X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # Define features and target (assuming the last column is the target)
    X = df.iloc[:, :-1].values  # All columns except the last one
    y = df.iloc[:, -1].values  # Last column as the target

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)


    n0 = X_train.shape[1]
    n1 = 50
    n2 = 1

    criterion = nn.MSELoss()

    # Dictionary to store eigenvalues for each epoch
    eigenvalues_at_epochs = {1: [], 500: [], 1000: []}

    for trial in range(num_trials):
        model = SimpleNN(n0, n1, n2)
        model.apply(initialize_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 1000

        for epoch in range(1, num_epochs + 1):
            model.train()
            optimizer.zero_grad()
            predictions = model(X_train)
            loss = criterion(predictions, y_train)
            loss.backward()
            optimizer.step()

            # Compute Hessian eigenvalues at specific epochs
            if epoch in eigenvalues_at_epochs:
                hessian = compute_hessian(model, X_train, y_train, criterion)
                eigenvalues_hessian = torch.linalg.eigvalsh(hessian).cpu().numpy()
                eigenvalues_at_epochs[epoch].extend(eigenvalues_hessian)  # Collect eigenvalues

        if (trial + 1) % 100 == 0:
            print(f"Completed trial {trial + 1}/{num_trials}")

    # Randomly sample eigenvalues and plot Wigner comparison for each epoch
    for epoch, eigenvalues in eigenvalues_at_epochs.items():
        if len(eigenvalues) > 0:
            print(f"Sampling eigenvalues for epoch {epoch}...")
            # Randomly sample up to 5000 eigenvalues
            sampled_eigenvalues = np.random.choice(eigenvalues, size=min(len(eigenvalues), 5000), replace=False)

            # Normalize the eigenvalues to match Wigner's law
            num_params = sum(p.numel() for p in model.parameters())  # Total number of parameters in the model
            plt = plot_wigner_comparison(sampled_eigenvalues, num_params)
            plt.title(f'Spectral Density of Hessian of Loss for (Epoch {epoch})')
            plt.show()

if __name__ == '__main__':
    run_model(num_trials=10)
