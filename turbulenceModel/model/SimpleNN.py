import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from turbulenceModel.utils.Loader import get_combined_dataframe

# Define a simple neural network as you specified
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

def run_model():
    data = get_combined_dataframe()

    # Select relevant columns for input features
    features = ['drone1_x', 'drone1_y', 'drone1_z', 'drone1_roll', 'drone1_pitch', 'drone1_yaw', 'drone1_thrust',
                'drone2_x', 'drone2_y', 'drone2_z', 'drone2_roll', 'drone2_pitch', 'drone2_yaw', 'drone2_thrust']
    target = 'downwash_force_at_pos_drone1'

    # Extract input and target data
    X = data[features].values  # Input features (shape: [samples, n_features])
    y = data[target].values  # Target labels (shape: [samples, 1])

    # Convert to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Ensure y is of shape [samples, 1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # Define the network parameters
    n0 = X_train.shape[1]  # Input size: number of features
    n1 = 3  # Hidden layer size (you can change this)
    n2 = 1  # Output size (downwash force)

    # Create the model
    model = SimpleNN(n0, n1, n2)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

    # Early stopping parameters
    patience = 10  # Stop after 10 epochs without improvement
    best_loss = float('inf')
    epochs_without_improvement = 0

    # Number of epochs for training
    num_epochs = 1000 # Number of training epochs

    # Training loop
    train_losses = []  # Store training losses for later plotting
    val_losses = []    # Store validation losses for later plotting

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        # Perform a forward pass
        optimizer.zero_grad()  # Zero out the gradients from the previous step
        predictions = model(X_train)  # Forward pass: model output

        # Compute the training loss
        loss = criterion(predictions, y_train)

        # Backpropagation
        loss.backward()  # Compute gradients via backpropagation

        # Update the model parameters
        optimizer.step()  # Perform one step of gradient descent

        # Compute validation loss
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_preds = model(X_test)
            val_loss = criterion(val_preds, y_test)

        # Track training and validation loss for convergence plotting
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        # Early stopping condition: if validation loss does not improve
        print("Val loss to loss diff is " + str(val_loss.item() - loss.item()))
        if (abs(val_loss.item() - loss.item())) > 0.0001:
            epochs_without_improvement = 0  # Reset counter
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
            break

        # Print loss for monitoring
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    # After training, evaluate the model on the test set
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for testing
        test_preds = model(X_test)

    # Calculate performance metrics for the test set
    test_loss = mean_squared_error(y_test.numpy(), test_preds.numpy())
    test_mae = mean_absolute_error(y_test.numpy(), test_preds.numpy())
    test_r2 = r2_score(y_test.numpy(), test_preds.numpy())

    # Print test performance metrics
    print("\nTest Performance Metrics:")
    print(f"Test MSE: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test R^2: {test_r2:.4f}")

    # Plot training and validation loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

    # Plot predicted vs actual for test set
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test.numpy(), test_preds.numpy(), color='blue', label='Predicted vs Actual')
    plt.plot([min(y_test.numpy()), max(y_test.numpy())], [min(y_test.numpy()), max(y_test.numpy())], color='red', label="Perfect prediction")
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual (Test Set)')
    plt.legend()
    plt.show()

    # Compute the residuals
    residuals = y_test - test_preds

    # Plot residuals vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(test_preds.numpy(), residuals.numpy(), color='blue', label='Residuals vs Predicted')
    plt.axhline(y=0, color='red', linestyle='--', label="Zero Residuals Line")
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    plt.legend()
    plt.show()

    # Plot histogram of residuals
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals.numpy(), kde=True, bins=30, stat="density", color="blue")
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Histogram of Residuals')
    plt.show()

    # Save the model's state_dict (weights and biases)
    torch.save(model.state_dict(), './force_trained_shallow_model.pth')


# Initialization for the weights
def weights_init(m):
    if isinstance(m, nn.Linear):
        # Initialize weights as Gaussian with mean=0, std=0.1
        nn.init.normal_(m.weight, mean=0.0, std=0.1)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

if __name__ == '__main__':
    run_model()
