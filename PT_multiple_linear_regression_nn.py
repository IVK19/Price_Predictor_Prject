import torch
from torch import nn
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np



data = pd.read_excel('modified_notebooks_data_frame_7_132.xlsx')

X_arr = data.drop('цена', axis=1).to_numpy().astype(np.float32)
y_arr = data['цена'].to_numpy().astype(np.float32)

X = torch.from_numpy(X_arr.astype(np.float32))
y = torch.from_numpy(y_arr.astype(np.float32))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class MultipleLinearRegressionModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=104):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(nn.Linear(in_features=input_features, out_features=hidden_units), 
                                                nn.RReLU(),
                                                nn.Linear(in_features=hidden_units, out_features=hidden_units), 
                                                nn.SiLU(),
                                                nn.Linear(in_features=hidden_units, out_features=hidden_units), 
                                                nn.ELU(),
                                                nn.Linear(in_features=hidden_units, out_features=hidden_units), 
                                                nn.ELU(),
                                                nn.Linear(in_features=hidden_units, out_features=hidden_units), 
                                                nn.Hardshrink(),
                                                nn.Linear(in_features=hidden_units, out_features=hidden_units), 
                                                nn.Hardshrink(),
                                                nn.Linear(in_features=hidden_units, out_features=hidden_units), 
                                                nn.Hardshrink(),
                                                nn.Linear(in_features=hidden_units, out_features=output_features))
        
    def forward(self, x):
        return self.linear_layer_stack(x)
    

torch.manual_seed(42)
pt_lr_nn_model_0 = MultipleLinearRegressionModel(input_features=13, output_features=1)


# Training
# Setup Loss function
loss_fn = nn.L1Loss()
# Setup our optimizer
optimizer = torch.optim.RAdam(params=pt_lr_nn_model_0.parameters(), lr=0.001)
# Let's write a training loop
epochs = 13200

for epoch in range(epochs+1):
    pt_lr_nn_model_0.train()
    # 1. Forward pass
    y_pred = pt_lr_nn_model_0(X_train).squeeze()
    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train)
    # 3. Optimizer zero grad
    optimizer.zero_grad()
    # 4. Perform backpropagation
    loss.backward()
    # 5. Optimizer step
    optimizer.step()
    # Testing
    pt_lr_nn_model_0.eval()
    with torch.inference_mode():
        test_pred = pt_lr_nn_model_0(X_test).squeeze()
        test_loss = loss_fn(test_pred, y_test)
        # Print out what's happening
        if epoch % 100 == 0:
            print(f'Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}')
# Turn model into evaluation mode
pt_lr_nn_model_0.eval()
# Make predictions on the test data
with torch.inference_mode():
    y_preds = pt_lr_nn_model_0(X_test)
print(y_preds, y_test)
# Saving and loading a trained model
# 1. Create models directory
# MODEL_PATH = Path('models')
# MODEL_PATH.mkdir(parents=True, exist_ok=True)
# # # 2. Create a model save path
# MODEL_NAME = 'pytorch_workflow_model_5.pth'
# MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
# # 3. Save the model state dictionary
# torch.save(obj=pt_lr_nn_model_0.state_dict(keep_vars=True), f=MODEL_SAVE_PATH)
ppn_model_5 = torch.jit.script(pt_lr_nn_model_0)
ppn_model_5.save('model5_1_scripted.pt')
# npp_model5 = MultipleLinearRegressionModel(input_features=13, output_features=1, hidden_units=104)
# npp_model5.load_state_dict(torch.load('models/pytorch_workflow_model_5.pth'))