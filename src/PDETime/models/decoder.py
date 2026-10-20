import torch.nn as nn
import torch

class RidgeRegression(nn.Module):
    """
    Ridge Regression model for linear regression with L2 regularization.
    
    Args:
        lambda_ (float): Regularization strength. Must be non-negative.
        
    Attributes:
        lambda_ (float): Regularization strength.
    """
    
    def __init__(self, lambda_):
        super().__init__()
        if lambda_ < 0:
            raise ValueError("Ridge regressor regularization strength must be non-negative")
        self.lambda_ = lambda_
        
    def forward(self, x, y):
        """
        Forward pass of the Ridge Regression model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, window_length, input_dim).
            y (torch.Tensor): Target tensor of shape (batch_size, window_length, output_dim).
            
        Returns:
            torch.Tensor: Predicted weights tensor of shape (batch_size, window_length-1, output_dim).
            torch.Tensor: Predicted bias tensor of shape (batch_size, output_dim).
        """
        batch_size, window_length, _ = x.shape
        xb = torch.cat([x, x.new_ones(batch_size, window_length, 1)], dim=-1)
        xb_L = xb.mT @ xb
        xb_L.diagonal(dim1=-2, dim2=-1).add_(self.lambda_)
        wb = torch.linalg.solve(xb_L, xb.mT @ y)
        return wb[:,:-1,:], wb[:,-1,:]

    
class Decoder(nn.Module):
    """
    Decoder module for PDETime model. Optimises loss for lookback window with Ridhe Regression.
    
    Args:
        lookback (int): Number of time steps to look back.
        horizon (int): Number of time steps to predict into the future.
        lambda_ (float, optional): Regularization parameter for Ridge Regression. Defaults to 0.
    """
    
    def __init__(self, lookback, horizon, lambda_=0.):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.ridge = RidgeRegression(lambda_=lambda_)
    
    def forward(self, tau, x):
        """
        Forward pass of the Decoder module.
        
        Args:
            tau (torch.Tensor): Input tensor of shape (batch_size, lookback + horizon, input_dim).
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            torch.Tensor: Predicted tensor of shape (batch_size, lookback + horizon, output_dim).
        """
        tau_r, tau_p = torch.split(tau, [self.lookback, self.horizon], dim=-2)
        w, b = self.ridge(tau_r, x)
        preds = torch.einsum('bwl,bld->bwd', tau, w) 
        preds += b.unsqueeze(-2).repeat(1, self.horizon + self.lookback, 1)
        return preds
