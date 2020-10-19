import numpy as np
import torch
from torch.autograd import Variable
from algo import ValueFunctionWithApproximation
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ValueFunctionWithNN(ValueFunctionWithApproximation,torch.nn.Module):
	
	def __init__(self,
	    state_dims):
		
		super().__init__()
		self.net = torch.nn.Sequential(
		torch.nn.Linear(state_dims, 32),
		torch.nn.ReLU(),
		torch.nn.Linear(32, 32),
		torch.nn.ReLU(),
		torch.nn.Linear(32, 32),
		torch.nn.ReLU(),
		torch.nn.Linear(32, 1))
		
		self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999))
		self.loss_func = torch.nn.MSELoss()    	

	def __call__(self,s):
		    # TODO: implement this method

		s = torch.from_numpy(s).float()
		    # print(tself
		self.net.eval()
		value=self.net(s)

		    #print(type(value))
		return value.detach().numpy()[0]

	def update(self,alpha,G,s_tau):
		    # TODO: implement this method
		    #self.net.train()
		s_tau = torch.from_numpy(s_tau).float()
		prediction = self.net(s_tau)

		
		self.net.train()
		G = torch.tensor(float(G))
		loss = self.loss_func(prediction,G)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

