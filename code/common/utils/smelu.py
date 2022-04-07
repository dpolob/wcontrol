import torch


class SmeLU(torch.nn.Module):
    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return self.smelu(x)
    
    def smelu(self, x):
        if x <= -self.beta:
            return 0
        elif torch.abs(x) <= self.beta:
            return (x + self.beta)**2 / (4 * self.beta)
        elif x >= self.beta:
            return x
            
        