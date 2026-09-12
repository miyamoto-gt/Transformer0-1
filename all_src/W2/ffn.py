import torch 

class FeedForward(torch.nn.Module):
    def __init__(self,d_model,mult=4):
        super().__init__()
        self.w1=torch.nn.Linear(d_model,mult*d_model)
        self.w2=torch.nn.Linear(mult*d_model,d_model)

    def forward(self,x):
        h=self.w1(x)       # xW1^T + b1      (B,T,d) -> (B,T,4d)
        h=h.clamp(min=0)   # max(0, ・)
        return self.w2(h)  # ・ W2 + b2     (B,T,4d) -> (B,T,d)