from model_rednet import REDNet30
from model_dncnn import DnCNN
from torchsummary import summary

# net = REDNet30()
net = DnCNN(channels=1,num_of_layers=30)
summary(net,input_size = (1,64,64))