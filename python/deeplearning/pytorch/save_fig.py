class LinearRegression(nn.Module):
    def __init__(self,input_size,output_size):
        super(LinearRegression,self).__init__()
        self.input_size, self.output_size = input_size, output_size
        self.fc1 = nn.Linear(self.input_size,self.output_size)
        self.sigmoid = nn.sigmoid()
        
    def forward(self,x):
        return self.fc1(x)
    
    
def save_fig():
    x_data = torch.tensor(x_data,dtype = torch.float)
    y_data = torch.tensor(y_data,dtype = torch.float)
    lr = 0.001
    epochs = 1000
    check_freq = 101
    criterion = nn.MSELoss()
    optimizier = torch.optim.SGD(model.parameters(),lr = lr)
    loss_list = []
    for i in range(epochs):
        pred= model(x_data)
        optimizier.zero_grad()
        loss = criterion(pred,y_data)
        loss.backward()
        optimizier.step()
        loss_list.append(loss.detach().numpy())
        
        if i % check_freq == 0:
            fig, (ax1, ax2) = plt.subplots(2,1,figsize = (15,15))
    #        ax1.plot(loss_list)
    #        ax2.plot(x_data, y_data,'bo')
            x_min , x_max = x_data.min().numpy(), x_data.max().numpy()
            
            weight = model.state_dict()['fc1.weight'][0][0].numpy()
            bias = model.state_dict()['fc1.bias'][0].numpy()
            
            y0 = x_min *weight + bias
            y1 = x_max * weight + bias
    #        ax2.plot([x_min,x_max],[y0,y1],'r',linewidth = 3)
            
            fig.savefig('./model'+str(i)+'.png')