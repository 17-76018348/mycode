def cuda():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


def dataset_plot():
    x_train_data, y_train_data, x_test_data, y_test_data = dataset_generator()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (30,15))
    cmap = cm.get_cmap('bwr_r', 2)
    ax1.grid()
    ax2.grid()
    ax1.set_title("Train Dataset", fontsize = 30)
    ax2.set_title("Test Dataset", fontsize = 30)
    fig.subplots_adjust(top = 0.9, bottom = 0.1, left = 0.1, right = 0.9,
                        wspace = 0.05)
    ax1.scatter(x_train_data[:,0], x_train_data[:,1], marker = 'o', color = cmap(y_train_data), alpha = 0.4)
    ax2.scatter(x_test_data[:,0], x_test_data[:,1], marker = 'o', color = cmap(y_test_data), alpha = 0.4)

def dataset2tensor():
    x_train_data = torch.tensor(x_train_data,dtype = torch.float)
    y_train_data = torch.tensor(y_train_data,dtype = torch.float)
    x_test_data = torch.tensor(x_test_data,dtype = torch.float)
    y_test_data = torch.tensor(y_test_data,dtype = torch.float)
    dataset = TensorDataset(x_train_data, y_train_data)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)


class MLP_Classifier(nn.Module):
    def __init__(self,input_size,n_neuron,n_neuron_2,output_size):
        super(MLP_Classifier,self).__init__()
        self.input_size, self.n_neuron,self.n_neuron_2,self.output_size = input_size, n_neuron, n_neuron_2,output_size
        self.fc1 = nn.Linear(self.input_size,self.n_neuron)
        self.fc2 = nn.Linear(self.n_neuron,self.n_neuron_2)
        self.fc3 = nn.Linear(self.n_neuron_2,self.output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    def forward(self,x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def train():
    input_size = 2
    lr = 0.03
    output_size = 1
    epochs = 5000
    loss_list = []
    n_neuron_list = [10,20,30]
    print("1")
    n_neuron = 30
    n_neuron_2 = 10
    model = MLP_Classifier(input_size,n_neuron,n_neuron_2,output_size)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(),lr = lr)
    for i in range(epochs):
            if  i % 100 == 0:
                print(i)
            for batch_idx, samples in enumerate(dataloader):
                x_train_data, y_train_data = samples
                optimizer.zero_grad()
                pred = model(x_train_data)
                     
                loss = criterion(pred,y_train_data)
                loss.backward()
                optimizer.step()
                
                loss_list.append(loss.detach().numpy())
    
    fig, ax = plt.subplots(figsize = (20,20))
    ax.plot(loss_list)

def dataset_generator():
     n_train_point = 5000
    x_train_data = np.random.uniform(low = -2, high = 2, size = (n_train_point, 2))
    y_train_data = np.zeros(shape = (n_train_point))
    
    n_test_point = 2000
    x_test_data = np.random.uniform(low = -2, high = 2, size = (n_test_point, 2))
    y_test_data = np.zeros(shape = (n_test_point))
    
    for data_idx in range(n_train_point):
        if x_train_data[data_idx, 1] >=  0.2*x_train_data[data_idx, 0] - 1 and x_train_data[data_idx, 0] <= -1 + 0.1*np.random.normal(0,1,1):
            y_train_data[data_idx] = 1.
        if x_train_data[data_idx, 1] <=  -0.2*x_train_data[data_idx, 0] + 1 and x_train_data[data_idx, 0] >= 1 + 0.1*np.random.normal(0,1,1):
            y_train_data[data_idx] = 1.
        if np.power(x_train_data[data_idx, 0], 2) + np.power(x_train_data[data_idx, 1], 2) <= 0.25 + 0.1*np.random.normal(0,1,1):
            y_train_data[data_idx] = 1.
    
    for data_idx in range(n_test_point):
        if x_test_data[data_idx, 1] >=  0.2*x_test_data[data_idx, 0] - 1 and x_test_data[data_idx, 0] <= -1 + 0.1*np.random.normal(0,1,1):
            y_test_data[data_idx] = 1.
        if x_test_data[data_idx, 1] <=  -0.2*x_test_data[data_idx, 0] + 1 and x_test_data[data_idx, 0] >= 1 + 0.1*np.random.normal(0,1,1):
            y_test_data[data_idx] = 1.
        if np.power(x_test_data[data_idx, 0], 2) + np.power(x_test_data[data_idx, 1], 2) <= 0.25 + 0.1*np.random.normal(0,1,1):
            y_test_data[data_idx] = 1.
            
    return x_train_data, y_train_data, x_test_data, y_test_data

def sphere_dataset():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    n_train_point = 5000
    x_train_data = np.random.uniform(low = -1.5, high = 1.5, size = (n_train_point, 3))
    y_train_data = np.zeros(shape = (n_train_point, 1))
    
    n_test_point = 5000
    x_test_data = np.random.uniform(low = -1.5, high = 1.5, size = (n_test_point, 3))
    y_test_data = np.zeros(shape = (n_test_point, 1))
    
    y_train_data = (np.power(x_train_data[:,1], 2) + np.power(x_train_data[:,0],2) + np.power(x_train_data[:,2],2) <= 1 + 0.1*np.random.normal(0, 1, size = (n_train_point))).astype(int)
    y_test_data = (np.power(x_test_data[:,1], 2) + np.power(x_test_data[:,0],2) + np.power(x_test_data[:,2],2) <= 1 + 0.1*np.random.normal(0, 1, size = (n_test_point))).astype(int)
    
    return x_train_data, y_train_data, x_test_data, y_test_data

def tester(x_train_data, y_train_data, model, trained_dict):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    x_train_data, y_train_data = x_train_data.cpu().numpy(), y_train_data.cpu().numpy().reshape(-1)    
    cmap = cm.get_cmap('bwr_r', 2)
    
    model.load_state_dict(trained_dict)
    
    
    fig, ax2 = plt.subplots(figsize = (15,15))
    ax2.scatter(x_train_data[:,0], x_train_data[:,1], marker = 'o', color = cmap(y_train_data), alpha = 0.4)
    test_x1 = np.linspace(-2, 2, 500)
    test_x2 = np.linspace(-2, 2, 600)
    X1, X2 = np.meshgrid(test_x1, test_x2)
    
    test_X = np.dstack((X1, X2)).reshape(-1,2)
    test_result = model(torch.tensor(test_X, dtype = torch.float, device = device))
    test_result = test_result.view(600,500).detach().cpu().numpy()
    ax2.pcolor(X1, X2, test_result, cmap = 'bwr_r', alpha = 0.2)
    ax2.axis('off')
    fig.savefig('./decision_boundary.png')

def error_rate():
    trained_dict = model.state_dict()
    model = MLP_Classifier(input_size, n_neuron, n_neuron_2, output_size)ifier(input_size,n_neuron,n_neuron_2,output_size)
    tester(x_test_data, y_test_data, model, trained_dict)

    
    
    test_x = torch.normal(0,1,size = (10,2))
    
    pred = model(test_x)
    
    label = torch.tensor([2,0,0,2,1,2,1,3,2,1])
    
    topv, topi = pred.topk(1,dim = 1)
    topi = topi.view(-1)
    n_correct = (topi == label).to(float).sum()
    print(n_correct/label.shape[0])





