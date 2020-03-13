
#node
class plus_node():
    def __init__(self):
        self.x,self.y,self.z = None,None,None
    def forward(self,x,y):
        self.x,self.y,self.z = x,y,x+y
        return self.z
    def backward(self,dL):
        return dL, dL
class plus5_node():
    def __init__(self):
        self.x1,self.x2,self.x3,self.x4,self.x5,self.z = None, None, None, None, None, None
    def forward(self,x1,x2,x3,x4,x5):
        self.x1,self.x2,self.x3,self.x4,self.x5,self.z = x1,x2,x3,x4,x5,x1+x2+x3+x4+x5
        return self.z
    def backward(self,dL):
        return dL,dL,dL,dL,dL
class plus4_node():
    def __init__(self):
        self.x1,self.x2,self.x3,self.x4,self.z = None, None, None, None, None
    def forward(self,x1,x2,x3,x4):
        self.x1,self.x2,self.x3,self.x4,self.z = x1,x2,x3,x4,x1+x2+x3+x4
        return self.z
    def backward(self,dL):
        return dL,dL,dL,dL
class minus_node():
    def __init__(self):
        self.x,self.y,self.z = None,None,None
    def forward(self,x,y):
        self.x, self.y, self.z = x, y, x - y
        return self.z
    def backward(self,dL):
        return dL, -1 * dL
class mul_node():
    def __init__(self):
        self.x, self.y, self.z = None, None, None
        
    def forward(self, x, y):
        self.x, self.y, self.z = x, y, x*y
        return self.z
    def backward(self, dL):
        return self.y*dL, self.x*dL   
class square_node():
    def __init__(self):
        self.x, self.z = None, None
    
    def forward(self, x):
        self.x, self.z = x, x*x
        return self.z
    
    def backward(self, dL):
        return 2*self.x*dL
class cost_node():
    def __init__(self):
        self.x, self.z = None, None
    
    def forward(self, x):
        self.x = x
        self.z = np.mean(self.x)
        return self.z
    def backward(self):
        return 1/len(self.x)*np.ones(shape = (len(self.x)))
    
    
    
def linear_regression_basic_y_equals_x:
# y = x forward and backpropagation -> plot
    theta = 0
    lr = 0.001
    epochs = 10
    x_data = np.array([1,2,3,4,5])
    y_data = np.array([1,2,3,4,5])
    mul = mul_node()
    minus = minus_node()
    square = square_node()
    
        def forward(x,y,theta):
            z_1 = mul.forward(x,theta)
            z_2 = minus.forward(y,z_1)
            loss = square.forward(z_2)
            return loss
        def backwward(theta,lr):
            dz_2 = square.backward_last()
            dy,dz_1 = minus.backward(dz_2)
            dx,dtheta = mul.backward(dz_1)
            theta = theta - lr * dtheta
            return theta
    
    
    loss_list = []
    theta_list = []
    for i in range(epochs):
        for j in range(len(x_data)):
            ##### Your Code(Forward Propagation) #####
            loss = forw(x = x_data[j], y = y_data[j] ,theta = theta)
            loss_list.append(loss)
    
            ##### Your Code(Forward Propagation) #####
            
            
            ##### Your Code(Backward Propagation) #####
            theta = backw(theta = theta, lr = lr)
            theta_list.append(theta)
    
            ##### Your Code(Backward Propagation) #####
            
    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(loss_list)
    ax.set_title("loss")
    
    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(theta_list)
    ax.set_title("theta")    
    
    
    
    
    
    
#plot loss_list and theta_list
    
    
def plot_loss_theta_list:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,10))
    ax1.grid()
    ax1.plot(loss_list)
    ax1.set_title("loss", fontsize = 50)
    
    #fig, ax1 = plt.subplots(figsize = (10,10))
    ax2.plot(theta4_list, label = r"$\theta_{4}$")
    ax2.plot(theta3_list, label = r"$\theta_{3}$")
    ax2.plot(theta2_list, label = r"$\theta_{2}$")
    ax2.plot(theta1_list, label = r"$\theta_{1}$")
    ax2.plot(theta0_list, label = r"$\theta_{0}$")
    fig.legend(fontsize = 'xx-large')
    ax2.set_title(r"$\theta_{1}, \theta_{0} Update$", fontsize = 50)
    ax2.grid()
    
    plt.plot(test_result)
    plt.grid()
    plt.show()


# numpy forward and backpropagation minibatch random shuffle\
def numpy_regression_minibatch_shuffle:    
    theta = 0
    lr = 0.00000000000000001
    epochs = 150
    batch_size = 100
    z1_node = mul_node()
    z2_node = minus_node()
    z3_node = square_node()
    c_node = cost_node()
    
    loss_list = []
    theta_list = []
    xy_data = list(zip(x_data,y_data))
    print(xy_data)
    for i in range(epochs):
        gradient_np = np.empty(0)
        theta_np = np.empty(0)
        batch_data = random.sample(xy_data,batch_size)
        batch_x , batch_y = zip(*batch_data)
        batch_x = np.array(list(batch_x))
        batch_y = np.array(list(batch_y))
        
        ##### Your Code(Forward Propagation) #####
        z1 = z1_node.forward(batch_x,theta)
        z2 = z2_node.forward(batch_y,z1)
        z3 = z3_node.forward(z2)
        
        ##### Your Code(Forward Propagation) #####
        
            
        cost = c_node.forward(z3)
        loss_list.append(cost)
        dz = c_node.backward()
        
        dz2 = z3_node.backward(dz)
        dy, dz1 = z2_node.backward(dz2)
        dx,dtheta = z1_node.backward(dz1)
        theta = theta - lr*np.mean(dtheta)
        theta_list.append(theta)
    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(loss_list)
    ax.set_title("loss")
    
    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(theta_list)
    ax.set_title(r"$\theta$")


def shuffle():
    import random
    list = ['a','b','c','d']
    random.shuffle(list)
    print(list)


# 3 input linear regression
def input_3_linear_regression

    theta3, theta2, theta1, theta0 = 0, 0, 0, 0
    lr = 0.001
    epochs = 6000
    
    
    z3_node = nd.mul_node()
    z2_node = nd.mul_node()
    z1_node = nd.mul_node()
    z5_node = nd.plus4_node()
    z6_node = nd.minus_node()
    loss_node = nd.square_node()
    c_node = nd.cost_node()
    
    loss_list = []
    
    theta3_list,theta2_list,theta1_list, theta0_list = [], [], [], []
    
    for i in range(epochs):
        # forward
        z3 = z3_node.forward(theta3,x_data[x_label[3]])
        z2 = z2_node.forward(theta2,x_data[x_label[4]])
        z1 = z1_node.forward(theta1,x_data[x_label[5]])
        z5 = z5_node.forward(z1,z2,z3,theta0)
        z6 = z6_node.forward(house_price,z5)
        loss = loss_node.forward(z6)
        cost = c_node.forward(loss)
        loss_list.append(cost)
        
        # backward
        dcost = c_node.backward()
        dloss = loss_node.backward(dcost)
        dy, dz6 = z6_node.backward(dloss)
        dz1,dz2,dz3,dtheta0 = z5_node.backward(dz6)
        
        dtheta3, dx3 = z3_node.backward(dz3)
        dtheta2, dx2 = z2_node.backward(dz2)
        dtheta1, dx1 = z1_node.backward(dz1)
        
        theta3 -= lr*np.sum(dtheta3)
        theta2 -= lr*np.sum(dtheta2)
        theta1 -= lr*np.sum(dtheta1)
        theta0 -= lr*np.sum(dtheta0)
        
        theta3_list.append(theta3)
        theta2_list.append(theta2)
        theta1_list.append(theta1)
        theta0_list.append(theta0)
        
        
    predict = theta3 * test_x[x_label[3]]\
              +theta2 * test_x[x_label[4]] + theta1 * test_x[x_label[5]] + theta0
    test_result =  test_y - predict
    print("평균:    ",np.mean(test_result))



# 4 input Linear regression
def input_4_linear_regression:
    theta4, theta3, theta2, theta1, theta0 = 0, 0, 0, 0, 0
    lr = 0.0003
    epochs = 12500
    
    z4_node = nd.mul_node()
    z3_node = nd.mul_node()
    z2_node = nd.mul_node()
    z1_node = nd.mul_node()
    z5_node = nd.plus5_node()
    z6_node = nd.minus_node()
    loss_node = nd.square_node()
    c_node = nd.cost_node()
    
    loss_list = []
    
    theta4_list,theta3_list,theta2_list,theta1_list, theta0_list = [], [], [], [], []
    
    for i in range(epochs):
        # forward
        z4 = z4_node.forward(theta4,x_data[x_label[2]])
        z3 = z3_node.forward(theta3,x_data[x_label[3]])
        z2 = z2_node.forward(theta2,x_data[x_label[4]])
        z1 = z1_node.forward(theta1,x_data[x_label[5]])
        z5 = z5_node.forward(z1,z2,z3,z4,theta0)
        z6 = z6_node.forward(house_price,z5)
        loss = loss_node.forward(z6)
        cost = c_node.forward(loss)
        loss_list.append(cost)
        
        # backward
        dcost = c_node.backward()
        dloss = loss_node.backward(dcost)
        dy, dz6 = z6_node.backward(dloss)
        dz1,dz2,dz3,dz4,dtheta0 = z5_node.backward(dz6)
        dtheta4, dx4 = z4_node.backward(dz4)
        dtheta3, dx3 = z3_node.backward(dz3)
        dtheta2, dx2 = z2_node.backward(dz2)
        dtheta1, dx1 = z1_node.backward(dz1)
        
        theta4 -= lr*np.sum(dtheta4)
        theta3 -= lr*np.sum(dtheta3)
        theta2 -= lr*np.sum(dtheta2)
        theta1 -= lr*np.sum(dtheta1)
        theta0 -= lr*np.sum(dtheta0)
        
        theta4_list.append(theta4)
        theta3_list.append(theta3)
        theta2_list.append(theta2)
        theta1_list.append(theta1)
        theta0_list.append(theta0)
        
        
    predict = theta4 * test_x[x_label[2]] + theta3 * test_x[x_label[3]]\
              +theta2 * test_x[x_label[4]] + theta1 * test_x[x_label[5]] + theta0
    test_result =  test_y - predict
    print("평균:    ",np.mean(test_result))

class regression2binary_classification():
    def __init__():
        pass
    def dataset_generator():
        # Pass/No Pass 학생을 각각 50명씩 만들도록 세팅
        n_P, n_NP = 50, 50
        # 학생들의 최소/최대 공부시간을 hour_m, hour_M으로 만들고 decision boundary를 hour_b로 만듦
        # 즉, NP학생들은 2~4시간 공부시간을 가지고 P학생들은 4~6시간의 공부시간을 가짐
        hour_m, hour_b, hour_M = 2, 4, 6
        
        ##### Your Code(Dataset Generation/Start) #####
        study_hour_P = np.random.uniform(low = hour_b, high = hour_M, size = (n_P,))
        study_hour_NP = np.random.uniform(low = hour_m, high = hour_b, size = (n_NP,))
        
        # P, NP학생들의 y값들은 각각 1, 0으로 만들어줌
        result_P = np.ones_like(study_hour_P)
        result_NP = np.zeros_like(study_hour_NP)
        ##### Your Code(Dataset Generation/End) #####
        
        print("study_hour_P:", study_hour_P[:5])
        print("study_hour_NP:", study_hour_NP[:5])
        print("result_P:", result_P[:5])
        print("result_NP:", result_NP[:5], '\n')
        print("study_hour_P.shape:", study_hour_P.shape)
        print("study_hour_NP.shape:", study_hour_NP.shape)
        print("result_P.shape:", result_P.shape)
        print("result_NP.shape:", result_NP.shape)
    def dataset_plot():
        ##### Your Code(Dataset Generation/Start) #####
        x_data = np.hstack([study_hour_P,study_hour_NP])
        y_data = np.hstack([result_P,result_NP])
        ##### Your Code(Dataset Generation/End) #####
        
        fig, ax = plt.subplots(figsize = (12,5))
        ax.plot(x_data[:n_P], y_data[:n_P], 'bo')
        ax.plot(x_data[n_P:], y_data[n_P:], 'ro')
        ax.grid()
    def training():
        Z1_node = mul_node()
        Z2_node = plus_node()
        Z3_node = minus_node()
        L_node = square_node()
        J_node = cost_node()
        theta1, theta0 = 0, 0# theta1, theta0 설정
        lr = 0.001# learning rate 설정
        epochs = 50000#총 epoch 설정
        
        cost_list = []
        theta1_list, theta0_list = [], []
        for i in range(epochs):
            ##### Your Code(Learning Process/Start) #####
            Z1 = Z1_node.forward(theta1,x_data)
            Z2 = Z2_node.forward(Z1,theta0)
            Z3 = Z3_node.forward(y_data,Z2)
            L = L_node.forward(Z3)
            J = J_node.forward(L)
            
            dL = J_node.backward()
            dZ3 = L_node.backward(dL)
            dY, dZ2 = Z3_node.backward(dZ3)
            dZ1, dTheta0 = Z2_node.backward(dZ2)
            dTheta1, dX = Z1_node.backward(dZ1)
            #### Your Code(Learning Process/End) #####
    
            theta1 = theta1 - lr*np.sum(dTheta1)
            theta0 = theta0 - lr*np.sum(dTheta0)
            
            cost_list.append(J)
            theta1_list.append(theta1)
            theta0_list.append(theta0)
    def plot_result():
        fig, ax = plt.subplots(2, 1, figsize = (12, 8))
        ax[0].set_title("Cost")
        ax[1].set_title(r'$\theta_{1} \quad and \quad \theta_{0}$')
        ax[0].plot(cost_list)
        ax[1].plot(theta1_list, label = r'$\theta_{1}$')
        ax[1].plot(theta0_list, label = r'$\theta_{0}$')
        ax[1].legend(loc = 'upper right', fontsize = 20)
        
        x_min, x_max = x_data.min(), x_data.max()
        y_min, y_max = x_min*theta1 + theta0, x_max*theta1 + theta0
        x_range = np.linspace(x_min, x_max, 1000)
        y_range = x_range*theta1 + theta0
        y_d_idx = np.where(np.abs(y_range - 0.5) == np.min(np.abs(y_range - 0.5)))
        x_d_val = x_range[y_d_idx]
        
        fig, ax = plt.subplots(figsize = (12,5))
        ax.plot(x_data[:n_P], y_data[:n_P], 'bo')
        ax.plot(x_data[n_P:], y_data[n_P:], 'ro')
        ax.plot([x_min, x_max], [y_min, y_max], 'r', linewidth = 2)
        ax.plot([x_range[y_d_idx], x_range[y_d_idx]], [0, y_range[y_d_idx]], 'purple', linewidth = 3)
        ax.plot(x_range[y_d_idx], y_range[y_d_idx], 'purple', marker = 'o', markersize = 10)
        ax.text(x_range[y_d_idx]*1.05, y_range[y_d_idx],
                s = "Decision Boundary:" + str(np.round(x_range[y_d_idx], 2)),
               fontdict = {'color':  'purple', 'fontsize': 20})
        ax.grid()
    def dataset_with_outlier():
        n_out = 20
        hour_out_m, hour_out_M = 12, 15
        ##### Your Code(Adding Outliers/Start) #####
        study_hour_outlier = np.random.uniform(low = hour_out_m, high = hour_out_M, size = n_out)
        result_outlier = np.ones_like(study_hour_outlier)
        
        study_hour_P = np.append(study_hour_P,study_hour_outlier)
        result_P = np.append(result_P,result_outlier)
        
        x_data = np.hstack([study_hour_P,study_hour_NP])
        y_data = np.hstack([result_P,result_NP])
        ##### Your Code(Adding Outliers/Start) #####
        
        print("study_hour_P.shape:",study_hour_P.shape)
        print("result_P.shape:", result_P.shape)
        
        fig, ax = plt.subplots(figsize = (12,5))
        ax.plot(x_data[:n_P + n_out], y_data[:n_P + n_out], 'bo')
        ax.plot(x_data[n_P + n_out:], y_data[n_P + n_out:], 'ro')
        ax.grid()

def sigmoid_function():
    np.random.seed(0)

    x_range = np.linspace(-10, 10, 1000)
    ##### Your Code(Sigmoid Function/Start) #####
    y = 1/(1+np.exp(-1*x_range))
    ##### Your Code(Sigmoid Function/Start) #####
    
    fig, ax = plt.subplots(figsize = (12, 5))
    ax.plot(x_range, y)
    ax.grid()
    
class sigmoid_node():
    def __init__(self):
        self.x, self.y = None, None
        
    def forward(self, x):
        self.x, self.y = x, 1/(1+np.exp(-1*x))
        return self.y
    def backward(self, dL):
        return self.y * (1 - self.y) * dL
def sigmoid_train():
    Z1_node = mul_node()
    Z2_node = plus_node()
    Z3_node = sigmoid_node()
    Z4_node = minus_node()
    L_node = square_node()
    J_node = cost_node()
    theta1, theta0 = 0, 0# theta1, theta0 설정
    lr = 0.03# learning rate 설정
    epochs = 10000#총 epoch 설정
    
    cost_list = []
    theta1_list, theta0_list = [], []    
    for i in range(epochs):
        ##### Your Code(Learning Process/Start) #####
        Z1 = Z1_node.forward(theta1,x_data)
        Z2 = Z2_node.forward(Z1,theta0)
        Z3 = Z3_node.forward(Z2)
        Z4 = Z4_node.forward(y_data,Z3)
        L = L_node.forward(Z4)
        J = J_node.forward(L)
        
        dL = J_node.backward()
        dZ4 = L_node.backward(dL)
        dY, dZ3 = Z4_node.backward(dZ4)
        dZ2 = Z3_node.backward(dZ3)
        dZ1, dTheta0 = Z2_node.backward(dZ2)
        dTheta1, dX = Z1_node.backward(dZ1)
        ##### Your Code(Learning Process/End) #####
        
        theta1 = theta1 - lr*np.sum(dTheta1)
        theta0 = theta0 - lr*np.sum(dTheta0)
        
        cost_list.append(J)
        theta1_list.append(theta1)
        theta0_list.append(theta0)