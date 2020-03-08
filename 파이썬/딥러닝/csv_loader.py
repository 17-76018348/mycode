
# csv load and move to dictionary
def csvload_move2dic:
    file = open("Real estate.csv")
    
    line_idx = 0
    house_price = []
    test_x = dict()
    test_y = []
    test_result = []
    x_label = []
    x_data = dict()
    corr_xy = []
    corr_xx = []
    for line in file:
        line_split = line.strip().split(',')
        if line_idx == 0:
            for i in range(1,7):
                x_label.append(line_split[i])
                x_data[line_split[i]] = []
                test_x[line_split[i]] = []
        elif line_idx >= 1 and line_idx<370:
            for i in range(1,7):
                x_data[x_label[i-1]].append(float(line_split[i]))
            house_price.append(float(line_split[-1]))
        elif line_idx>=370:
            for i in range(1,7):
                test_x[x_label[i-1]].append(float(line_split[i]))
            test_y.append(float(line_split[-1]))
        line_idx += 1

# normalization
def normalization_dic_mean_std:
    for i in range(len(x_label)):
       test_x[x_label[i]] = np.array(test_x[x_label[i]])
       test_x[x_label[i]] -= np.mean(x_data[x_label[i]])
       test_x[x_label[i]] /= np.std(x_data[x_label[i]])
    
       x_data[x_label[i]] = np.array(x_data[x_label[i]])
       x_data[x_label[i]] -= np.mean(x_data[x_label[i]])
       x_data[x_label[i]] /= np.std(x_data[x_label[i]])