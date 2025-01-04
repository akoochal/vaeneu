import numpy as np
import torch

class Dataset:
    def __init__(self,ts_df,horizon,history_window_size,device) -> None:
        
        self.horizon = horizon
        self.history_window_size = history_window_size
                
        self.device = device
        self.train_horizon = 1

        self.train_data = []
        test_data = []
        
        # Test set size is 5 x horizon
        n_test_samples = 5
        row_np = ts_df.loc[:,"series_value"][0].to_numpy()[:,np.newaxis]
            
        self.train_data = row_np[:-n_test_samples*self.horizon]

        test_data = row_np[-n_test_samples*self.horizon-self.history_window_size:]

        test_temp = []
        # Moving window wiht length of horizon on raw test set.
        for i in range(n_test_samples):
            test_temp.append(test_data[i*self.horizon:self.history_window_size+(i+1)*self.horizon])

        test_data = np.array(test_temp)

        self.test_x = test_data[:,:self.history_window_size]
        self.test_y = test_data[:,-self.horizon:]

        self.mean = self.train_data.mean()
        self.std = self.train_data.std()
        self.train_data = self.transform(self.train_data)
        self.test_x = self.transform(self.test_x)
        
    def get_random_batch(self,batch_size):
        x = []
        y = []
        starts = np.random.randint(self.train_data.shape[0]-self.train_horizon-self.history_window_size,size=batch_size)
        for start in starts:
            x.append(self.train_data[start:start+self.history_window_size])
            y.append(self.train_data[start+self.history_window_size:start+self.history_window_size+self.train_horizon])


        x = torch.tensor(np.array(x),dtype=torch.float32,device=self.device)
        y = torch.tensor(np.array(y),dtype=torch.float32,device=self.device)
        
        return x,y
    
    def get_test_set(self):        
        return torch.tensor(np.array(self.test_x),dtype=torch.float32,device=self.device), self.test_y
    
    def transform(self,data):
        return (data - self.mean) / self.std

    def inverse_transform(self,data):
        return data * self.std + self.mean