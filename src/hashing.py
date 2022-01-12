import json
from random import shuffle
from re import X
import numpy as np
from numpy.core.fromnumeric import shape
from sklearn.metrics import mean_squared_error, pairwise
from sklearn.model_selection import train_test_split
import sys
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd

import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.modules.module import Module
from torch.utils.data import Dataset, DataLoader

datapath = "../data/patients.json"

# for lsh
num_hash_tables = 3
hash_dimension = 5

class HashTable:
    def __init__(self, hash_size, inp_dimensions):
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_table = dict()
        self.projections = np.random.randn(self.hash_size, inp_dimensions)
        
    def generate_hash(self, inp_vector): #a hash is a styring of 01s
        bools = (np.dot(inp_vector, self.projections.T) > 0).astype('int')
        return ''.join(bools.astype('str'))

    def __setitem__(self, inp_vec, index): # HashTableObj[input_vector] = label
        # questo serve per metterci gli indici dentro!!!
        hash_value = self.generate_hash(inp_vec)
        self.hash_table[hash_value] = self.hash_table\
            .get(hash_value, list()) + [index]
        
    def __getitem__(self, inp_vec):
        hash_value = self.generate_hash(inp_vec)
        return self.hash_table.get(hash_value, [])

class LSH: #manage multiple hastables
    def __init__(self, num_tables, hash_size, inp_dimensions):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_tables = list()
        for i in range(self.num_tables):
            self.hash_tables.append(HashTable(self.hash_size, self.inp_dimensions))
    
    def __setitem__(self, inp_vec, index):
        for table in self.hash_tables:
            table[inp_vec] = index
    
    def __getitem__(self, inp_vec):
        results = list()
        for table in self.hash_tables:
            results.extend(table[inp_vec])
        return list(set(results))


def main():
    with open(datapath, "r") as fp:
        data = json.load(fp)

    conditions = data["Conditions"]
    therapies = data["Therapies"]
    patients = data["Patients"]

    UxT = np.empty((len(patients), len(therapies)))
    UxT[:] = np.NaN
    # UxC = np.empty((len(patients), len(conditions)))
    # UxC[:] = np.NaN
    CxT = np.empty((len(conditions), len(therapies)))
    CxT[:] = np.NaN
    

    lshUxT = LSH(num_hash_tables, hash_dimension, len(therapies))
    lshCxT = LSH(num_hash_tables, hash_dimension, len(therapies))

    print(f'len users={len(patients)}, len ther= {len(therapies)}, len cond= {len(conditions)}')
    # print(f'UxT {UxT.shape} UxC {UxC.shape} CxT {CxT.shape}')

    for pat in patients:

        # populate U x T
        for tr in pat["trials"]: #patient therapy matrix
            UxT[int(pat["id"])][int(tr["therapy"])] = float(tr["successful"]) / 100 #since i do /100 it doesnt need normalization

        np.nan_to_num(UxT[int(pat["id"])], copy=False, nan=0.0)
        lshUxT[UxT[int(pat["id"])]] = int(pat["id"])
            
            # 2.01: WHAT HAPPENS IF I USE 1, 0s or 1, -1 ?
            #answer: it's just bad
            # if(int(tr["successful"]) >= 50):
            #     UxT[int(pat["id"])][int(tr["therapy"])] = 1.0
            # else:
            #     UxT[int(pat["id"])][int(tr["therapy"])] = -1.0

        # populate 
        for cd in pat["conditions"]: #patient condition matrix

            # UxC contains cured and uncured conditions
            # UxC[int(pat["id"])][int(cd["id"])] = 1.0

            # C x T contains ratings of therapies for cured conditions only
            if cd["cured"] != "NULL":
                for tr in pat["trials"]: #for each cured condition

                    if tr["condition"] == cd["id"]: #insert the rating for the therapies that were tried
                        
                        if isinstance(CxT[int(cd["id"])][int(tr["therapy"])], np.ndarray):
                            np.append(CxT[int(cd["id"])][int(tr["therapy"])], float(tr["successful"]) / 100)

                        else:
                            if ~np.isnan(CxT[int(cd["id"])][int(tr["therapy"])]):

                                CxT[int(cd["id"])][int(tr["therapy"])] = np.array( CxT[int(cd["id"])][int(tr["therapy"])] )
                                np.append(CxT[int(cd["id"])][int(tr["therapy"])], float(tr["successful"]) / 100)
                            
                            else:
                                CxT[int(cd["id"])][int(tr["therapy"])] = float(tr["successful"]) / 100
                                
                        #TO DO : MANAGE DOUBLES (average, weighted average)

    #handle duplicate entries inCxT
    for i in range(len(conditions)):
        for k in range(len(therapies)):
            if isinstance(CxT[i][k], np.ndarray):
                print("ciao")
                CxT = np.nanmean(CxT)

    newpatient = int(sys.argv[1])
    newcond = int(sys.argv[2])

    smallerUxT = np.array( [UxT[i] for i in lshUxT[UxT[newpatient]]] )
    print(f'LSH bucket dim={len(lshUxT[UxT[newpatient]])}, \n {smallerUxT}')


    #input nxm
    #nxk, k, kxm
    u, s, vh = randomized_svd(smallerUxT, n_components=20, n_iter=2, random_state=42)
    s = np.diag(s)
    q = np.dot(s, vh)
    # print(f'SVD\n{u}\n{s}\n{vh}')
    print(f'SVD\n{u.shape}\n{s.shape}\n{vh.shape}')
    print(f'SVD\n{u.shape}\n{q.shape}')

    # x_reconstr = np.dot(u, q)
    # nonzero_UxT = smallerUxT[smallerUxT.nonzero()]
    # nonzero_predictions = x_reconstr[smallerUxT.nonzero()]
    # mse = mean_squared_error(nonzero_UxT, nonzero_predictions)
    # print(mse)

    # print(smallerUxT[0],"\n",b[0])
    # for i in range(len(therapies)):
    #     if(smallerUxT[0][i] == 0):
    #         print(smallerUxT[0][i],"  ",b[0][i])

    num_latent_fact = 20
  
    
    # p_param = torch.from_numpy(u).float()
    # q_param = torch.from_numpy(q).float()

    # p_param.requires_grad = True
    # q_param.requires_grad = True
    p_param = torch.randn(smallerUxT.shape[0], num_latent_fact, requires_grad=True)
    q_param = torch.randn(num_latent_fact, smallerUxT.shape[1], requires_grad=True)

    target = torch.from_numpy(smallerUxT).float()

    def forward():
        return torch.mm(p_param, q_param)

    learning_rate = 1
    n_epochs = 20000

    def loss(y, y_pred):
        return ((y_pred - y)**2).mean()

    for epoch in range(n_epochs):

        prediction = forward()
        l = loss(target[target.nonzero(as_tuple=True)], prediction[target.nonzero(as_tuple=True)])

        print(f'loss={l}')
        l.backward()

        with torch.no_grad():
            p_param -= learning_rate * p_param.grad
            q_param -= learning_rate * q_param.grad
            # print(q_param.grad)
            p_param.grad.zero_()
            q_param.grad.zero_()

        if epoch == 10000:
            learning_rate = 0.1
        if epoch == 18000:
            learning_rate = 0.01
        
            
    
    with torch.no_grad():
        x_hat = torch.mm(p_param, q_param).detach().numpy()

    print(x_hat)
    print("\n\n")
    a = loss(smallerUxT[2][np.nonzero(smallerUxT[2])], x_hat[2][np.nonzero(smallerUxT[2])])
    print(f'{smallerUxT[2][np.nonzero(smallerUxT[2])]}\n{x_hat[2][np.nonzero(smallerUxT[2])]}')
    print(a)
    print("\n\n")
    a = loss(smallerUxT[5][np.nonzero(smallerUxT[5])], x_hat[5][np.nonzero(smallerUxT[5])])
    print(f'{smallerUxT[5][np.nonzero(smallerUxT[5])]}\n{x_hat[5][np.nonzero(smallerUxT[5])]}')
    print(a)
    print("\n\n")
    a = loss(smallerUxT[0][np.nonzero(smallerUxT[0])], x_hat[0][np.nonzero(smallerUxT[0])])
    print(f'{smallerUxT[0][np.nonzero(smallerUxT[0])]}\n{x_hat[0][np.nonzero(smallerUxT[0])]}')
    print(a)
if __name__=="__main__":
    main()