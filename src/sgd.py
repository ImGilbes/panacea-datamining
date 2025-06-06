import json
from random import shuffle
from re import M, T, X
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

# datapath = "../data/patients.json"
datapath = "../data/" + sys.argv[1]

cured = 'isCured'
# cured= 'cured'

# for lsh
num_hash_tables = 20
hash_dimension = 20

#clipping factor
c = 10

n_epochs = 50

num_latent_fact = 40

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
            UxT[int(pat["id"])][int(tr["therapy"][2:])-1] = float(tr["successful"]) / 100 #since i do /100 it doesnt need normalization

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
            if cd["isCured"] != "NULL":
                for tr in pat["trials"]: #for each cured condition

                    if tr["condition"][2:] == cd["id"][2:]: #insert the rating for the therapies that were tried
                        if isinstance(CxT[int(cd["kind"][4:])-1][int(tr["therapy"][2:])-1], np.ndarray):
                            np.append(CxT[int(cd["kind"][4:])-1][int(tr["therapy"][2:])-1], float(tr["successful"]) / 100)

                        else:
                            if ~np.isnan(CxT[int(cd["kind"][4:])-1][int(tr["therapy"][2:])-1]):

                                CxT[int(cd["kind"][4:])-1][int(tr["therapy"][2:])-1] = np.array( CxT[int(cd["kind"][4:])-1][int(tr["therapy"][2:])-1] )
                                np.append(CxT[int(cd["kind"][4:])-1][int(tr["therapy"][2:])-1], float(tr["successful"]) / 100)
                            
                            else:
                                CxT[int(cd["kind"][4:])-1][int(tr["therapy"][2:])-1] = float(tr["successful"]) / 100
                                
                        #TO DO : MANAGE DOUBLES (average, weighted average)

    #handle duplicate entries inCxT
    for i in range(len(conditions)):
        for k in range(len(therapies)):
            if isinstance(CxT[i][k], np.ndarray):
                CxT[i][k] = np.nanmean(CxT[i][k])

    newpatient = int(sys.argv[2])
    if sys.argv[2] != "../data/patients.json":
        # newcond = int(sys.argv[3][2:])
        for cond in patients[int(sys.argv[2])]["conditions"]:
            if cond["id"] == sys.argv[3]:
                newcond = int(cond["kind"][4:]) - 1
    else:
        newcond = int(sys.argv[3])
    # newcond = int(sys.argv[3])

    smallerUxT = np.array( [UxT[i] for i in lshUxT[UxT[newpatient]]] )
    print(f'LSH bucket dim={len(lshUxT[UxT[newpatient]])}, \n {smallerUxT}')

    #input nxm
    #nxk, k, kxm
    # u, s, vh = randomized_svd(smallerUxT, n_components=20, n_iter=2, random_state=42)
    # s = np.diag(s)
    # q = np.dot(s, vh)
    # # print(f'SVD\n{u}\n{s}\n{vh}')
    # print(f'SVD\n{u.shape}\n{s.shape}\n{vh.shape}')
    # print(f'SVD\n{u.shape}\n{q.shape}')

    # x_reconstr = np.dot(u, q)
    # nonzero_UxT = smallerUxT[smallerUxT.nonzero()]
    # nonzero_predictions = x_reconstr[smallerUxT.nonzero()]
    # mse = mean_squared_error(nonzero_UxT, nonzero_predictions)
    # print(mse)

    # print(smallerUxT[0],"\n",b[0])
    # for i in range(len(therapies)):
    #     if(smallerUxT[0][i] == 0):
    #         print(smallerUxT[0][i],"  ",b[0][i])

    
    # p_param = torch.from_numpy(u).float()
    # q_param = torch.from_numpy(q).float()

    # p_param.requires_grad = True
    # q_param.requires_grad = True

    # p_param = torch.randn(smallerUxT.shape[0], num_latent_fact, requires_grad=True)
    # q_param = torch.randn(num_latent_fact, smallerUxT.shape[1], requires_grad=True)

    p_param = torch.rand(smallerUxT.shape[0], num_latent_fact, requires_grad=True)
    q_param = torch.rand(num_latent_fact, smallerUxT.shape[1], requires_grad=True)

    def loss(y, y_pred):
        return ((y_pred - y)**2).mean()

    learning_rate = 0.01

    i_usr_pred, i_item_pred = np.nonzero(smallerUxT)

    muUxT = np.array(smallerUxT[np.nonzero(smallerUxT)].mean())
    muUxT = torch.from_numpy(muUxT).float()
    bu = torch.zeros(len(patients), requires_grad=True)
    bi = torch.zeros(len(therapies), requires_grad=True)

    for epoch in range(n_epochs):
        for (i, j) in list(zip(i_usr_pred, i_item_pred)):
            
            app = np.array(smallerUxT[i][j])
            target = torch.from_numpy(app).float()

            #p = n x k, q = k x m, matrice iniziale = n x m
            prediction = torch.dot(p_param[i], torch.t(q_param)[j])
            prediction = prediction + muUxT + bu[i] + bi[j]

            error = loss(target, prediction)

            error.backward()
            
            # print(f'loss={error}')

            with torch.no_grad():
                if (torch.norm(p_param.grad[i]) > c):
                    p_param.grad[i] = c*p_param.grad[i]/torch.norm(p_param.grad[i])
                if (torch.norm(torch.t(q_param.grad)[j]) > c):
                    torch.t(q_param.grad)[j] = c*torch.t(q_param.grad)[j]/torch.norm(torch.t(q_param.grad)[j])
                p_param[i] -= learning_rate * p_param.grad[i]
                torch.t(q_param)[j] -= learning_rate * torch.t(q_param.grad)[j]
                p_param.grad.zero_()
                q_param.grad.zero_()

                #update bias
                bu[i] -= learning_rate * bu.grad[i]
                bi[j] -= learning_rate * bi.grad[j]
                bu.grad.zero_()
                bi.grad.zero_()

    with torch.no_grad():
        uxt_hat = torch.mm(p_param, q_param)
        for i in range(smallerUxT.shape[0]):
            for j in range(smallerUxT.shape[1]):
                uxt_hat[i][j] = uxt_hat[i][j] + muUxT + bu[i] + bi[j]
        print(f'\n\n\nloss={loss(smallerUxT[np.nonzero(smallerUxT)], uxt_hat[np.nonzero(smallerUxT)])}\n\n')
        smallerUxT = uxt_hat.detach().numpy()
        # l = loss(target[target.nonzero(as_tuple=True)], prediction[target.nonzero(as_tuple=True)])

    #lsh on CxT
    for row in range(CxT.shape[0]):
        np.nan_to_num(CxT[row], copy=False, nan=0.0)
        lshCxT[CxT[row]] = row
    
    smallerCxT = np.array( [CxT[i-1] for i in lshCxT[CxT[newcond]]] )
    print(f'LSH bucket dim={len(lshCxT[CxT[newcond]])}, \n {smallerCxT}')

    #indice di newpatient/cond on the new UxT table -> smaller UxT
    newpatient_index = lshUxT[UxT[newpatient]].index(newpatient)
    newcond_index = lshCxT[CxT[newcond]].index(newcond)
    # print(newpatient_index, "  ", newcond_index)

    # SVD on C x T
    p_param = torch.rand(smallerCxT.shape[0], num_latent_fact, requires_grad=True)
    q_param = torch.rand(num_latent_fact, smallerCxT.shape[1], requires_grad=True)

    learning_rate = 0.01

    i_usr_pred, i_item_pred = np.nonzero(smallerCxT)

    app = np.array(smallerCxT[np.nonzero(smallerCxT)].mean())
    muCxT = torch.from_numpy(app).float()
    bu = torch.zeros(len(conditions), requires_grad=True)
    bi = torch.zeros(len(therapies), requires_grad=True)

    for epoch in range(n_epochs):
        for (i, j) in list(zip(i_usr_pred, i_item_pred)):
            
            app = np.array(smallerCxT[i][j])
            target = torch.from_numpy(app).float()

            prediction = torch.dot(p_param[i], torch.t(q_param)[j])
            prediction = prediction + muCxT + bu[i] + bi[j]

            error = loss(target, prediction)

            error.backward()

            with torch.no_grad():
                if (torch.norm(p_param.grad[i]) > c):
                    p_param.grad[i] = c*p_param.grad[i]/torch.norm(p_param.grad[i])
                if (torch.norm(torch.t(q_param.grad)[j]) > c):
                    torch.t(q_param.grad)[j] = c*torch.t(q_param.grad)[j]/torch.norm(torch.t(q_param.grad)[j])
                p_param[i] -= learning_rate * p_param.grad[i]
                torch.t(q_param)[j] -= learning_rate * torch.t(q_param.grad)[j]
                p_param.grad.zero_()
                q_param.grad.zero_()

                #update bias
                bu[i] -= learning_rate * bu.grad[i]
                bi[j] -= learning_rate * bi.grad[j]
                bu.grad.zero_()
                bi.grad.zero_()

    with torch.no_grad():
        cxt_hat = torch.mm(p_param, q_param)
        for i in range(smallerCxT.shape[0]):
            for j in range(smallerCxT.shape[1]):
                cxt_hat[i][j] = cxt_hat[i][j] + muCxT + bu[i] + bi[j]
        print(f'\n\n\nloss={loss(smallerCxT[np.nonzero(smallerCxT)], cxt_hat[np.nonzero(smallerCxT)])}\n\n')
        smallerCxT = cxt_hat.detach().numpy()

    usrsim_UxT= pairwise.cosine_similarity(smallerUxT)
    # thsim_UxT= pairwise.cosine_similarity(smallerUxT.T)

    condsim_CxT= pairwise.cosine_similarity(smallerCxT)
    # thsim_CxT= pairwise.cosine_similarity(smallerCxT.T)


    res = []
    for th_i in range(len(therapies)):
        score = np.sum(usrsim_UxT[newpatient_index] * smallerUxT.T[th_i]) / np.sum(usrsim_UxT[newpatient_index]) + np.sum(condsim_CxT[newcond_index] * smallerCxT.T[th_i]) / np.sum(condsim_CxT[newcond_index])
        # print(score)
        res.append((th_i, score))
    res.sort(key=lambda tup: tup[1], reverse=True)
    k = 5
    for count, (i, val) in enumerate(res[:k]):
        print(f'#{count} -> {therapies[i]["name"]} -> score: {val:.5f}\n')


    #questa roba è malvagia, fa porprio schifo
    # k = 10 #find the indices of the k max values in array - It works https://www.kite.com/python/answers/how-to-find-the-n-maximum-indices-of-a-numpy-array-in-python
    # idx=np.argpartition(smallerCxT[newcond_index], len(smallerCxT[newcond_index]) - k)[-k:]
    # bestth_i = idx[np.argsort((-smallerCxT[newcond_index])[idx])]

    # res = []
    # for i in bestth_i:
    #     # if smallerUxT[newpatient_index][i] > 1:
    #     #     smallerUxT[newpatient_index][i] = 0.99
    #     res.append( (i, smallerUxT[newpatient_index][i]) )
    # res.sort(key=lambda tup: tup[1], reverse=True)

    # print(f'\n{patients[newpatient]["name"]}, condition: {conditions[newcond]["name"]}\n')
    # k = 5
    # for count, (i, val) in enumerate(res[:k]):
    #     print(f'#{count} -> {therapies[i]["name"]} ->{val:.5f}\n')


if __name__=="__main__":
    main()

# z -> M(i, j)
# for each z in M
#     z = (z - min(M)) / (max(M) - min(M))
# https://developers.google.com/machine-learning/data-prep/transform/normalization
# https://stats.stackexchange.com/questions/12200/normalizing-variables-for-svd-pca

# punteggio = sum(usrsimUxT[user] * smallerUxT.T[therapy]) / sum(usrsimUxT[user]) + sum(condsimCxT[cond] * smallerCxT.T[therapy]) / sum(condsimCxT[cond])

# m(x, i) -> global_mean + bu[x] + bi[i]
# m.shape = smallerUxT
# m -> pytorch tensor , gradient required