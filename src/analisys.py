
import json
from random import shuffle
from re import X
import numpy as np
from numpy.core.fromnumeric import shape
from sklearn.metrics import mean_squared_error, pairwise
from sklearn.model_selection import train_test_split
import sys

from torch._C import dtype

datapath = "../data/patients.json"

with open(datapath, "r") as fp:
    data = json.load(fp)

conditions = data["Conditions"]
therapies = data["Therapies"]
patients = data["Patients"]

UxT = np.empty((len(patients), len(therapies)))
UxT[:] = np.NaN
UxC = np.empty((len(patients), len(conditions)))
UxC[:] = np.NaN
CxT = np.empty((len(conditions), len(therapies)))
CxT[:] = np.NaN

#pandas works with dataframe[cols][rows]
print(f'len users={len(patients)}, len ther= {len(therapies)}, len cond= {len(conditions)}')
# print(f'UxT {UxT.shape} UxC {UxC.shape} CxT {CxT.shape}')

for pat in patients:

    # populate U x T
    for tr in pat["trials"]: #patient therapy matrix
        UxT[int(pat["id"])][int(tr["therapy"])] = float(tr["successful"]) / 100 #since i do /100 it doesnt need normalization
        
        # 2.01: WHAT HAPPENS IF I USE 1, 0s or 1, -1 ?
        #answer: it's just bad
        # if(int(tr["successful"]) >= 50):
        #     UxT[int(pat["id"])][int(tr["therapy"])] = 1.0
        # else:
        #     UxT[int(pat["id"])][int(tr["therapy"])] = -1.0

    # populate 
    for cd in pat["conditions"]: #patient condition matrix

        # UxC contains cured and uncured conditions
        UxC[int(pat["id"])][int(cd["id"])] = 1.0

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

print(f'UxT\n {UxT} \nUxC\n {UxC} \nCxT \n{CxT}')


# STANDARDIZATION (useless)
# mu = np.nanmean(CxT)
# std = np.nanstd(CxT)
# for i in range(len(conditions)):
#     for k in range(len(therapies)):
#         if ~np.isnan( CxT[i][k] ):
#             CxT[i][k] = (CxT[i][k] - mu) / std
# print(f'Standardization:\n {CxT}')

# mu = np.nanmean(UxT)
# std = np.nanstd(UxT)
# for i in range(len(patients)):
#     for k in range(len(therapies)):
#         if ~np.isnan( UxT[i][k] ):
#             UxT[i][k] = (UxT[i][k] - mu) / std
# print(f'Standardization:\n {UxT}')

# # indexes = UxT != 0 #synonym: np.nonzero(matrix)
# # mu = UxT[indexes].mean() #mean without nans
# #np.nan - mean without nans


np.nan_to_num(UxT, copy=False, nan=0.0)
muUxT = np.nanmean(UxT)
thmeanUxT = np.nanmean(UxT, axis= 0)
usrmeanUxT = np.nanmean(UxT, axis= 1)

np.nan_to_num(muUxT, copy=False, nan=0.0)
np.nan_to_num(thmeanUxT, copy=False, nan=0.0)
np.nan_to_num(usrmeanUxT, copy=False, nan=0.0)
# UxT.fillna(0)

# print(f'UxT:\n mu= {mu:.6f}, thmu = {thmu[0]:.6f}, thmulen= {len(thmu)}, usrmu= {usrmu[0]:.6f}, usrlen =  {len(usrmu)}')

np.nan_to_num(CxT, copy=False, nan=0.0)
# muCxT = np.nanmean(CxT)
# thmeanCxT = np.nanmean(CxT, axis= 0)
# condmeanCxT = np.nanmean(CxT, axis=1)
#THEY CAN STILL CONTAIN NANS - DO THIS TO REPLACE THEM
# np.nan_to_num(muCxT, copy=False, nan=0.0)
# np.nan_to_num(thmeanCxT, copy=False, nan=0.0)
# np.nan_to_num(condmeanCxT, copy=False, nan=0.0)

# print(f'CxT:\n mu= {muCxT:.6f},\n thmu = {thmeanCxT},\n thmulen= {len(thmeanCxT)},\n condmu= {condmeanCxT},\n condmulen =  {len(condmeanCxT)}\n')


UxTuser_sim= pairwise.cosine_similarity(UxT)
UxTth_sim = pairwise.cosine_similarity(UxT.T)
# print(f'UxTuser_sim{UxTuser_sim.shape}:\n{UxTuser_sim} \n\n UxTth_sim{UxTth_sim.shape}=\n{UxTth_sim}')

# UxCcond_sim = pairwise.cosine_similarity(UxC)
# UxCth_sim = pairwise.cosine_similarity(UxC.T)
# print(f'CxTcond_sim{UxCcond_sim.shape}:\n{UxCcond_sim} \n\n UxCth_sim{UxCth_sim.shape}=\n{UxCth_sim}')

# CxTcond_sim = pairwise.cosine_similarity(CxT)
# CxTth_sim = pairwise.cosine_similarity(CxT.T)
# print(f'CxTcond_sim{CxTcond_sim.shape}:\n{CxTcond_sim} \n\n CxTth_sim{CxTth_sim.shape}=\n{CxTth_sim}')



# predict user UxT not included in data
# predictions = user_sim.dot(UxT) / np.array( [np.abs(user_sim).sum(axis=1)] ).T
# np.nan_to_num(predictions, copy=False, nan=0.0)
# print(predictions)

# nprattrain, nprattest = train_test_split(UxT, test_size=0.5, shuffle=False)
# user_sim= pairwise.cosine_similarity(nprattrain)
# test_pred = user_sim.dot(UxT) / np.array( [np.abs(user_sim).sum(axis=1)] ).T
# print(test_pred, test_pred.shape)

# nonzero_UxT = UxT[UxT.nonzero()]
# nonzero_predictions = predictions[UxT.nonzero()]

# mse = mean_squared_error(nonzero_UxT, nonzero_predictions)
# print(mse)

# UxC = UxC.to_numpy()
# np.nan_to_num(UxC, copy=False, nan=0.0)
# # print(UxC[UxC.nonzero()])
# user_sim_UxC = pairwise.cosine_similarity(UxC)
# # print(f'user UxC similarity {user_sim_UxC.shape}:\n {user_sim_UxC}')

# estimate = baselineusr + np.dot(UxT.T[newpatient], th_sim[newth]) / np.sum(th_sim[newth])
# print(f'\nuser {newpatient} th {newth} estimate: {estimate}:\n {np.dot(UxT.T[newpatient], th_sim[newth])}/{np.sum(th_sim[newth])}')

newpatient = int(sys.argv[1])
newcond = int(sys.argv[2])

# print(f'{CxT[newcond]}\n{len(CxT[newcond])}')
k = 10 #find the indices of the k max values in array - It works https://www.kite.com/python/answers/how-to-find-the-n-maximum-indices-of-a-numpy-array-in-python
k = np.count_nonzero(CxT[newcond]) #define neighbours as the tested therapies for the given condition

idx=np.argpartition(CxT[newcond], len(CxT[newcond]) - k)[-k:]
indices = idx[np.argsort((-CxT[newcond])[idx])]

#give out best therapies! - baseline from UxT
# base_est = muUxT + usrmeanUxT[newpatient] - muUxT + thmeanUxT[i] - muUxT

#cosine pred with values in U x T
# r_hat(x, i) = b(x, i) + sum( sim(i, j) * (UxT(x, j) - b(x, j)) ) / sum( sim(i, j) ) 
# cosine_predictions = UxTth_sim.dot(UxT.T) / np.array( [np.abs(UxTth_sim).sum(axis=0)] ).T

#cosine pred with values in C x T - let's tryyy
# cosine_predictions = CxTth_sim.dot(UxT.T) / np.array( [np.abs(CxTth_sim).sum(axis=0)] ).T

# cosine_predictions = cosine_predictions.T # because i want patients on the rows (dim: U x T)

est = []

mse_est_pred = []

for i in indices: # for all the therapies that have been used the input condition
    
    if(UxT[newpatient][i] == 0): # for the therapies those therapies that the patient hasn't tried yet
    # if(True):

        #item based
        # cur_pred = thmeanUxT[i] + np.sum((UxT[newpatient] - thmeanUxT) * UxTth_sim[i]) / np.sum(np.abs(UxTth_sim[i]))
        #user based
        cur_pred = usrmeanUxT[newpatient] + (np.sum((UxT.T[i] - usrmeanUxT) * UxTuser_sim[newpatient]) / np.sum(np.abs(UxTuser_sim[newpatient])))
        est.append( (i, cur_pred) )
        mse_est_pred.append(cur_pred)

    else:
        #TO DO: MODEL INTERACTION WITH OLD ratings! RIGHT NOW OLD JUST SUBSTITUTE predictions
        est.append((i, UxT[newpatient][i]))

        # mse_est_base.append( UxT[newpatient][i])
        mse_est_pred.append( UxT[newpatient][i])

est.sort(key=lambda tup: tup[1], reverse=True) # sort by baseline rating

k = 5
print(f'\n{k} best therapies for {patients[newpatient]["name"]},(id={patients[newpatient]["id"]}) for {conditions[newcond]["name"]}(id={conditions[newcond]["id"]}):\n')
for count, (i, val) in enumerate(est[:k]):
    print(f'{count} -> {therapies[i]["name"]} ->{val}\n')
    
    # print(CxT[newcond], f'mean CxT{i}: {CxT[newcond].mean(), condmeanCxT[newcond]}')

# nonzero_UxT = UxT[UxT.nonzero()]
# nonzero_predictions = predictions[UxT.nonzero()]

# mse = mean_squared_error(nonzero_UxT, nonzero_predictions)

#  c = [a[index] for index in b] GET SUBLIST OF A LIST FROM LIST OF INDEXES
# numpy.where(x == 0)[0] indices of the elements that are zero

# correct mse (i need compare methods):
# mse_est_pred2 = []
# # nonzero_i = np.intersect1d(indices, np.nonzero(UxT[newpatient]))
# nonzero_i = np.nonzero(UxT[newpatient])
# for th_i in nonzero_i:
#     #user based
#     cur_pred = usrmeanUxT[newpatient] + (np.sum((UxT.T[th_i] - usrmeanUxT) * UxTuser_sim[newpatient]) / np.sum(np.abs(UxTuser_sim[newpatient])))
#     mse_est_pred2.append(cur_pred)

#     #item based
#     # cur_pred = thmeanUxT[th_i] + np.sum((UxT[newpatient] - thmeanUxT) * UxTth_sim[th_i]) / np.sum(np.abs(UxTth_sim[th_i]))
#     # mse_est_pred2.append(cur_pred)

MSE_pred = np.sqrt( mean_squared_error(mse_est_pred, [UxT[newpatient][app] for app in indices]))

# print(f'predictions = {mse_est_pred}\nreal values= {[UxT[newpatient][app] for app in indices]}')
print(f'MSE_pred={MSE_pred}')


# # provo a rifare le predizioni con ML
# # r_hat(x, i) = b(x, i) + sum( w(i, j) * (r(x, j) - b(x, j)) )
# # r_hat(x, i) = b(x, i) + dot(w(i), r(x) - b(x))
# import torch
# import torch.nn as nn
# import numpy as np
# import math
# from torch.nn.modules.module import Module
# from torch.utils.data import Dataset, DataLoader

# # step 1: targets are the set of nonzero therapies for the current patient
# # UxT[new patient] = [ r 0 0 0 r r r 0 r ... 0 0 r 0] (r = rating)

# # rated_ths = UxT[newpatient][np.nonzero(UxT[newpatient])]
# # y = torch.from_numpy(rated_ths).float()

# y = torch.from_numpy(UxT[newpatient]).float()

# #r(x,j) - b(x,j)
# # biases = []
# # # for i in np.nonzero(UxT[newpatient]):
# # for i in range(len(UxT[newpatient])): 
# #     biases.append(usrmeanUxT[newpatient] + thmeanUxT[i] - muUxT)
# # biases = np.asarray_chkfinite(biases)

# # # [ r-b 0 0 0 r-b r-b r-b ... 0]
# # x = UxT[newpatient]
# # for ind in range(len(therapies)):
# #     if x[ind] > 0:
# #         x[ind] = x[ind] - biases[ind]

# #itembased
# # [ r-b 0 0 0 r-b r-b r-b ... 0]
# x = UxT[newpatient] - thmeanUxT
# x = torch.from_numpy(x).float()


# # x = torch.reshape(x, (-1, 1))
# y = torch.reshape(y, (-1, 1))

# class LinearRegression(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LinearRegression, self).__init__()
#         # self.linear = nn.Linear(input_dim, output_dim)
#         self.linear = nn.Linear(input_dim, output_dim)
        

#     def forward(self, x, i):
#         # return self.linear(x)
        
#         return thmeanUxT[i] + self.linear(x) # itembased
#         # return usrmeanUxT[newpatient] + self.linear(x) # userbased

# model = LinearRegression(len(therapies), 1) #item based
# # model = LinearRegression(len(patients), 1) #user based

# learning_rate = 0.00001
# iter = 100

# loss = nn.MSELoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# for epoch in range(iter):
#     for i, cury in enumerate(y):
#         # print(i, cury)
#         if cury > 0:
            
#             #usrbased
#             # x = UxT.T[i] - usrmeanUxT
#             # x = torch.from_numpy(x).float()

#             y_pred = model(x, i)
            
#             # print(cury, y_pred)
#             tmp_loss = loss(cury, y_pred)
#             print(f'y={cury}, pred={y_pred}, loss={tmp_loss}')

#             tmp_loss.backward()

#             optimizer.step()

#             optimizer.zero_grad()

# with torch.no_grad():
#     mse_est_predML = []
#     nonzero_i = np.nonzero(UxT[newpatient])
#     x = UxT[newpatient] - thmeanUxT
#     x = torch.from_numpy(x).float()
#     for i in indices:
#         # x = UxT.T[i] - usrmeanUxT
#         # x = torch.from_numpy(x).float()

#         y_pred = model(x, i)
#         mse_est_predML.append(y_pred.detach().numpy())

#     MSE_predML = np.sqrt( mean_squared_error(mse_est_predML, [UxT[newpatient][app] for app in indices]))

#     print(f'predictionsML = {mse_est_predML}\nreal values= {[UxT[newpatient][app] for app in indices]}')
#     print(f'MSE_predML={MSE_predML}')
