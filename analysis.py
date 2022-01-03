import json
from random import shuffle
import numpy as np
from sklearn.metrics import mean_squared_error, pairwise
from sklearn.model_selection import train_test_split
import sys

datasetpath = "./dataset/patients.json"

with open(datasetpath, "r") as fp:
    dataset = json.load(fp)

conditions = dataset["Conditions"]
therapies = dataset["Therapies"]
patients = dataset["Patients"]

UxT = np.empty((len(patients), len(therapies)))
UxT[:] = np.NaN
UxC = np.empty((len(patients), len(conditions)))
UxC[:] = np.NaN
CxT = np.empty((len(conditions), len(therapies)))
CxT[:] = np.NaN

#pandas works with dataframe[cols][rows]
print(f'len users={len(patients)}, len ther= {len(therapies)}, len cond= {len(conditions)}')
print(f'UxT {UxT.shape} UxC {UxC.shape} CxT {CxT.shape}')

for pat in patients:

    # populate U x T
    for tr in pat["trials"]: #patient therapy matrix
        UxT[int(pat["id"])][int(tr["therapy"])] = float(tr["successful"]) / 100 #since i do /100 it doesnt need normalization

    # populate 
    for cd in pat["conditions"]: #patient condition matrix

        # UxC contains cured and uncured conditions
        UxC[int(pat["id"])][int(cd["id"])] = 1.0

        # C x T contains ratings of therapies for cured conditions only
        if cd["cured"] == True:

            for tr in pat["trials"]: #for each cured condition
                if tr["condition"] == cd["id"]: #insert the rating for the therapies that were tried

                    if ~np.isnan(CxT[int(cd["id"])][int(tr["therapy"])]):
                        CxT[int(cd["id"])][int(tr["therapy"])] = float(tr["successful"]) / 100

                    else:
                        CxT[int(cd["id"])][int(tr["therapy"])] = float(tr["successful"]) / 100
                    #TO DO : MANAGE DOUBLES (average, weighted average)

print(f'UxT\n {UxT} \nUxC\n {UxC} \nCxT \n{CxT}')

npUxT = UxT
npUxC = UxC
npCxT = CxT

# STANDARDIZATION (useless)
# mu = np.nanmean(npCxT)
# std = np.nanstd(npCxT)
# for i in range(len(conditions)):
#     for k in range(len(therapies)):
#         if ~np.isnan( npCxT[i][k] ):
#             npCxT[i][k] = (npCxT[i][k] - mu) / std
# print(f'Standardization:\n {npCxT}')

# mu = np.nanmean(npUxT)
# std = np.nanstd(npUxT)
# for i in range(len(patients)):
#     for k in range(len(therapies)):
#         if ~np.isnan( npUxT[i][k] ):
#             npUxT[i][k] = (npUxT[i][k] - mu) / std
# print(f'Standardization:\n {npUxT}')

# # indexes = npUxT != 0 #synonym: np.nonzero(matrix)
# # mu = npUxT[indexes].mean() #mean without nans
# #np.nan - mean without nans

muUxT = np.nanmean(npUxT)
thmeanUxT = np.nanmean(npUxT, axis= 0)
usrmeanUxT = np.nanmean(npUxT, axis= 1)

np.nan_to_num(npUxT, copy=False, nan=0.0)
np.nan_to_num(thmeanUxT, copy=False, nan=0.0)
np.nan_to_num(usrmeanUxT, copy=False, nan=0.0)
np.nan_to_num(npUxT, copy=False, nan=0.0)
# UxT.fillna(0)

# print(f'UxT:\n mu= {mu:.6f}, thmu = {thmu[0]:.6f}, thmulen= {len(thmu)}, usrmu= {usrmu[0]:.6f}, usrlen =  {len(usrmu)}')

# muCxT = np.nanmean(npCxT)
# thmeanCxT = np.nanmean(npCxT, axis= 0)
# condmeanCxT = np.nanmean(npCxT, axis=1)
#THEY CAN STILL CONTAIN NANS - DO THIS TO REPLACE THEM
# np.nan_to_num(muCxT, copy=False, nan=0.0)
# np.nan_to_num(thmeanCxT, copy=False, nan=0.0)
# np.nan_to_num(condmeanCxT, copy=False, nan=0.0)
np.nan_to_num(npCxT, copy=False, nan=0.0)

# print(f'CxT:\n mu= {muCxT:.6f},\n thmu = {thmeanCxT},\n thmulen= {len(thmeanCxT)},\n condmu= {condmeanCxT},\n condmulen =  {len(condmeanCxT)}\n')



# UxTuser_sim= pairwise.cosine_similarity(npUxT)
# UxTth_sim = pairwise.cosine_similarity(npUxT.T)
# print(f'UxTuser_sim{UxTuser_sim.shape}:\n{UxTuser_sim} \n\n UxTth_sim{UxTth_sim.shape}=\n{UxTth_sim}')

# UxCcond_sim = pairwise.cosine_similarity(npUxC)
# UxCth_sim = pairwise.cosine_similarity(npUxC.T)
# print(f'CxTcond_sim{UxCcond_sim.shape}:\n{UxCcond_sim} \n\n UxCth_sim{UxCth_sim.shape}=\n{UxCth_sim}')

# CxTcond_sim = pairwise.cosine_similarity(npCxT)
# CxTth_sim = pairwise.cosine_similarity(npCxT.T)
# print(f'CxTcond_sim{CxTcond_sim.shape}:\n{CxTcond_sim} \n\n CxTth_sim{CxTth_sim.shape}=\n{CxTth_sim}')



# predict user UxT not included in data
# predictions = user_sim.dot(npUxT) / np.array( [np.abs(user_sim).sum(axis=1)] ).T
# np.nan_to_num(predictions, copy=False, nan=0.0)
# print(predictions)

# nprattrain, nprattest = train_test_split(npUxT, test_size=0.5, shuffle=False)
# user_sim= pairwise.cosine_similarity(nprattrain)
# test_pred = user_sim.dot(npUxT) / np.array( [np.abs(user_sim).sum(axis=1)] ).T
# print(test_pred, test_pred.shape)

# nonzero_UxT = npUxT[npUxT.nonzero()]
# nonzero_predictions = predictions[npUxT.nonzero()]

# mse = mean_squared_error(nonzero_UxT, nonzero_predictions)
# print(mse)

# npUxC = UxC.to_numpy()
# np.nan_to_num(npUxC, copy=False, nan=0.0)
# # print(npUxC[npUxC.nonzero()])
# user_sim_UxC = pairwise.cosine_similarity(npUxC)
# # print(f'user UxC similarity {user_sim_UxC.shape}:\n {user_sim_UxC}')

newpatient = int(sys.argv[1])
newcond = int(sys.argv[2])

# estimate = baselineusr + np.dot(npUxT.T[newpatient], th_sim[newth]) / np.sum(th_sim[newth])
# print(f'\nuser {newpatient} th {newth} estimate: {estimate}:\n {np.dot(npUxT.T[newpatient], th_sim[newth])}/{np.sum(th_sim[newth])}')

newpatient = int(sys.argv[1])
newcond = int(sys.argv[2])

print(f'{npCxT[newcond]}\n{len(npCxT[newcond])}')
k = 5 #find the indices of the k max values in array - It works https://www.kite.com/python/answers/how-to-find-the-n-maximum-indices-of-a-numpy-array-in-python
idx=np.argpartition(npCxT[newcond], len(npCxT[newcond]) - k)[-k:]
indices = idx[np.argsort((-npCxT[newcond])[idx])]
# print(indices)
# for i in indices:
#     print(npCxT[newcond][i])

#give out best therapies! - baseline from UxT
# base_est = muUxT + usrmeanUxT[newpatient] - muUxT
base_est = usrmeanUxT[newpatient]
print(f'\n{k} best therapies for {conditions[newcond]["name"]}\n')
for i in indices:
    print(f'i:{i} -> {therapies[i]["name"]} ->{base_est + thmeanUxT[i] - muUxT}\n')



