import json
from random import shuffle
import numpy as np
from numpy.core.fromnumeric import shape
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
                    
                    if isinstance(CxT[int(cd["id"])][int(tr["therapy"])], np.ndarray):
                        np.append(CxT[int(cd["id"])][int(tr["therapy"])], float(tr["successful"]) / 100)

                    else:
                        if ~np.isnan(CxT[int(cd["id"])][int(tr["therapy"])]):

                            CxT[int(cd["id"])][int(tr["therapy"])] = np.array( CxT[int(cd["id"])][int(tr["therapy"])] )
                            np.append(CxT[int(cd["id"])][int(tr["therapy"])], float(tr["successful"]) / 100)
                        
                        else:
                            CxT[int(cd["id"])][int(tr["therapy"])] = float(tr["successful"]) / 100
                    #TO DO : MANAGE DOUBLES (average, weighted average)

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

muUxT = np.nanmean(UxT)
thmeanUxT = np.nanmean(UxT, axis= 0)
usrmeanUxT = np.nanmean(UxT, axis= 1)

np.nan_to_num(UxT, copy=False, nan=0.0)
np.nan_to_num(thmeanUxT, copy=False, nan=0.0)
np.nan_to_num(usrmeanUxT, copy=False, nan=0.0)
np.nan_to_num(UxT, copy=False, nan=0.0)
# UxT.fillna(0)

# print(f'UxT:\n mu= {mu:.6f}, thmu = {thmu[0]:.6f}, thmulen= {len(thmu)}, usrmu= {usrmu[0]:.6f}, usrlen =  {len(usrmu)}')

# muCxT = np.nanmean(CxT)
# thmeanCxT = np.nanmean(CxT, axis= 0)
condmeanCxT = np.nanmean(CxT, axis=1)
#THEY CAN STILL CONTAIN NANS - DO THIS TO REPLACE THEM
# np.nan_to_num(muCxT, copy=False, nan=0.0)
# np.nan_to_num(thmeanCxT, copy=False, nan=0.0)
np.nan_to_num(condmeanCxT, copy=False, nan=0.0)
np.nan_to_num(CxT, copy=False, nan=0.0)

# print(f'CxT:\n mu= {muCxT:.6f},\n thmu = {thmeanCxT},\n thmulen= {len(thmeanCxT)},\n condmu= {condmeanCxT},\n condmulen =  {len(condmeanCxT)}\n')



# UxTuser_sim= pairwise.cosine_similarity(UxT)
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

newpatient = int(sys.argv[1])
newcond = int(sys.argv[2])

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

cosine_predictions = UxTth_sim.dot(UxT.T) / np.array( [np.abs(UxTth_sim).sum(axis=0)] ).T
cosine_predictions = cosine_predictions.T # because i want patients on the rows

est = []

mse_est_base = [] # dot product on neighbour set only (tested therapies for the input condition)
mse_est_pred = [] # dot product on whole matrix

for i in indices:
    
    if(UxT[newpatient][i] == 0):
    # if(True):

        bxi = usrmeanUxT[newpatient] + thmeanUxT[i] - muUxT
        
        # NEIGHBOURS ONLY DOT PRODUCT
        # cosine_est = 0
        # cos_sum = 0
        # for p in indices:
        #     if p != i:
        #         cos_sum = cos_sum + UxTth_sim[i][p]
        #         cosine_est = cosine_est + ( UxT[newpatient][p] * UxTth_sim[i][p] )
        # cosine_est = cosine_est / cos_sum # dot product / sum of similarities
        # est.append((i, bxi + cosine_est) )

        est.append( (i, bxi + cosine_predictions[newpatient][i]) )

        # print(cosine_est, cosine_predictions[newpatient][i])
        # mse_est_base.append(bxi + cosine_est)
        # mse_est_pred.append( bxi + cosine_predictions[newpatient][i])

    else:
        #TO DO: MODEL INTERACTION WITH OLD ESTIMATES! RIGHT NOW, OLD JUST SUBSTITUTE ESTIMATES
        est.append((i, UxT[newpatient][i]))

        # mse_est_base.append( UxT[newpatient][i])
        # mse_est_pred.append( UxT[newpatient][i])

est.sort(key=lambda tup: tup[1], reverse=True) #sort by baseline rating

k = 5
print(f'\n{k} best therapies for {conditions[newcond]["name"]}\n')
for (i, val) in est[:k]:
    print(f'{i} -> {therapies[i]["name"]} ->{val}\n')
    
    # print(CxT[newcond], f'mean CxT{i}: {CxT[newcond].mean(), condmeanCxT[newcond]}')

# print(UxT[newpatient])
# print(CxT[newcond])


# nonzero_UxT = UxT[UxT.nonzero()]
# nonzero_predictions = predictions[UxT.nonzero()]

# mse = mean_squared_error(nonzero_UxT, nonzero_predictions)


#  c = [a[index] for index in b] GET SUBLIST OF A LIST FROM LIST OF INDEXES
# MSE_base = mean_squared_error(mse_est_base, [UxT[newpatient][app] for app in indices])
# MSE_pred = mean_squared_error(mse_est_pred, [UxT[newpatient][app] for app in indices])
# print(f'MSE_base={MSE_base}, MSE_pred={MSE_pred}')