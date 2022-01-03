import json
from random import shuffle
import pandas as pd
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

ratings = pd.DataFrame(index=range(len(patients)), columns=range(len(therapies)), dtype=np.float32 )
history = pd.DataFrame(index=range(len(patients)), columns=range(len(conditions)), dtype= np.float32)

#pandas works with dataframe[cols][rows]
print(f'len users={len(patients)}, len ther= {len(therapies)}, len cond= {len(conditions)}')


for pat in patients:
    id = int(pat["id"])

    for tr in pat["trials"]: #patient therapy matrix

        th = int(tr["therapy"]) #get the therapy id
        ratings.loc[th][id] = float(tr["successful"]) / 100

    for cd in pat["conditions"]: #patient condition matrix

        cond = int(cd["id"])
        history.loc[cond][id] = 1.0

# ratings.loc[2][1] = 0.1

#dataframe.to_numpy()
npratings = ratings.to_numpy()

#np.nan
mu = np.nanmean(npratings)
thmu = np.nanmean(npratings, axis= 0)
usrmu = np.nanmean(npratings, axis= 1)
# indexes = npratings != 0 #synonym: np.nonzero(matrix)
# mu = npratings[indexes].mean()

print(f'mu= {mu:.6f}, thmu = {thmu[0]:.6f}, thmulen= {len(thmu)}, usrmu= {usrmu[0]:.6f}, usrlen =  {len(usrmu)}')

np.nan_to_num(npratings, copy=False, nan=0.0)
np.nan_to_num(thmu, copy=False, nan=0.0)
np.nan_to_num(usrmu, copy=False, nan=0.0)
ratings.fillna(0)
# print(ratings)
# print(npratings)
print(f'usrmu {usrmu.shape}=\n{usrmu}\nthmu {thmu.shape}=\n{thmu}')

# print(history)

user_sim= pairwise.cosine_similarity(npratings)
th_sim = pairwise.cosine_similarity(npratings.T)
# print(user_sim, th_sim)
print(f'user similaty{user_sim.shape}:\n{user_sim} \n\n itemsim{th_sim.shape}=\n{th_sim}')

# predict user ratings not included in data
predictions = user_sim.dot(npratings) / np.array( [np.abs(user_sim).sum(axis=1)] ).T
np.nan_to_num(predictions, copy=False, nan=0.0)
# print(predictions)

# nprattrain, nprattest = train_test_split(npratings, test_size=0.5, shuffle=False)
# user_sim= pairwise.cosine_similarity(nprattrain)
# test_pred = user_sim.dot(npratings) / np.array( [np.abs(user_sim).sum(axis=1)] ).T
# print(test_pred, test_pred.shape)

nonzero_ratings = npratings[npratings.nonzero()]
nonzero_predictions = predictions[npratings.nonzero()]

mse = mean_squared_error(nonzero_ratings, nonzero_predictions)
# print(mse)

nphistory = history.to_numpy()
np.nan_to_num(nphistory, copy=False, nan=0.0)
# print(nphistory[nphistory.nonzero()])
user_sim_history = pairwise.cosine_similarity(nphistory)
# print(f'user history similarity {user_sim_history.shape}:\n {user_sim_history}')

#baseline estimates for the user
newpatient = int(sys.argv[1])
# newcond = int(sys.argv[2])
newth = int(sys.argv[2]) #let's pretend we a therapy
baselineusr = mu + usrmu[newpatient] 
print(f'base line evaluation: {baselineusr}')

estimate = baselineusr + np.dot(npratings.T[newpatient], th_sim[newth]) / np.sum(th_sim[newth])
print(f'\nuser {newpatient} th {newth} estimate: {estimate}:\n {np.dot(npratings.T[newpatient], th_sim[newth])}/{np.sum(th_sim[newth])}')




