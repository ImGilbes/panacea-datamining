import json
from json import encoder
from names_data import NameDatasetV1 # v1
from random import seed
from random import choice
from random import randint

#script constants
conditions_file = "data/conditions.json"
therapies_file = "data/therapies.json"
merged_file = "data/patients.json"

maxcond = 10
maxtrials = 5 #lets keep this low atm

with open(conditions_file, "r") as fp:
    condlist = json.load(fp)

with open(therapies_file, "r") as fp:
    thlist = json.load(fp)

#returns a string YY-MM-DD
def random_date():
    y = randint(1980, 2020)
    m = randint(1, 12)

    maxday = 31
    if(m == 2):
        if(y % 4 == 0):
            maxday = 29
        else:
            maxday = 28
    elif (m == 4 or m == 6 or m == 7 or m == 9 or m == 11):
        maxday = 30

    d = randint(1, maxday)

    return str(str(y)+"-"+str(m)+"-"+str(d))

# random date in yy-mm-dd str format with year limits
def random_date_lim(a, b = 2020):
    if a <= b:
        y = randint(a, b)
        m = randint(1, 12)

        maxday = 31
        if(m == 2):
            if(y % 4 == 0):
                maxday = 29
            else:
                maxday = 28
        elif (m == 4 or m == 6 or m == 7 or m == 9 or m == 11):
            maxday = 30

        d = randint(1, maxday)

        return str(str(y)+"-"+str(m)+"-"+str(d))
    else:
        return random_date()


def date_to_int_list (d):#converts date in a yy-mm-dd str format into a list of 3 elements

    d = d.split("-")

    for i in range(3):
        d[i] = int(d[i])

    return d

#dates are still in the string yy-mm-dd format
def is_older_then(a, b): #returns True if a older than b, False otherwise
    ret = False
    a = date_to_int_list(a)
    b = date_to_int_list(b)

    if a[0] < b[0]: #compare years
        ret = True

    elif a[0] == b[0]:
        if a[1] < b[1]: #compare months

            ret = True

        elif a[1] ==  b[1]:
            if a[2] < b[2]: #compare days

                ret = True

    return ret


# trial = {
#     id
#     start, end
#     therapyid, condition id
#     successfull
# }
#generates the trials of a patient for a given condition (that has been chones randomly)
def generate_trials(condition, diagnosed, id):#contition is the condition as a dictionary, diagnosed is a str yy-mm-dd date
    trialid = id
    diagnosed = date_to_int_list(diagnosed)
    trials = []

    for _ in range(randint(0, maxtrials)):

        newth = choice(thlist) #i leave the code able to generate the same therapy for the same condition (or even for different conditions) more then once

        start = random_date_lim(diagnosed[0], diagnosed[0] + 2) #up to two years after diagnosys

        app = date_to_int_list(start)
        end = random_date_lim(app[0])

        successful = randint (0, 100)

        thid = newth["id"]
        condid = condition["id"]

        trials.append( {"id":str(trialid), "condition":condition["id"], "therapy": newth["id"], "start": start, "end": end, "successful": successful} )

        trialid = trialid + 1
    return trials

def has_been_cured(trials):
    ret = False
    
    if len(trials) > 0:
        for t in trials:
            if t["successful"] >= 30: #considero solo le terapie almeno buone per il 30%
                if randint(0,100) < t["successful"]:
                    ret = True

    return ret


#GENERATE THE NEW PATIENT FOR THE HOSPITAL!
def main():
    m = NameDatasetV1()
    n = list(m.first_names)
    ln = list(m.last_names)

    #
    patients = []
    curid = 0

    for _ in range(1):
        curcond = []
        curth = []

        #select a new condition for the patient
        newcond = choice(condlist)

        for _ in range(randint(1, maxcond)):

            cond = choice(condlist)
            while cond == newcond: #not allow previously diagnosed cond to be the same as the new cond
                cond = choice(condlist)

            if cond not in curcond:
                
                diagnosed = random_date()
                cond["diagnosed"] = diagnosed #python: if dict key doesn't exist, a new one will be created

                trials = generate_trials(cond, diagnosed, len(curth))
                cond["cured"] = has_been_cured(trials)

                curcond.append(cond)
                for el in trials: #because trials is a list
                    curth.append(el)

        # patients.append( {"id": "p"+str(curid), "name": choice(n) +" "+ choice(ln), "conditions": curcond, "trials": curth} )
        patients.append( {"id": str(curid), "name": choice(n) +" "+ choice(ln), "conditions": curcond, "trials": curth} )
        curid = curid + 1
    
    with open("data/newpatient.json", "w+") as fp:
        fp.write(json.dumps( { "Conditions": newcond, "Patients": patients}, indent=4 ))

if __name__=="__main__":
    main()