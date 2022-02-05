# panacea-datamining, developed by Gianluca Sassetti

Run the final version with:
$python sgd.py <dataset-name> <patientID> <conditionID>

Be sure to have pytorch installed before running. In my case i used Anaconda to manage the installation of Pytorch and the creation of a virtual
environment that uses python3.7. The developed code can be usd with older versions of python as well.

Running sgd.py will output, besides the 5 best recommended therapies, also metrics stating the efficacy of the procedure (matrices after LSH, reconstruction mse), that have been left there on purpose.

The source code of the final version is in src/sgd.py
Other files contain previous versions, developed only for the local dataset (there are differences with the indexes).
The old, testing-only local daset can still be found in the folder data, or can be recreted with the dedicated scripts in the src folder.