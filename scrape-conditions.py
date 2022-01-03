import requests
from bs4 import BeautifulSoup
import json

conditions_file = "dataset/conditions.json"

url = "https://www.nhsinform.scot/illnesses-and-conditions/a-to-z"
page = requests.get(url)

soup = BeautifulSoup(page.content, 'html.parser')

# print(soup)

#html = soup.find() #soup.find() retrieves the first tag in the tree
# h2 = soup.find_all('h2', {"class": "module__title"}) #in the doc conditions are h2 and class module__title
conds = soup.find_all('a', {"class": "nhs-uk__az-link"})
# print(conds)

condlist = []
curid = 0
for i in conds:

    # print(i.get_text().strip())
    disease = i.get_text().strip()
    # print(disease)
    # condlist.append( {"id": "c"+str(curid), "name": disease, "type": disease} )
    condlist.append( {"id": str(curid), "name": disease, "type": disease} )

    curid = curid + 1

#conditions = json.dumps({ "conditions" : condlist}, indent=4)
conditions = json.dumps(condlist, indent=4)

with open(conditions_file, "w+") as file:
    file.write(conditions)