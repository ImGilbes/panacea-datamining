import requests
from bs4 import BeautifulSoup
import json

therapies_file = "dataset/therapies.json"

url = "https://en.wikipedia.org/wiki/List_of_therapies"
page = requests.get(url)

soup = BeautifulSoup(page.content, 'html.parser')

#print(soup.prettify())

ths = soup.find_all('a', attrs={'title':lambda x: "therapy" in str(x)})


thlist = []
thid = 0
for i in ths:
    curtherapy = i.get_text().strip()
    # thlist.append( {"id": "t"+str(thid), "name": curtherapy, "type": curtherapy} )
    thlist.append( {"id": str(thid), "name": curtherapy, "type": curtherapy} )
    thid = thid + 1

with open(therapies_file, "w+") as fp:
    #fp.write(json.dumps( {"therapies": thlist}, indent=4))
    fp.write(json.dumps(thlist, indent=4))