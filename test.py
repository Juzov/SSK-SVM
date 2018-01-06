from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup

filename = "reut2-000.sgm"
f = open(filename, "w")


def readData():
    y = []
    with open('reut2-000.sgm', 'r',encoding='ascii') as datafile:
        reader = csv.reader(datafile)
        for i,row in enumerate(reader):
            y[i] = row[0]
    datafile.close()
    return y

page_soup = soup(page_html, "html.parser")

for strong_tag in soup.find_all('strong'):
    print strong_tag.text, strong_tag.snext_sibling

readData()
