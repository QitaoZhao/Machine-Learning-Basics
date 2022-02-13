import urllib.request
import re
from bs4 import BeautifulSoup

def getHtml(url):
    page = urllib.request.urlopen(url)
    html = page.read().decode('utf-8')
    return html

def getWord(html):
    bs = BeautifulSoup(html, "html.parser") # 实例化对象
    namelist = bs.findAll("p")
    return namelist

def download(url, filename):
	html = getHtml(url)
	namelist = getWord(html)

	File = open(filename, 'w')
	for name in namelist:
	    File.write(name.get_text())

	File.close()

hamilton = [1, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
			30, 31, 32, 33, 34, 35, 36, 59, 60, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 
			74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85]

madison = [10, 14, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 58]

unknown = [49, 50, 51, 52, 53, 54, 55, 56, 57, 62, 63]

for index in hamilton:
# for index in madison:
# for index in unknown:
	print(index)
	if index < 10:
		download("https://avalon.law.yale.edu/18th_century/fed0%d.asp" % index, '%d.txt' % index)
	else:
		download("https://avalon.law.yale.edu/18th_century/fed%d.asp" % index, '%d.txt' % index)		

