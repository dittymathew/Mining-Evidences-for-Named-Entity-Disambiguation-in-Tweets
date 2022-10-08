from bs4 import BeautifulSoup
import urllib2,re

def getWikipediaContent(title,entity):
  qry=wiki_url+title
  uri = re.sub(' ','_',qry)
  try: wikidata = urllib2.urlopen(uri).read()
  except  urllib2.URLError as e:
    wikidata =''
    print e.reason
  soup = BeautifulSoup(wikidata)
  paras = soup.find_all('p')
  con_str=''
  for par in paras:
#    print par.text
#    print '------------------------------------------------'
    content =par.text.encode('latin-1', 'ignore')
    if entity in content:
      con_str += '['+title+']'+content+'\n'
  return con_str

def getAllReferentCandidates(title,referent_entity):
  print title
  uri = re.sub(' ','_',wiki_url+title+'_(disambiguation)')
  try: wikidata = urllib2.urlopen(uri).read()
  except  urllib2.URLError as e:
    wikidata =''
    print e.reason
  soup = BeautifulSoup(wikidata)
  links =soup.find_all('a')
  docs_str=''
  f =0
  for link in links:
    linktext = link.text.encode('latin-1', 'ignore')
    if linktext == referent_entity:
      f=1
    if title in linktext and 'disambiguation' not in linktext:
      docs_str += getWikipediaContent(linktext,entity)
  if  f==0:
    docs_str += getWikipediaContent(referent_entity,entity)

  return docs_str

wiki_url = "http://en.wikipedia.org/wiki/"

data = open('data/selectedTweets.xml').read()
soup = BeautifulSoup(data, 'xml')
tweets =soup.findAll('Tweet')
labelset =[]
docs='data/ReferDocs/'
for tweet in tweets:
  tweetid =int(tweet.find('Id').text.encode('latin-1', 'ignore'))
  text =tweet.find('Text').text.encode('latin-1', 'ignore')
  entity =tweet.find('Entity').text.encode('latin-1', 'ignore')
  referent_entity = re.sub('_',' ',tweet.find('ReferentEntity').text.encode('latin-1', 'ignore'))
  if referent_entity not in labelset and tweetid >=406:
    labelset.append(referent_entity)
    doc_content =getAllReferentCandidates(entity,referent_entity)    
    wp =open(docs+entity+'.txt','w')
    wp.write(text+'\n')
    wp.write(doc_content)
    wp.close()


