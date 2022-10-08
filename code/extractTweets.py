from bs4 import BeautifulSoup
import re

wp =open('data/selectedTweets.xml','w')
wp.write('<Tweets>\n')
data = open('data/TwitterNEED-MasterBranch/Microposts2014_Collection_train.xml').read()
soup = BeautifulSoup(data, 'xml')
tweets =soup.findAll('Tweet')
id=1
for tweet in tweets:
  entity =tweet.find('Text')
  referent_entity = tweet.find('Entity')
  
  if referent_entity !=None  and entity !=None:
    print entity,referent_entity
    wp.write('<Tweet>\n')
    wp.write('<Id>'+str(id)+'</Id>')
    wp.write('<Text>'+tweet.find('TweetText').text.encode('latin-1', 'ignore')+'</Text>\n')
    wp.write('<Entity>'+entity.text.encode('latin-1', 'ignore')+'</Entity>\n')
    wp.write('<ReferentEntity>'+referent_entity.text.encode('latin-1', 'ignore')+'</ReferentEntity>\n')
    wp.write('</Tweet>\n')
    id +=1

wp.write('</Tweets>\n')


