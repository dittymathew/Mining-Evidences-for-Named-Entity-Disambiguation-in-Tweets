from eff_geed_full_V1 import LLDA
from mechanize import Browser
from nltk.tokenize import RegexpTokenizer
from bs4 import BeautifulSoup
from optparse import OptionParser
import re,os,nltk,urllib2

remove_list =['youtube','wikipedia','linkedin','facebook','twitter','blog','quora','ppt','pdf','doc','maps','spread.html','^\?']

def extractDefnUrl(url):
  br = Browser()
  br.set_proxies({'http':'http://cs13d018:5qgo~skY@hproxy.iitm.ac.in:3128',
                                           'https':'https://cs13d018:5qgo~skY@hproxy.iitm.ac.in:3128'})
  br.set_handle_robots(False)
  res=br.open(url)
  # response = urllib.request.urlopen(url)
  html = res.read()
  return html


def getSearchResultDDG(query):
  q =t= re.sub(' ', '_', query)
  query = urllib2.quote(query)
#  uri = 'https://api.duckduckgo.com/?q=%s&format=json&no_html=1&no_redirect=1&kp=-1' % query
  uri ='http://duckduckgo.com/html/?q='+query
  proxy = urllib2.ProxyHandler({'http': 'http://cs13d018:5qgo~skY@hproxy.iitm.ac.in:3128'})
  opener = urllib2.build_opener(proxy)
  urllib2.install_opener(opener)
  try: data  = urllib2.urlopen (uri).read()
  except urllib2.URLError as e:
    print e.reason
    return []
#  data = urllib2.urlopen (uri).read()
 # data = extractDefnUrl(uri)
  parsed = BeautifulSoup(data)
  topics = parsed.findAll('a')
  results =[link['href'] for link in topics if link.has_key('href')]
  if results != []:
    results.pop(0)
  no_results= len(results)-10
  results =[r for r in set(results[0:no_results])]
  prune_results =[]
  for res in results:
    f=0
    for key in remove_list:
      if re.findall(key,res):
        f=1
        break
    if f==0:
      try: con_type =urllib2.urlopen (res).info()['Content-Type']
      except urllib2.URLError as e:
        print e.reason
        con_type=''
      if re.findall('pdf$',con_type) :
        f=1
    if f==0:
      prune_results.append(res)
  return prune_results

def jaccard(A,B):
    sim= float(len(set(A) & set(B)))/float(len(set(A) | set(B)))
 #   print sim
    return sim

def getExternalCorpus(query):
  search_results =getSearchResultDDG(query)
#  print 'Search Results ',len(search_results)
  if len(search_results)>20:
    search_results =search_results[0:20]
  C =[]
  error=0
  for uri in search_results:
    try: html  = urllib2.urlopen (uri).read()
    except urllib2.URLError as e:
      print e.reason
      error =1
    if error ==0:
      C_raw = re.sub('[\n\t]','',nltk.clean_html(html))
      C.append(C_raw)
  return C


def findOverlapC(C,D):
  tokenizer = RegexpTokenizer(r'\w+')
  overlapContent=[]
  unoverlapped =[]
  for dc in C:
    tokenized_Cdata = tokenizer.tokenize(dc.lower())
    f =0

    for d in D:
      tokenized_labeldata =tokenizer.tokenize(d.lower())
      if jaccard(tokenized_Cdata,tokenized_labeldata) >=0.02:
        f=1
        break
    if f==1:
      overlapContent.append(dc)
    else:
      unoverlapped.append(dc)
  return (overlapContent,unoverlapped)


def load_corpus(docContent):
    corpus = []
    labels = []
    labelmap = dict()
#    f = open(filename, 'r')
    for line in docContent:
        mt = re.match(r'\[(.+?)\](.+)', line)
        if mt:
            label = mt.group(1).split(',')
            for x in label: labelmap[x] = 1
            line = mt.group(2)
        else:
            label = None
        doc = re.findall(r'\S+',line.lower())
        if len(doc)>0:
            corpus.append(doc)
            labels.append(label)
#    f.close()
    return labelmap.keys(), corpus, labels

def llda_main(doccon):
    parser = OptionParser()
#    parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.001)
    parser.add_option("--alpha_others", dest="alpha_others", type="float", help="parameter alpha_others", default=0.1)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.001)
    parser.add_option("--beta_others", dest="beta_others", type="float", help="parameter beta_others", default=0.1)
    parser.add_option("--beta_bg", dest="beta_bg", type="float", help="parameter beta_bg", default=0.1)
    parser.add_option("--gamma1", dest="gamma1", type="float", help="parameter gamma1", default=0.0003)
    parser.add_option("--gamma2", dest="gamma2", type="float", help="parameter gamma2", default=0.001)    
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=200)
    parser.add_option("--df", dest="threshold", type="int", help="threshold of document freaquency to cut words", default=0)
    parser.add_option("--n", dest="num", type="int", help="number of top words", default=20)

    (options, args) = parser.parse_args()
#    if not options.filename: parser.error("need corpus filename(-f)")

    labelset, corpus, labels = load_corpus(doccon)
#    print labelset,corpus,labels
    llda = LLDA(options.alpha, options.alpha_others, options.beta, options.beta_bg, options.beta_others, options.gamma1, options.gamma2)
    llda.set_corpus(labelset, corpus, labels)
    for i in range(options.iteration):
        #sys.stderr.write("-- %d " % (i + 1))
        llda.inference(i, options.iteration, labelset)
    llda.output_word_topic_dist(options.num,labelset)
    llda.output_topic_doc_dist(labelset)
    return llda.output_doc_label(labelset)


if __name__ == "__main__":
  data = open('data/selectedTweets.xml').read()
  soup = BeautifulSoup(data, 'xml')
  tweets =soup.findAll('Tweet')
  docs='data/ReferDocs/4/'
  noOfDocProcessed=0
  truePredict={}
  for i in range(0,6):
    truePredict[i]=0
  fto = open('tweetids_output', 'w')
  for tweet in tweets :
    tweetid =int(tweet.find('Id').text.encode('latin-1', 'ignore'))
    twittertext =tweet.find('Text').text.encode('latin-1', 'ignore')
    entity =tweet.find('Entity').text.encode('latin-1', 'ignore')
    referent_entity = re.sub('_',' ',tweet.find('ReferentEntity').text.encode('latin-1', 'ignore'))
    
    print 'Twitterid:'+'\n'
    print tweetid

    
#    docs ='data/ReferDocs/1'
#    docCopy ='data/ReferDocCopy/'
    fls =os.listdir(docs)
    if entity+'.txt' in fls and tweetid > 1586:
      noOfDocProcessed +=1
      label_doc=open(docs+entity+'.txt').read().strip().split('\n')
      label_doc.pop(0)
      sure_label_doc =label_doc
#      open(docCopy+entity+'.txt','w') .write(''.join(line+'\n') for line in label_doc)
      label_doc.append(twittertext)
      doc_labels =llda_main(label_doc)
      n_docs =len(doc_labels)

      predict_label= doc_labels[n_docs-1]   # Zeroth iteration
#      print doc_labels
      print  'Iteration : 0 Predicted : ', predict_label.lower(),' True : ',referent_entity.lower()
      if predict_label.lower()==referent_entity.lower():
        truePredict[0] +=1
      
      else:
        C = getExternalCorpus(entity)
#        print 'External corpus Docs ',len(C)
#        print 'Labeled docs ',len(label_doc)
## for iterations



        for iter in range(1,6) :
          if C ==[]:
            break
          (Dadd,C) = findOverlapC(C,label_doc)  # C= C-Dadd
#          print len(Dadd)
#        print Dadd
#        C =set(C) -set(Dadd)
#          print len(C)
          label_doc= sure_label_doc  # Set of currenly labeled documents
          for d in Dadd:
            label_doc.append(d) #Add overlaped documents
          label_doc.append(twittertext) ## Added tweet at the end
          doc_labels =llda_main(label_doc)
          n_docs =len(doc_labels)
          predict_label= doc_labels[n_docs-1]   # Zeroth iteration


          if predict_label.lower()==referent_entity.lower():
            truePredict[iter] +=1
          else:
            for i in range(0,n_docs):
              if doc_labels[i] !='default' and label_doc[i] not in sure_label_doc:  # Newly labeled documents
                sure_label_doc.append('['+doc_labels[i]+']'+label_doc[i])
          print  'Iteration : ',iter,' Predicted : ', predict_label.lower(),' True : ',referent_entity.lower()


      print 'Total No of Docs :',noOfDocProcessed
      for i in range(0,6):
        print 'Correct Predictions iter ',i,' ',truePredict[i]


        
