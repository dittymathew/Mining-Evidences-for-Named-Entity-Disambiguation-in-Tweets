#!/usr/bin/env python
# -*- coding: utf-8 -*-

from optparse import OptionParser
import sys, re, numpy, copy

def load_corpus(filename):
    corpus = []
    labels = []
    labelmap = dict()
    f = open(filename, 'r')
    for line in f:
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
    f.close()
    return labelmap.keys(), corpus, labels

class LLDA:
    def __init__(self, alpha, alpha_others, beta, beta_bg, beta_others, gamma1, gamma2):
        self.alpha = alpha
        self.alpha_others = alpha_others
        self.beta = beta
        self.beta_bg = beta_bg
        self.beta_others = beta_others
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        
    def term_to_id(self, term):
        if term not in self.vocas_id:
            voca_id = len(self.vocas)
            self.vocas_id[term] = voca_id
            self.id_vocas[voca_id] = term
            self.vocas.append(term)
        else:
            voca_id = self.vocas_id[term]
        return voca_id

    def complement_label(self, label):
        if not label: return numpy.ones(len(self.labelmap))
        vec = numpy.zeros(len(self.labelmap))
        for x in label: vec[self.labelmap[x]] = 1.0
        return vec

    def set_corpus(self, labelset, corpus, labels):
        labelset.insert(0, "background")
        labelset.insert(1, "default")
        self.labelmap = dict(zip(labelset, range(len(labelset))))
        self.K = len(self.labelmap)

        self.vocas = []
        self.vocas_id = dict()
        self.id_vocas = dict()
        self.labels = [self.complement_label(label) for label in labels]
        self.docs = [[self.term_to_id(term) for term in doc] for doc in corpus]

        self.M = len(corpus)
        self.V = len(self.vocas)

        self.z_m_n = []
        self.z_m = numpy.zeros(self.M, dtype=int)
        self.n_m_z = numpy.zeros((self.M, self.K), dtype=int)
        self.n_z_t = numpy.zeros((self.K, self.V), dtype=int)
        self.n_z = numpy.zeros(self.K, dtype=int)
        self.n_m = numpy.zeros(self.M, dtype=int)
        self.m_z = numpy.zeros(self.K, dtype=int)

        for m, doc, label in zip(range(self.M), self.docs, self.labels):
            N_m = len(doc)
            self.n_m[m] += N_m
            l_m = 0
            if label[0] == 0.0:
                for l in range(len(label)): 
                    if label[l] == 1.0:
                        l_m = l
            else:
                label[0] = 0.0
                l_m = numpy.random.multinomial(1, label / label.sum()).argmax()
                label[0] = 1.0
                
            self.z_m[m] = l_m
            self.m_z[l_m] += 1

            z_n = numpy.zeros(N_m)
            for x in range(N_m):
                t = numpy.random.multinomial(1, [1/2.]*2).argmax() 
                if t == 1:
                    z_n[x] = 0
                else:
                    z_n[x] = l_m

            self.z_m_n.append(z_n)
            for t, z in zip(doc, z_n):
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1

    def inference(self, round, totalround, labelset):
        V = len(self.vocas)
        for m, doc, label in zip(range(len(self.docs)), self.docs, self.labels):
            if label[0] == 0.0:
                new_z = self.z_m[m]
                for n in range(len(doc)):
                    t = doc[n]
                    z_w = self.z_m_n[m][n]
                    self.n_z[z_w] -= 1
                    self.n_z_t[z_w, t] -= 1
                    self.n_m_z[m, z_w] -= 1
                p_m_z = numpy.zeros(self.K, dtype=float)
                p_m_z[new_z] = 1
                
            else:
                z = self.z_m[m]
                self.m_z[z] -= 1
                            
                for n in range(len(doc)):
                    t = doc[n]
                    z_w = self.z_m_n[m][n]
                    self.n_z[z_w] -= 1
                    self.n_z_t[z_w, t] -= 1
                    self.n_m_z[m, z_w] -= 1
                            
                p_m_z = numpy.zeros(self.K, dtype=float)
                p_m_z[0] = 0
                
                #Shared Computation
                temp1 = (self.gamma1+self.n_z[0]) / (self.gamma1+self.gamma2+self.n_m.sum()-len(doc))
                temp3 = (self.gamma2+self.n_z.sum()-self.n_z[0]) / (self.gamma1+self.gamma2+self.n_m.sum()-len(doc))
                partialtemp2 = (self.V*self.beta_bg+self.n_z[0])
                fixedval = 1e+4 / ((self.K-1)*self.alpha+self.alpha_others+self.M-1)
                
                #Others Label
                p_m_z[1] = fixedval * (self.alpha_others+self.m_z[1]) 
                for n in range(len(doc)):
                    t = doc[n]
                    z_w = self.z_m_n[m][n]                    
                    temp2 = (self.beta_bg+self.n_z_t[0, t]) / partialtemp2                    
                    temp4 = (self.beta_others+self.n_z_t[1, t]) / (self.V*self.beta_others+self.n_z[1])
                    p_m_z[1] = 1e+4 * p_m_z[1]*(temp1*temp2+temp3*temp4)
                
                
                #Regular Labels
                for k in range(2, self.K):
                    p_m_z[k] = fixedval * (self.alpha+self.m_z[k]) 
                    for n in range(len(doc)):
                        t = doc[n]
                        z_w = self.z_m_n[m][n]
                        temp2 = (self.beta_bg+self.n_z_t[0, t]) / partialtemp2
                        temp4 = (self.beta+self.n_z_t[k, t]) / (self.V*self.beta+self.n_z[k])
                        p_m_z[k] = 1e+4 * p_m_z[k]*(temp1*temp2+temp3*temp4)
                                    
                new_z = numpy.random.multinomial(1, p_m_z / p_m_z.sum()).argmax()
                self.z_m[m] = new_z
                self.m_z[new_z] += 1
                
##            if round+1 == totalround and label[0] != 0.0:
                #print "Doc Label"
##                for ik in numpy.argsort(-p_m_z):
##                    print "%s,%f,%f\t" % (labelset[ik],p_m_z[ik],p_m_z[ik]/p_m_z.sum()),
##                print ""
                
            if new_z == 1:
                betaVal = self.beta_others
            else:
                betaVal = self.beta
                
            #Shared Computation    
            temp1 = (self.gamma1+self.n_z[0]) / (self.gamma2+self.n_z.sum()-self.n_z[0])
            temp3 = (self.V*betaVal+self.n_z[new_z]) / (self.V*self.beta_bg+self.n_z[0])    
            n_z_t_copy = copy.deepcopy(self.n_z_t)
            timesval = temp1*temp3
                
            for n in range(len(doc)):
                t = doc[n]
                temp2 = (self.beta_bg+n_z_t_copy[0, t]) / (betaVal+n_z_t_copy[new_z, t])
                p_bg = timesval*temp2/(1+timesval*temp2)
                bg = numpy.random.multinomial(1, [p_bg, 1-p_bg]).argmax()
                if bg == 0:
                    new_z_w = 0
                else:
                    new_z_w = new_z
            
                self.z_m_n[m][n] = new_z_w
                self.n_z[new_z_w] += 1
                self.n_z_t[new_z_w, t] += 1
                self.n_m_z[m, new_z_w] += 1
                

    #def phi(self):
    #    V = len(self.vocas)
    #    return (self.n_z_t + self.beta) / (self.n_z[:, numpy.newaxis] + V * self.beta)
    
    def phi(self):
        phival = numpy.zeros((self.K, self.V), dtype=float)
        for v in range(0, self.V):
            phival[0, v] = (self.n_z_t[0, v]+self.beta_bg)/(self.n_z[0]+self.V*self.beta_bg)
        for v in range(0, self.V):
            phival[1, v] = (self.n_z_t[1, v]+self.beta_others)/(self.n_z[1]+self.V*self.beta_others)
        for k in range(2, self.K):
            for v in range(0, self.V):
                phival[k, v] = (self.n_z_t[k, v]+self.beta)/(self.n_z[k]+self.V*self.beta)
        return phival

    def theta(self):
        K = self.K
        return (self.n_m_z + self.alpha) / (self.n_m[:, numpy.newaxis] + K * self.alpha)

    def perplexity(self, doc):
        pass

    def output_word_topic_dist(self, num,labelset):
        phi=self.phi()
##        for k in xrange(self.K):
##            print "%s\t"%(labelset[k]),
##            for w in numpy.argsort(-phi[k])[:num]:
##                print "%s, " % (self.vocas[w]),
##            print ""

    def output_topic_doc_dist(self,labelset):
        theta=self.theta()
##        for d in xrange(len(self.docs)):
            #doc = self.docs[d]
            #for w in range(len(doc)):
            #    print "%s\t"%(self.id_vocas[doc[w]]),
##            for z in numpy.argsort(-theta[d]):
##                print " %s,%f " % (labelset[z],theta[d][z]),
##            print ""
            
    def output_doc_label(self,labelset):
        labels =[]
        for d in xrange(len(self.docs)):
            labels.append(labelset[self.z_m[d]])
 ##           print " %s " % (labelset[self.z_m[d]]),
##            print ""
        return labels
    
class LLDA_Main:
  def __init__():
    self.a=''

  def main():
    parser = OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
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
    if not options.filename: parser.error("need corpus filename(-f)")

    labelset, corpus, labels = load_corpus(options.filename)
#    print labelset,corpus,labels
    llda = LLDA(options.alpha, options.alpha_others, options.beta, options.beta_bg, options.beta_others, options.gamma1, options.gamma2)
    llda.set_corpus(labelset, corpus, labels)
    for i in range(options.iteration):
        #sys.stderr.write("-- %d " % (i + 1))
        llda.inference(i, options.iteration, labelset)
##    llda.output_word_topic_dist(options.num,labelset)
##    llda.output_topic_doc_dist(labelset)
    return llda.output_doc_label(labelset)

if __name__ == "__main__":
    main()
