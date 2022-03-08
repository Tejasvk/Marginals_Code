
'''
This file includes all the locally differentially private mechanisms we designed for the SIGMOD work.
I am aware that this code can be cleaned a bit and there is a redundancy. But this helps keeping the code plug-n-play.
I can simply copy a class and use it in a different context.

http://dimacs.rutgers.edu/~graham/pubs/papers/sigmod18.pdf
'''

import numpy as np
import itertools
from scipy.linalg import hadamard
import pandas as pd
import xxhash
import sys
import random
#np.seterr(all='raise')

BIG_PRIME = 9223372036854775783
def rr2 (bit,bern):
    if bern:
        return bit
    return -bit


def pop_probmat(prob,sz):
    probmat =np.zeros((sz,sz))
    d = np.log2(sz)
    for i in range(0,sz):
        for j in range(0,sz):
            perturbed = count_1(np.bitwise_xor(i,j))
            #print i,bin(i),j,bin(j) ,bin(np.bitwise_xor(i,j)),perturbed
            probmat[i][j] = np.power(1.0-prob,perturbed) *  np.power(prob,d-perturbed)
    return probmat

def mps (num,bern,rnum):
    if bern:
        return num
    return rnum

def L1(a,b):
    a = np.abs(a)
    b= np.abs(b)
    return round(np.abs(a-b).sum(),4)


def count_1(num):
    cnt =0
    while num !=0:
        num = np.bitwise_and(num,num-1)
        cnt+=1
    return  cnt



def random_number():
    return random.randrange(1, BIG_PRIME - 1)


def compute_marg(misc_vars
                 ,irr_estimate
                 ,ips_estimate
                 ,iht_pert_ns_estimate
                 ,iolh_estimate
                 ,mps_pert_dict
                 ,mrr_pert_dict
                 ,mht_pert_dict
                 ,icms_estimate
                 ,icmsht_estimate
                 ):

    ### These lists store L1 error for each k way marginal.
    irr_l1_array = []
    iht_l1_array = []
    ips_l1_array =[]

    iolh_l1_array =[]
    icms_l1_array = []
    icmsht_l1_array = []

    mps_l1_array= []
    mrr_l1_array=[]
    mht_l1_array = []


    s = misc_vars.allsubsetsint.shape[0]
    temp_array2= np.zeros(s)
    input_dist_margs = np.zeros(np.power(2,misc_vars.d))


    marg_from_irr    = np.zeros(np.power(2,misc_vars.d))
    marg_from_iht    = np.zeros(s)
    marg_from_ips    = np.zeros(np.power(2,misc_vars.d))
    marg_from_iolh   = np.zeros(np.power(2,misc_vars.d))
    marg_from_icms   = np.zeros(np.power(2,misc_vars.d))
    marg_from_icmsht = np.zeros(np.power(2,misc_vars.d))

    all_cords = np.array(range(0, np.power(2,misc_vars.d)))
    temp_array = np.zeros(np.power(2, misc_vars.d))
    ### We now evaluate each marginal using the method described in Barak et al's paper.
    for beta in misc_vars.allsubsetsint:

        if count_1(beta) != misc_vars.k:
            continue

        alphas=misc_vars.alphas_cache[beta]["alphas"]
        gammas = alphas

        marg_from_irr.fill(0.0)
        marg_from_ips.fill(0.0)
        marg_from_iht.fill(0.0)
        marg_from_iolh.fill(0.0)
        marg_from_icms.fill(0.0)
        marg_from_icmsht.fill(0.0)

        input_dist_margs.fill(0.0)
        real_indices = []
        for alpha in alphas:
            temp_array.fill(0.0)
            temp_array2.fill(0.0)
            try:
                f_alpha = misc_vars.f[alpha]
            except:
                f_alpha = np.zeros(np.power(2,misc_vars.d))
                for i in all_cords:
                    f_alpha[i] = np.power(-1.0, count_1(np.bitwise_and(alpha, i)))
                misc_vars.f[alpha] = f_alpha

            for gamma in gammas:
                temp_array[gamma]+=misc_vars.f[alpha][gamma]
                temp_array2[misc_vars.coef_dict[gamma]] +=np.power(-1.0,count_1(np.bitwise_and(gamma,alpha)))
            try:
                input_dist_margs    += (temp_array * misc_vars.f[alpha].dot(misc_vars.input_dist))
                marg_from_irr       += (temp_array * misc_vars.f[alpha].dot(irr_estimate))
                marg_from_ips       += (temp_array * misc_vars.f[alpha].dot(ips_estimate))
                marg_from_icms      += (temp_array * misc_vars.f[alpha].dot(icms_estimate))
                marg_from_icmsht    += (temp_array * misc_vars.f[alpha].dot(icmsht_estimate))
                marg_from_iolh      += (temp_array * misc_vars.f[alpha].dot(iolh_estimate))

            except:
                print ("Unexpected error:", sys.exc_info())


            marg_from_iht  += (temp_array2 *  iht_pert_ns_estimate[misc_vars.coef_dict[alpha]])

            real_indices.append(misc_vars.coef_dict[alpha])

        ### input######

        m_inp = np.abs(np.take(input_dist_margs,gammas)) ## Extracting counts from marginal indices specified by "gammas".
        m_inp/=m_inp.sum()
        #### INPUT_HT #############
        m_inp_ht = np.abs(np.take(marg_from_iht,real_indices)) ## Extracting counts from marginal indices specified by "gammas".
        m_inp_ht/=m_inp_ht.sum()
        iht_l1_array.append(L1(m_inp_ht,m_inp))

        ######## INPUT_PS ###########
        ips_marg = np.abs(np.take(marg_from_ips,gammas)) ## Extracting counts from marginal indices specified by "gammas".
        ips_marg/=ips_marg.sum()
        ips_l1_array.append(L1(ips_marg,m_inp))

        ######## INPUT_RR ##########
        m_irr = np.abs(np.take(marg_from_irr, gammas)) ## Extracting counts from marginal indices specified by "gammas".
        m_irr /= m_irr.sum()

        irr_l1_array.append(L1(m_irr,m_inp))

        ######### INPUT_OLH ##########

        try:
            m_iolh  = np.abs(np.take(marg_from_iolh,gammas)) ## Extracting counts from marginal indices specified by "gammas".
            m_iolh/=m_iolh.sum()
            iolh_l1_array.append(L1(m_iolh,m_inp))
        except:
            ## incase we drop INPUT_OLH from execution.
            #print ("Unexpected error:", sys.exc_info())
            iolh_l1_array.append(0.0)


        try:
            icms_marg = np.abs(np.take(marg_from_icms,gammas)) ## Extracting counts from marginal indices specified by "gammas".
            icms_marg/=icms_marg.sum()
            icms_l1_array.append(L1(icms_marg,m_inp))
        except:
            # incase we drop INPUT_CMS from execution.

            #print ("Unexpected error:", sys.exc_info())
            icms_l1_array.append(0.0)
        try:
            icmsht_marg = np.abs(np.take(marg_from_icmsht,gammas)) ## Extracting counts from marginal indices specified by "gammas".
            icmsht_marg/=icmsht_marg.sum()
            icmsht_l1_array.append(L1(icmsht_marg,m_inp))
        except:
            # incase we drop INPUT_HTCMS from execution.
            #print (icms_marg)
            #print ("Unexpected error:", sys.exc_info())
            icmsht_l1_array.append(0.0)


        ######### MARG_RR ###############
        mrr_l1_array.append(L1(m_inp,mrr_pert_dict[np.binary_repr(beta,width=misc_vars.d)[::-1]]))
        #print (m_inp)
        ######### MARG_HT #####################
        mht_l1_array.append(L1(mht_pert_dict[np.binary_repr(beta,width=misc_vars.d)[::-1]],m_inp))
        ########## MARG_PS #####################
        mps_l1_array.append(L1(mps_pert_dict[np.binary_repr(beta, width=misc_vars.d)[::-1]], m_inp))


    irr_l1    = np.array(irr_l1_array).mean(axis=0)
    ips_l1    = np.array(ips_l1_array).mean(axis=0)
    iht_l1    = np.array(iht_l1_array).mean(axis=0)
    iolh_l1   = np.array(iolh_l1_array).mean(axis=0)
    icms_l1   = np.array(icms_l1_array).mean(axis=0)
    icmsht_l1 = np.array(icmsht_l1_array).mean(axis=0)

    mrr_l1    = np.array(mrr_l1_array).mean(axis=0)
    mps_l1    = np.array(mps_l1_array).mean(axis=0)
    mht_l1    = np.array(mht_l1_array).mean(axis=0)

    #print (irr_l1_array,mrr_l1,iht_l1_array,mht_l1,ips_l1,mps_l1,iolh_l1_array,icms_l1_array,icmsht_l1_array)

    return (irr_l1,mrr_l1,iht_l1, mht_l1, ips_l1, mps_l1, iolh_l1, icms_l1, icmsht_l1)

class INPUT_RR(object):
    def perturb2(self):
        return
    def perturb(self,index_of_1,p):
        i = 0
        while i < self.sz:
            item = 0.0
            if i == index_of_1:
                item = 1.0
            if self.bern_irr[p][i]:
                self.irr[i] += item
            else:
                self.irr[i] += (1.0 - item)
            i += 1

    ## It is possible to simulate InputRR using Binomial distributions. We
    ## use this simulation for rapid completion.
    def correction2(self,miscvar):
        i=0
        irr2 = np.zeros(self.sz)
        while i < self.sz:
            irr2[i] = np.random.binomial(miscvar.input_dist[i],0.5,size=1)[0] +\
                    np.random.binomial(self.population- miscvar.input_dist[i],1.0-self.prob,size=1)[0]
            irr2[i]/=self.population
            irr2[i] = (self.irr[i] + self.prob - 1.0) / (2.0 * self.prob - 1.0)
            i+=1
        np.copyto(self.irr,irr2)
        #print (irr2)

    ## just repeat reconstruction of each index to reduce variance.
    def correction3(self,miscvar):
        i=0
        while i <self.sz:
            j=0
            while j<5:
                self.irr[i] += (np.random.binomial(miscvar.input_dist[i],0.5,size=1)[0] +\
                              np.random.binomial(self.population- miscvar.input_dist[i],self.prob,size=1)[0])
                j+=1

            self.irr[i]/=(5.0*self.population)

            self.irr[i] = (self.irr[i]-self.prob) / (0.5 -self.prob);
            #self.irr[i] = (self.irr[i] + self.prob - 1.0) / (2.0 * self.prob - 1.0)
            i+=1

        #print (self.irr)

    def correction(self):

        self.irr/=self.population
        #print (self.irr)
        for i in range(0,self.sz):
            self.irr[i] = (self.irr[i]+self.prob-1.0)/(2.0*self.prob-1.0)
        #self.irr/=self.irr.sum()
        #print (self.irr.round(4))

    def __init__(self,e_eps,d,population):
        self.population=population
        self.d = d
        self.sz = np.power(2, self.d)
        self.eps = np.log(e_eps)
        self.e_eps = np.power(np.e,(self.eps/2.0))
        self.prob = self.e_eps/(1.0+self.e_eps)
        #print (self.prob,"input-RR")

        self.problist = [self.prob,1.0-self.prob]
        #self.bern_irr = np.random.choice([True,False], size=self.sz * self.population, p=self.problist).reshape(self.population, self.sz)
        #self.sample_index = np.random.choice(range(0, self.sz), size=self.population)
        self.irr = np.zeros(np.power(2,self.d))

class MARG_RR(object):
    def perturb(self,index_of_1,p,rand_quests):
        i = 0
        if not rand_quests in self.marg_dict:
            self.marg_dict[rand_quests] = np.zeros(self.sz)
            self.marg_freq[rand_quests] = 0.0

        self.marg_freq[rand_quests] += 1.0
        while i < self.sz:
            item = 0.0
            if i == index_of_1:
                item = 1.0
            if self.bern[p][i]:
                self.marg_dict[rand_quests][i] += item
            else:
                self.marg_dict[rand_quests][i] += (1.0 - item)
            i += 1

    def perturb2(self,index_of_1,p,rand_quests):
        i = 0
        if not rand_quests in self.marg_dict:
            self.marg_dict[rand_quests] = np.zeros(self.sz)
            self.marg_freq[rand_quests] = 0.0

        self.marg_freq[rand_quests] += 1.0
        while i < self.sz:
            item = 0.0
            b = self.bern_q
            if i == index_of_1:
                item = 1.0
                b = self.bern_p
            if b[p][i]:
                self.marg_dict[rand_quests][i] += item
            else:
                self.marg_dict[rand_quests][i] += (1.0 - item)
            i += 1

    def perturb3(self,index_of_1,p,rand_quests):
        try:
            self.marg_freq[rand_quests] += 1.0
            self.true_marg[rand_quests][index_of_1]+= 1.0
        except:
            self.marg_dict[rand_quests] = np.zeros(self.sz)
            self.marg_freq[rand_quests] = 0.0
            self.true_marg[rand_quests] = np.zeros(self.sz)

            self.marg_freq[rand_quests] += 1.0
            self.true_marg[rand_quests][index_of_1]+= 1.0



    def correction(self):
        #print ("--------------------------------")
        for marg in self.marg_dict:
            self.marg_dict[marg] /= self.marg_freq[marg]
            for i in range(0,self.sz):
                self.marg_dict[marg][i] = (self.marg_dict[marg][i]+self.prob-1.0)/(2.0*self.prob-1.0)

            self.marg_dict[marg]/=self.marg_dict[marg].sum()

    def correction2(self):
        for marg in self.marg_dict:
            #print ("--------------------------------")
            self.marg_dict[marg] /= self.marg_freq[marg]
            for i in range(0,self.sz):
                #self.marg_dict[marg][i] = (self.marg_dict[marg][i]+self.prob-1.0)/(2.0*self.prob-1.0)
                self.marg_dict[marg][i] = (self.marg_dict[marg][i]-(self.prob)) / (0.5 -(self.prob))

            self.marg_dict[marg]/=self.marg_dict[marg].sum()
    def correction3(self):
        for marg in self.marg_dict:
            #self.marg_dict[marg] /= self.marg_freq[marg]
            i=0
            #print (self.marg_dict[marg])
            total = self.marg_freq[marg]
            while i <self.sz:
                j=0
                while j <5:
                    self.marg_dict[marg][i] += (np.random.binomial(self.true_marg[marg][i],0.5,size=1)[0] +\
                                  np.random.binomial(self.marg_freq[marg]- self.true_marg[marg][i],self.prob,size=1)[0])
                    j+=1

                self.marg_dict[marg][i] /= (5.0*total)
                #self.marg_dict[marg][i] = (self.marg_dict[marg][i]+self.prob-1.0)/(2.0*self.prob-1.0)
                self.marg_dict[marg][i] = (self.marg_dict[marg][i]-(self.prob)) / (0.5 -(self.prob))
                i+=1
            self.marg_dict[marg]/=self.marg_dict[marg].sum()


    def __init__(self,d,k,e_eps,population,k_way):
        self.d = d
        self.k = k
        self.population= population
        self.k_way = k_way
        self.sz = np.power(2,self.k)
        self.eps = np.log(e_eps)
        self.e_eps = np.power(np.e,self.eps/2.0)
        self.prob = self.e_eps / (1.0+self.e_eps)
        #print (self.prob,"marg-RR")
        self.problist = [self.prob,1.0-self.prob]
        self.k_way_marg_ps = np.random.choice(self.k_way,size=self.population)
        self.bern = np.random.choice([True, False], size=self.sz * self.population, p=self.problist).reshape(self.population, self.sz)
        self.bern_p = np.random.choice([True, False], size=self.sz * self.population).reshape(self.population, self.sz)

        self.bern_q = np.random.choice([True, False], size=self.sz * self.population, p=self.problist[::-1]).reshape(self.population, self.sz)

        self.marg_dict = {}
        self.marg_freq={}
        self.true_marg={}


class MARG_HT(object):
    def perturb(self,index_of_1,p,rand_quests):
        if not rand_quests in self.marg_dict:
            self.marg_dict[rand_quests] = np.zeros(self.sz)
            self.marg_freq[rand_quests] = np.zeros(self.sz)
        cf =self.rand_coef[p]

        self.marg_freq[rand_quests][cf] += 1.0
        htc = self.f[index_of_1][cf]

        if self.bern[p]:
            self.marg_dict[rand_quests][cf] += htc
        else:
            self.marg_dict[rand_quests][cf] += -htc
    def correction(self):
        for rm in self.marg_dict:
            self.marg_freq[rm][self.marg_freq[rm] == 0.0] = 1.0

            self.marg_dict[rm]/=self.marg_freq[rm]
            self.marg_dict[rm]/=(2.0*self.prob-1.0)
            self.marg_dict[rm][0]=1.0
            #print ("-------------------")
            #print (self.marg_dict[rm])
            self.marg_dict[rm]= np.abs(self.marg_dict[rm].dot(self.f))
            self.marg_dict[rm]/=self.marg_dict[rm].sum()
            #print (self.marg_dict[rm].round(4))
    def pop_probmat(self):
        probmat =np.zeros((self.sz,self.sz))
        for i in range(0,self.sz):
            for j in range(0,self.sz):
                if i ==j:
                    probmat[i][j]= self.prob
                else:
                    probmat[i][j]= (1.0-self.prob)/(self.sz-1.0)
        return probmat
    def compute_all_marginals(self):
        for marg_int in self.k_way:
            self.correct_noise_mps(marg_int)


    def __init__(self,d,k,e_eps,population,k_way,cls):
        self.d = d
        self.k = k
        self.population= population
        self.sz = np.power(2,self.k)
        self.e_eps = e_eps
        self.f = hadamard(self.sz).astype("float64")
        self.prob = (self.e_eps/(1.0+self.e_eps))
        self.problist = [self.prob,1.0-self.prob]
        self.coef_dist = np.zeros(cls)
        self.k_way = k_way
        self.k_way_marg_ps = np.random.choice(self.k_way,size=self.population)
        self.rand_coef= np.random.choice(range(0,self.sz),size=population)
        self.bern = np.random.choice([True, False], size= self.population, p=self.problist)#.reshape(self.population, self.sz)
        self.marg_freq = {}
        self.marg_dict = {}
        self.marg_noisy = np.zeros(self.sz)

class MARG_PS(object):
    def perturb(self,index_of_1,p,rand_quests):

        try:
            freq = self.rand_cache[index_of_1]["freq"]
        except:
            i = 0
            while i < self.sz:
                options = list(range(0, self.sz))
                options.remove(i)
                self.rand_cache[i] = {"rnum": np.random.choice(np.array(options), size=10000), "freq": 0}
                i += 1
            freq = self.rand_cache[index_of_1]["freq"]
        if freq > 9990:
            options = list(range(0, self.sz))
            options.remove(index_of_1)
            self.rand_cache[index_of_1]["rnum"] = np.random.choice(np.array(options), size=10000)
            self.rand_cache[index_of_1]["freq"] = 0
        rnum = self.rand_cache[index_of_1]["rnum"][freq]
        try:
            self.marg_ps_pert_aggr[rand_quests].append(mps(index_of_1, self.bern[p], rnum))
        except:
            self.marg_ps_pert_aggr[rand_quests] = [mps(index_of_1, self.bern[p], rnum)]
        self.rand_cache[index_of_1]["freq"] += 1

    def correct_noise_mps(self,marg_int):
        self.marg_int=marg_int
        self.marg_ps_noisy.fill(0.0)
        if type(self.marg_ps_pert_aggr[marg_int]) != "numpy.ndarray":
            for rm in self.marg_ps_pert_aggr:
                self.marg_ps_pert_aggr[rm] = np.array(self.marg_ps_pert_aggr[rm])
        #print (self.marg_ps_pert_aggr.keys())
        for index in self.marg_ps_pert_aggr[marg_int]:
            self.marg_ps_noisy[index]+=1.0
        self.marg_ps_noisy/=self.marg_ps_noisy.sum()
        #marg_ps_recon = np.copy(marg_noisy)
        self.marg_ps_recon = self.mat_inv.dot(self.marg_ps_noisy)
        self.marg_ps_recon/=self.marg_ps_recon.sum()
        #print (self.marg_ps_recon.round(4))
        return self.marg_ps_recon
    def pop_probmat(self):
        probmat =np.zeros((self.sz,self.sz))
        for i in range(0,self.sz):
            for j in range(0,self.sz):
                if i ==j:
                    probmat[i][j]= self.prob
                else:
                    probmat[i][j]= (1.0-self.prob)/(self.sz-1.0)
        return probmat
    def compute_all_marginals(self):
        for marg_int in self.k_way:
            self.marg_dict[marg_int]=self.correct_noise_mps(marg_int)


    def __init__(self,d,k,e_eps,population,k_way):
        self.d = d
        self.k = k
        self.population= population
        self.k_way = k_way
        self.sz = np.power(2,self.k)
        #self.data = data
        self.e_eps = e_eps
        self.prob = (self.e_eps/(self.e_eps+self.sz-1.0))
        #print self.prob,"marg-ps"
        self.probmat = self.pop_probmat()
        self.problist = [self.prob,1.0-self.prob]
        self.mat = self.pop_probmat()
        self.mat_inv = np.linalg.inv(self.mat)
        self.k_way_marg_ps = np.random.choice(self.k_way,size=self.population)
        self.bern = np.random.choice([True, False], p=self.problist, size=self.population)
        self.marg_ps_pert_aggr = {}
        self.rand_cache = {}
        self.marg_int = None
        self.marg_ps_noisy = np.zeros(self.sz)
        self.marg_dict = {}

## From Ninghui Li et al's USENIX paper.
## https://www.usenix.org/system/files/conference/usenixsecurity17/sec17-wang-tianhao.pdf
## This algorithm indeed does well for high order marginals but doesn't outperform INPUT_HT
## for small k's i.e. 2,3, the one's that are the most interesting.
## We trade the gain in accuracy by computational cost. The encoding (or decoding) cost is O(dN).

class INPUT_OLH(object):
    def __init__(self,e_eps, d, population,g=1):
        self.d = d
        self.population= population
        self.sz = int(np.power(2,self.d))
        #self.data = data
        self.e_eps = e_eps
        if g == 1:
            self.g = int(np.ceil(e_eps+1.0))
        else:
            self.g = g
        #print (self.g)

        self.prob = (self.e_eps/(self.e_eps+self.g-1.0))
        self.problist = [self.prob,1.0-self.prob]
        self.bern_ps = np.random.choice([False,True], size=self.population, p=self.problist)
        self.uni_dist = np.random.choice(range(self.g),size=self.population).astype("int32")
        #self.hash_cache = np.array( map(str,range(self.sz)),dtype="str") ## works with Python2
        self.hash_cache = np.array(range(self.sz),dtype="str")
        #self.hashed_pdist = np.zeros(self.population)
        self.estimate = np.zeros(self.sz)


    def perturb(self,x,p):

        if self.bern_ps[p]:
            #x_hash= (xxhash.xxh32(self.hash_cache[x], seed=p).intdigest()) % self.g
            pert_val= (xxhash.xxh32(self.hash_cache[x], seed=p).intdigest()) % self.g
        else:
            pert_val=self.uni_dist[p]
        dom_index = 0
        while dom_index<self.sz:
            if pert_val == (xxhash.xxh32(self.hash_cache[dom_index], seed=p).intdigest() % self.g):
                self.estimate[dom_index]+=1.0
            dom_index+=1

    def correction(self):
        p=0
        while p <self.sz:
            self.estimate[p]=(self.estimate[p] - (self.population/self.g))/(self.prob -(1.0/self.g))
            p+=1

        self.estimate/=self.estimate.sum()
        #print(self.estimate.round(4))

class INPUT_HT(object):
    def perturb(self,index_of_1,p):
        rc = self.rand_coefs[p]
        index = self.misc_vars.coef_dict[rc]
        self.coef_dist[index] += 1.0
        cf = np.power(-1.0, count_1(np.bitwise_and(index_of_1, rc)))
        self.iht_pert_ns_estimate[index] += rr2(cf, self.bern_ht[p])

    def correction(self):
        self.coef_dist[self.coef_dist==0.0]=1.0
        self.iht_pert_ns_estimate/=self.coef_dist
        self.iht_pert_ns_estimate/=(2.0*self.prob-1.0)
        self.iht_pert_ns_estimate[0] = 1.0
        self.coef_dist[self.coef_dist<=0.0]=0.0



    def __init__(self,d,k,e_eps,population,misc_vars):
        self.d = d
        self.k = k
        self.misc_vars = misc_vars
        self.population= population
        self.sz = np.power(2,self.k)
        self.e_eps = e_eps
        self.prob = self.e_eps/(1.0+self.e_eps)
        self.problist = [self.prob,1.0-self.prob]
        self.bern_ht = np.random.choice([True,False],p=self.problist,size=self.population)
        self.rand_coefs = np.random.choice(self.misc_vars.allsubsetsint,size=self.population)
        self.iht_pert_ns_estimate = np.zeros(self.misc_vars.allsubsetsint.shape[0])
        #iht_pert_ns_estimate.fill(0.0)
        self.coef_dist = np.zeros(self.misc_vars.cls)

## From Apple's paper.
## https://machinelearning.apple.com/2017/12/06/learning-with-privacy-at-scale.html
## This algorithm might be a bad performer. But just adding it for a comparison.
class INPUT_CMS:
    def __init__(self, w, d,population,e_eps,domain):
        '''
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be between 0 and 1, exclusive")
        if epsilonh <= 0 or epsilonh >= 1:
            raise ValueError("epsilon must be between 0 and 1, exclusive")

        #self.w = int(np.ceil(np.e / epsilonh))
        #self.d = int(np.ceil(np.log(1 / delta)))
        '''
        self.w=w
        self.d =d
        self.population=population
        self.hash_functions = [self.__generate_hash_function() for i in range(self.d)]
        self.M = np.zeros(shape=(self.d, self.w))
        #print (self.w,self.d,self.w*self.d,self.M.shape)

        self.hash_chooser = np.random.choice(range(self.d),size=self.population)
        self.epsilon = np.log(e_eps)
        self.flip_prob = 1.0/(1.0+np.power(np.e,self.epsilon/2.0))
        problist = [self.flip_prob,1.0-self.flip_prob]
        self.bern = np.random.choice([True,False],p=problist,size=self.population*self.w).reshape(self.population,self.w)
        self.c_eps = (np.power(np.e,self.epsilon/2.0)+1.0)/(np.power(np.e,self.epsilon/2.0)-1.0)
        self.estimate = np.zeros(int(np.power(2,domain)))

    def __generate_hash_function(self):
        a = random_number()
        b= random_number()
        return lambda x: (a * x + b) % BIG_PRIME % self.w

    def perturb(self, key,p):
        hash_choice = self.hash_chooser[p]
        hashed_key = self.hash_functions[hash_choice](abs(hash(str(key))))
        cnt = 0
        while cnt< self.w:
            item = -1.0
            if cnt == hashed_key:
                item = 1.0
            if self.bern[p][cnt]:
                item = -item
            self.M[hash_choice][cnt]+=(self.d * (item*self.c_eps*0.5+0.5))
            cnt+=1
    def query(self,key):
        l =0
        avg=0.0
        hsh_str= abs(hash(str(key)))
        while l < self.d:
            hashed_key = self.hash_functions[l](hsh_str)
            avg+=self.M[l][hashed_key]
            l+=1
        avg/=self.d
        est = ((1.0*self.w)/(self.w-1.0))* (avg- (1.0*self.population)/self.w)
        return est

    def correction(self):
        cnt=0
        while cnt <self.estimate.shape[0]:
            self.estimate[cnt]=self.query(cnt)
            cnt+=1
        self.estimate[self.estimate < 0.0] = 0.0
        self.estimate/=self.estimate.sum()

## From Apple's paper.
## https://machinelearning.apple.com/2017/12/06/learning-with-privacy-at-scale.html
## This algorithm might be a bad performer. But just adding it for a comparison.
class INPUT_HTCMS:
    #def __init__(self, delta, epsilonh,population,e_eps):
    def __init__(self, w, d,population,e_eps,domain):
        self.w=int(w)
        self.d =int(d)
        self.ht = hadamard(self.w, dtype="float32")
        self.population=population
        self.hash_functions = [self.__generate_hash_function() for i in range(self.d)]
        self.M = np.zeros(shape=(self.d, self.w))
        #print (self.w,self.d,self.w*self.d,self.M.shape)
        self.hash_chooser = np.random.choice(range(self.d),size=self.population).astype("int32")
        self.coef_chooser = np.random.choice(range(self.w),size=self.population).astype("int32")
        #self.hash_choice_counter = np.zeros(self.d)
        self.flip_prob = 1.0/(1.0+e_eps)
        problist = [self.flip_prob,1.0-self.flip_prob]
        self.bern = np.random.choice([True,False],p=problist,size=self.population)
        self.c_eps = (e_eps+1.0)/(e_eps-1.0)
        self.estimate = np.zeros(int(np.power(2,domain)))


    def __generate_hash_function(self):
        a = random_number()
        b= random_number()
        return lambda x: (a * x + b) % BIG_PRIME % self.w

    def perturb(self, key,p):
        hash_choice = self.hash_chooser[p]
        #self.hash_choice_counter[hash_choice]+=1.0
        hashed_key = self.hash_functions[hash_choice](abs(hash(str(key))))
        rand_coef = self.coef_chooser[p]
        item = self.ht[rand_coef][hashed_key]
        if self.bern[p]:
            item = -item
        self.M[hash_choice][rand_coef]+=(self.d * item*self.c_eps)

    def correction(self):
        cnt = 0
        while cnt < self.d:
            #print self.M[cnt]
            self.M[cnt] = self.ht.dot(self.M[cnt])
            cnt+=1

        cnt=0
        while cnt <self.estimate.shape[0]:
            self.estimate[cnt]=self.query(cnt)
            cnt+=1
        self.estimate[self.estimate < 0.0] = 0.0
        self.estimate/=self.estimate.sum()


    def query(self,key):
        l =0
        avg=0.0
        hsh_str= abs(hash(str(key)))
        while l < self.d:
            hashed_key = self.hash_functions[l](hsh_str)
            avg+=self.M[l][hashed_key]
            l+=1
        avg/=self.d
        est = ((1.0*self.w)/(self.w-1.0))* (avg- (1.0*self.population)/self.w)
        return est

class INPUT_PS(object):

    def perturb2(self,index_of_1,p):
        if self.bern_ps[p]:
            self.ips_ps_pert_aggr[index_of_1] += 1.0
        else:
            self.ips_ps_pert_aggr[self.rand_coef_ps[p]] += 1.0

    def perturb(self,index_of_1,p):
        try:
            freq = self.rand_cache[index_of_1]["freq"]
        except:
            i = 0
            while i < self.sz:
                options = list(range(0, self.sz))
                options.remove(i)
                self.rand_cache[i] = {"rnum": np.random.choice(np.array(options), size=10000), "freq": 0}
                i += 1
            freq = self.rand_cache[index_of_1]["freq"]
        if freq > 9990:
            options = list(range(0, self.sz))
            options.remove(index_of_1)
            self.rand_cache[index_of_1]["rnum"] = np.random.choice(np.array(options), size=10000)
            self.rand_cache[index_of_1]["freq"] = 0
        rnum = self.rand_cache[index_of_1]["rnum"][freq]
        ips_output = mps(index_of_1, self.bern[p], rnum)
        self.ips_ps_pert_aggr[ips_output] += 1.0
        self.rand_cache[index_of_1]["freq"] += 1

    def correction2(self):
        self.ips_ps_pert_aggr /= self.population
        #print self.ips_ps_pert_aggr, "pert",self.ips_ps_pert_aggr.sum()

        for i in range(0, self.sz):
            self.ips_ps_pert_aggr[i] = (self.ips_ps_pert_aggr[i] * self.sz + self.probps - 1.0) / (self.probps * (self.sz + 1.0) - 1.0)
        #print self.ips_ps_pert_aggr.round(4)
    def correction(self):
        self.ips_ps_pert_aggr /= self.ips_ps_pert_aggr.sum()

        for i in range(0,self.sz):
            self.ips_ps_pert_aggr[i] = (self.ips_ps_pert_aggr[i]*self.sz+self.prob-1.0)/(self.prob*(self.sz+1.0)-1.0)
                #print self.marg_ps_recon.round(4)

        '''
        self.ips_ps_pert_aggr /= self.ips_ps_pert_aggr.sum()
        # marg_ps_recon = np.copy(marg_noisy)
        self.ips_ps_pert_aggr = np.abs(self.mat_inv.dot(self.ips_ps_pert_aggr))
        self.ips_ps_pert_aggr /= self.ips_ps_pert_aggr.sum()
        '''

        #return self.ips_ps_pert_aggr
    def pop_probmat(self):
        probmat =np.zeros((self.sz,self.sz))
        for i in range(0,self.sz):
            for j in range(0,self.sz):
                if i ==j:
                    probmat[i][j]= self.prob
                else:
                    probmat[i][j]= (1.0-self.prob)/(self.sz-1.0)
        return probmat


    def __init__(self,d,k,e_eps,population,misc_vars):
        self.d = d
        self.k = k
        self.population= population
        self.k_way = misc_vars.k_way
        self.sz = np.power(2,self.d)
        self.e_eps = e_eps
        self.prob = (self.e_eps/(self.e_eps+self.sz-1.0))
        #print (self.prob,"input-ps")

        self.problist = [self.prob,1.0-self.prob]
        self.probps = (self.e_eps - 1.0) / (self.e_eps + self.sz - 1.0)
        self.problist2 = [self.probps, 1.0 - self.probps]
        self.rand_coef_ps = np.random.choice(np.array(range(0, self.sz)), size=self.population)
        self.bern_ps = np.random.choice([True, False], size=self.population, p=[self.probps, 1.0 - self.probps])

        #self.mat = self.pop_probmat()
        #self.mat_inv = np.linalg.inv(self.mat)    n = gc.collect()
        self.bern = np.random.choice([True, False], p=self.problist, size=self.population)
        self.ips_ps_pert_aggr = np.zeros(self.sz)
        self.rand_cache = {}
        self.marg_int = None
        self.rand_cache = {}


    #inp_trans_menthods.loc[l]=np.array([population,d,len(iway),input_ht_pert,iht_pert_ns_estimate,had_coefs,input_ps,input_rr],dtype="object")
def change_mapping(d):
    if d:
        return "1"
    return "0"
def get_real_data(population,d):
    data = pd.read_pickle("data/nyc_taxi_bin_sample.pkl").sample(population,replace=True)
    data =  data.as_matrix()
    f = np.vectorize(change_mapping)
    i = data.shape[1]

    remainder = d % i
    ncopies = d/i
    copies = []
    j = 0
    while j < ncopies:
        copies.append(data)
        j+=1
    #print data[:,range(0,remainder)]
    copies.append(data[:,range(0,remainder)])
    #rand_perm = np.random.choice(range(0,d),replace=False,size=d)
    #print rand_perm
    data_high = np.concatenate(tuple(copies),axis=1)#[:,rand_perm]
    #print (data_high.shape)
    #columns= data.columns.tolist()
    #print columns
    #data = f(data_high)
    return f(data_high).astype("str")


class MARGINAL_VARS(object):
    #We cache the set of necessary and sufficient indices to evaluate each <= k way marginal.
    def compute_downward_closure(self):
        all_cords = np.array(range(0, np.power(2, self.d)))
        ## iterate over all possible <=k way marginals.
        for beta in self.allsubsetsint:
            marg_str = bin(beta)[2:]
            marg_str = "0" * (self.d - len(marg_str)) + marg_str
            parity = np.power(2, count_1(beta))
            alphas = np.zeros(parity, dtype="int64")
            cnt = 0
            for alpha in all_cords:
                if np.bitwise_and(alpha, beta) == alpha:
                    alphas[cnt] = alpha
                    cnt += 1
            ### we add marginals in string formats incase needed.
            self.alphas_cache[marg_str] = {"alphas": alphas, "probps": ((self.e_eps - 1.0) / (parity + self.e_eps - 1.0))}
            self.alphas_cache[beta] = {"alphas": alphas, "probps": ((self.e_eps - 1.0) / (parity + self.e_eps - 1.0))}

    ## This method finds the set of <=k way marginal indices i.e. list of all subsets of length <=k from d.
    def get_k_way_marginals(self):
        j = 0
        marginal = np.array(["0"] * self.d)
        while j <= self.k:
            subsets = list(itertools.combinations(range(0, self.d), j))
            subsets = np.array([list(elem) for elem in subsets])
            for s in subsets:
                marginal.fill("0")
                for b in s:
                    marginal[b] = "1"
                self.allsubsetsint.append(int("".join(marginal)[::-1], 2))
                if j == self.k:
                    # k_way.append(int("".join(marginal),2))
                    self.k_way.append("".join(marginal)[::-1])
                    self.k_way_bit_pos.append(s)
                    # print s,marginal,"".join(marginal)
            j += 1
        self.allsubsetsint = np.array(self.allsubsetsint, dtype="int64")
        self.k_way = np.array(self.k_way, dtype="str")

        self.k_way_bit_pos = np.array(self.k_way_bit_pos, dtype="int64")

        self.allsubsetsint.sort()
        #print (self.allsubsetsint)

        ## We tie marginals indices and corresponding bit positions together.
        #print (dict(zip(self.k_way, self.k_way_bit_pos)))
        return dict(zip(self.k_way, self.k_way_bit_pos))

    def __init__(self,d,k,e_eps):
        self.d = d
        self.k = k
        self.input_dist = np.zeros(np.power(2, self.d))
        self.allsubsetsint = []
        self.k_way = []
        self.k_way_bit_pos = []
        self.e_eps = e_eps
        #self.f = hadamard(np.power(2,self.d)).astype("float64")
        self.f = {}
        self.alphas_cache = {}
        self.k_way_bit_pos_dict  =self.get_k_way_marginals()
        self.cls = self.allsubsetsint.shape[0]
        self.coef_dict = dict(zip(self.allsubsetsint, np.array(range(0, self.cls), dtype="int64")))
        self.compute_downward_closure()


'''
Main driver routine that accepts all parameters and
runs perturbation simulation.
'''
def driver(d,k,e_eps,population,misc_vars):
    width = 256
    no_hash = 5
    ###### Use the NYC Taxi data.
    #data = get_real_data(population, d)
    #######  Use synthetic data if you don't have the taxi data. ########
    data = np.random.choice(["1","0"],p=[0.3,0.7],size=d*population).reshape(population,d)

    misc_vars.input_dist.fill(0.0)

    ##### Input Based Algorithms ########
    iht_obj    = INPUT_HT(d, k, e_eps, population, misc_vars)

    ips_obj    = INPUT_PS(d, k, e_eps, population, misc_vars)
    irr_obj    = INPUT_RR(e_eps, d, population)
    iolh_obj   = INPUT_OLH(e_eps, d, population)

    icms_obj   = INPUT_CMS(width, no_hash,population,e_eps,d)
    icmsht_obj = INPUT_HTCMS(width, no_hash,population,e_eps,d)

    ############ Marginal Based Algorithms #########
    mps_obj = MARG_PS(d, k, e_eps, population, misc_vars.k_way)
    mrr_obj = MARG_RR(d, k, e_eps, population, misc_vars.k_way)
    mht_obj = MARG_HT(d, k, e_eps, population, misc_vars.k_way, misc_vars.cls)

    p = 0
    while p < population:
        x = data[p]
        index_of_1 = int("".join(x), 2)
        misc_vars.input_dist[index_of_1] += 1.0
        ############# input_RR###############
        #irr_obj.perturb(index_of_1,p)
        #irr_obj.perturb2()

        #########################input-PS #################################
        ips_obj.perturb2(index_of_1,p)
        ########################################
        iht_obj.perturb(index_of_1, p)
        ##########################INPUT_OLH ###############################
        #INPUT_OLH is a compute intense scheme. Hence we don't run it for larger d's.
        if d < 10:
            iolh_obj.perturb(index_of_1,p)
        ##########################inp_CMS ########################
        icms_obj.perturb(index_of_1,p)
        ##########################inp_HTCMS ########################
        icmsht_obj.perturb(index_of_1,p)
        ########### marg-ps ###########
        rand_questions = mps_obj.k_way_marg_ps[p]
        responses = misc_vars.k_way_bit_pos_dict[rand_questions]
        # print rand_questions,responses
        index_of_1 = int("".join(data[p][responses]), 2)
        mps_obj.perturb(index_of_1, p, rand_questions)

        ######################### marg-ht ############################
        rand_questions = mht_obj.k_way_marg_ps[p]
        responses = misc_vars.k_way_bit_pos_dict[rand_questions]
        # print rand_quests,responses
        index_of_1 = int("".join(data[p][responses]), 2)
        mht_obj.perturb(index_of_1, p, rand_questions)

        ######################### marg-rs #################################
        rand_questions = mrr_obj.k_way_marg_ps[p]
        responses = misc_vars.k_way_bit_pos_dict[rand_questions]
        index_of_1 = int("".join(data[p][responses]), 2)
        mrr_obj.perturb3(index_of_1, p, rand_questions)

        p += 1

    irr_obj.correction3(misc_vars)
    #irr_obj.correction2(misc_vars)

    misc_vars.input_dist /= population
    #irr_obj.correction()
    #print (misc_vars.input_dist.round(4))
    ips_obj.correction()
    iht_obj.correction()
    if d < 10:
        iolh_obj.correction()
    icms_obj.correction()
    icmsht_obj.correction()
    #print(icmsht_obj.estimate)
    mht_obj.correction()
    mrr_obj.correction3()
    mps_obj.compute_all_marginals()
    return compute_marg(misc_vars
                        , irr_obj.irr
                        , ips_obj.ips_ps_pert_aggr
                        , iht_obj.iht_pert_ns_estimate
                        , iolh_obj.estimate
                        , mps_obj.marg_dict
                        , mrr_obj.marg_dict
                        , mht_obj.marg_dict
                        , icms_obj.estimate
                        , icmsht_obj.estimate
                        )


'''
Call this method is used when you want to vary k keeping d, eps fixed.
eps = 1.1
d = 9
'''
def vary_k():

    ## number of repetitions.
    rpt = 5
    e_eps = 3.0

    d = 9
    counter = 0
    ## dfmean and dfstd store the results. We use them in our plotting script.
    l1 = np.zeros((rpt, 9))

    dfmean = pd.DataFrame(columns=["population", "d", "k", "e_eps", "irr_l1", "mrr_l1", "iht_l1", "mht_l1", "ips_l1", "mps_l1","iolh_l1","icms_l1","icmsht_l1"])
    dfstd = pd.DataFrame(columns=["irr_l1_std", "mrr_l1_std", "iht_l1_std", "mht_l1_std", "ips_l1_std", "mps_l1_std","iolh_l1_std","icms_l1_std","icmsht_l1_std"])
    ## parameters of the sketch
    width = 256
    no_hash = 5

    # population variable. We prefer to keep it in the powers of two.
    population = np.power(2, 18)
    for k in reversed(range(1,d)):
        misc_vars = MARGINAL_VARS(d, k, e_eps)
        l1.fill(0.0)
        print ("------------------")
        for itr in (range(rpt)):
            irr_l1, mrr_l1, iht_l1, mht_l1, ips_l1, mps_l1, iolh_l1,icms_l1,icmsht_l1 = driver(d,k,e_eps,population,misc_vars)
            l1[itr] = np.array([irr_l1, mrr_l1, iht_l1, mht_l1, ips_l1, mps_l1,iolh_l1,icms_l1,icmsht_l1])
            print (l1[itr])
        conf = [population, d, k, e_eps]
        conf.extend(l1.mean(axis=0))
        dfmean.loc[counter] = conf
        dfstd.loc[counter] = l1.std(axis=0)
        #print (conf)
        counter += 1

    dfstdcols = list(dfstd.columns.values)
    for c in dfstdcols:
        dfmean[c] = dfstd[c]

    #print (dfmean)
    dfmean.to_pickle("data/all_mechanisms_vary_"+str(d)+".pkl")
    ## (irr_l1,mrr_l1,iht_l1, mht_l1, ips_l1, mps_l1, iolh_l1, icms_l1, icmsht_l1)

    #dfmean.to_pickle("all_mechanisms_vary_k_fo.pkl")

'''
Call this method when you want to vary d holding k, eps, N fixed.
Fixed k, eps values,
k= 3
eps = 1.1
N = 2^18
'''
def vary_d():
    print ("------------------")
    population = int(np.power(2,19))
    e_eps = 3.0
    rpt =4
    l1 = np.zeros((rpt, 9))
    ## Parameters for sketches
    width = 256
    no_hash = 5
    k=3
    dfmean = pd.DataFrame(columns=["population", "d", "k", "e_eps", "irr_l1", "mrr_l1", "iht_l1", "mht_l1", "ips_l1", "mps_l1","iolh_l1","icms_l1","icmsht_l1"])
    dfstd = pd.DataFrame(columns=["irr_l1_std", "mrr_l1_std", "iht_l1_std", "mht_l1_std", "ips_l1_std", "mps_l1_std","iolh_l1_std","icms_l1_std","icmsht_l1_std"])
    counter =0
    for d in ([4,6,8,10,12,16]):
        l1.fill(0.0)
        misc_vars = MARGINAL_VARS(d, k, e_eps)
        for itr in (range(rpt)):
            print (d, itr)
            print ("computing marginals.")
            irr_l1, mrr_l1, iht_l1, mht_l1, ips_l1, mps_l1, iolh_l1,icms_l1,icmsht_l1 = driver(d,k,e_eps,population,misc_vars)
            l1[itr] = np.array([irr_l1, mrr_l1, iht_l1, mht_l1, ips_l1, mps_l1,iolh_l1,icms_l1,icmsht_l1])
            print (l1[itr])
        conf = [population, d, k, e_eps]
        conf.extend(l1.mean(axis=0))
        dfmean.loc[counter] = conf
        dfstd.loc[counter] = l1.std(axis=0)
        #print (conf)
        counter += 1

    dfstdcols = list(dfstd.columns.values)
    for c in dfstdcols:
        dfmean[c] = dfstd[c]

    dfmean.fillna(0.0,inplace=True)
    dfmean.to_pickle("data/all_mechanisms_vary_d.pkl")

'''
Call this method when you want to vary eps, d and k holding N fixed.
'''
def driver_vary_all():
    rpt = 5
    e_eps_arr = np.array([1.1,1.6,2.1,2.5, 3.0,3.5])
    counter=0
    ## Parameters for sketches
    width = 256
    no_hash = 5
    l1 = np.zeros((rpt, 9))

    dfmean = pd.DataFrame(columns=["population", "d", "k", "e_eps", "irr_l1", "mrr_l1", "iht_l1", "mht_l1", "ips_l1", "mps_l1","iolh_l1","icms_l1","icmsht_l1"])
    dfstd = pd.DataFrame(columns=["irr_l1_std", "mrr_l1_std", "iht_l1_std", "mht_l1_std", "ips_l1_std", "mps_l1_std","iolh_l1_std","icms_l1_std","icmsht_l1_std"])
    for population in [np.power(2,16)]:
        for k in reversed([1,2,3]):
            for e_eps in e_eps_arr:
                for d in ([4,8,16]):
                    misc_vars = MARGINAL_VARS(d,k,e_eps)
                    l1.fill(0.0)
                    print ("------------------")
                    for itr in range(0,rpt):
                        print (d, itr)
                        irr_l1, mrr_l1, iht_l1, mht_l1, ips_l1, mps_l1, iolh_l1,icms_l1,icmsht_l1 = driver(d,k,e_eps,population,misc_vars)
                        l1[itr] = np.array([irr_l1, mrr_l1, iht_l1, mht_l1, ips_l1, mps_l1,iolh_l1,icms_l1,icmsht_l1])
                        print (l1[itr])
                    conf = [population,d,k,e_eps]
                    conf.extend(l1.mean(axis=0))
                    dfmean.loc[counter]= conf
                    dfstd.loc[counter] = l1.std(axis=0)
                    print (conf)
                    counter+=1

    dfstdcols = list(dfstd.columns.values)
    for c in dfstdcols:
        dfmean[c] = dfstd[c]

    print (dfmean)

    dfmean.to_pickle("data/all_mechanisms_vary_all.pkl")


if __name__ == "__main__":
    ### Vary the total number of questions i.e. d.
    vary_d()
    ### Vary k, the number of subset of attributes we are interested in.
    #vary_k()

    #driver_vary_all()
