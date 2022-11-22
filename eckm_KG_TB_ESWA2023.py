import numpy as np
import pandas as pd
import time
import os
import sys
import heapq
from scipy.spatial import distance
import math
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score 
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
import random
import shutil
from sklearn import datasets
from sklearn.metrics import *
from sklearn.cluster import *
#from fcmeans import FCM
from scipy.spatial.distance import hamming
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cityblock

#Target Data (Benchmark)
from sklearn.datasets import load_iris
scikitdata = load_iris()

# Matplotlib Warning Message Eliminator 
plt.rcParams.update({'figure.max_open_warning': 0})

#Randam Seed Value, Epsilon, SCIKIT data nature, PCA control
state = 40
eps = 0.001
scdt = 0 # for scikit data, scdt = 1
pca = 0 # 0 for no PCA and 1 for PCA
io = 0 # 1 for all circles and 0 for all interior hull circles 


# Paths and Parent Directory
owd = os.getcwd()
datafilepath = "C:/Users/NITTTR/Desktop/Bk_Desktop/Submited/Paper1/Github/compound_data.txt"
#Give Your Original Path of the text file of data set
targetfilepath = "C:/Users/NITTTR/Desktop/Bk_Desktop/Submited/Paper1/Github/compound_target.txt"
#Give Your Original Path of tne target label of the data set
pre_basepath = os.path.join(owd,"RUNS")
isdir = os.path.isdir(pre_basepath)
if isdir == False:
    os.makedirs(pre_basepath)
controlpath = owd
inputparapath = owd
rstcon = sys.stdout

print("----------------------------")
print("Please, Follow Instructions")
print("----------------------------")

# Controls of the CODE
infs_str = input("Type 1 for Terminal Inputs OR Type 0 for File Inputs: ")
infs = int(infs_str)
if infs == 1:  
    outfs_str = input("Type 1 for Terminal Outputs OR Type 0 for File Outputs: ")
    outfs = int(outfs_str)
    version_str = input("Type 1 for Full Version OR Type 0 for Lite Version: ")
    version = int(version_str)
    datasource_str = input("Type 0 for Generating Data OR Type 1 for Reading Datafile:  ")
    datasource = int(datasource_str)
    comcal_str = input("Type 0 for Completeness Calculation OR Type 1 for Avoiding it")
    comcal = int(comcal_str)
    homcal_str = input("Type 0 for Homogeneity Calculation OR Type 1 for Avoiding it")
    homcal = int(homcal_str)
    accal_str = input("Type 0 for Accuracy Calculation OR Type 1 for Avoiding it")
    accal = int(accal_str)
    ercal_str = input("Type 0 for Error Calculation OR Type 1 for Avoiding it")
    ercal = int(ercal_str)
    keydim_str = input("Type 0 for Raw Data KM OR Type 1 PCA data KM")
    keydim = int(keydim_str)
    ermy_str = input("Type 0 for Not Sorting labels OR Type 1 Sorting Labels")
    ermy = int(ermy_str)
    noc_str = input("Pleae, Give number of attributes: ")
    attributes = int(noc_str)
    runcount_str = input("Pleae, Give number of runs: ")
    runcount = int(runcount_str)
else:
    os.chdir(controlpath)
    confile = open('control.txt', 'r')
    con=[]
    for line in confile:
        name, value = line.split("=")
        value = value.strip()
        con.append(int(value))
    infs=con[0]
    outfs=con[1]
    version=con[2]
    datasource=con[3]
    comcal=con[4]
    homcal=con[5]
    accal=con[6]
    ercal=con[7]
    keydim=con[8]
    ermy=con[9]
    attributes=con[10]
    runcount=con[11]
    confile.close()      

rstcon = sys.stdout
start=time.time()

# Dedicated Directory Creation for Outputs
readfile=os.path.basename(datafilepath)
if datasource == 0:
    foldername = "Generated_Data"
else:
    if scdt ==0:
        foldername = os.path.splitext(readfile)[0]
    else:
        foldername = "SCKITDATA"
base_path = os.path.join(pre_basepath, foldername)
isdir = os.path.isdir(base_path)  
if isdir == True:
    shutil.rmtree(base_path)
os.makedirs(base_path)

if outfs == 0:
     os.chdir(base_path)
     fsumm = open('summary.txt', 'w+')
     sys.stdout = fsumm

# Run List and Method Counts
runlist = list(range(runcount))
metcount = 7
metlist =list(range(metcount))

# Parameters Reading from Input File
if infs == 0:
    os.chdir(inputparapath)
    infile = open('input.txt', 'r')
    para=[]
    for line in infile:
            name, value = line.split("=")
            value = value.strip()
            para.append(int(value))
    selection=para[0]
    cfi=para[1]
    seleck=para[2]
    fignature=para[3]
    kmax=para[4]
    kstat=para[5]
    ds=para[6]
    nicc=para[7]
    infile.close()      
else:
    sys.stdout = rstcon
    selection_str = input("Type 0 for Generating Data OR Type 1 for Reading Datafile:  ")
    selection = int(selection_str)
    cfi_str = input("Please, Give maximum iteration numbers for circles finding:  ")
    cfi = int(cfi_str)
    seleck_str = input("Type 0 for static k OR Type 1 for Silhouette Optimum k:  ")
    seleck=int(seleck_str)
    fignature_str = input("Type 0 for Multiple Figs Fig OR Type 1 for Single Frame Fig:  ")
    fignature=int(fignature_str)
    kmax_str = input("Please, supply Kmax for Silhouette Scores: ")
    kmax = int(kmax_str)
    kstat_str = input("Desire Number of Clusters (static): ")
    kstat = int(kstat_str)   
    ds_str = input("Enter size of data points for Generated Data: ")
    ds = int(ds_str)
    nicc_str = input("Enter number of initial cluster centers for Generated Data: ")
    nicc = int(nicc_str)

# Loop Starts for Multiple Runs
iters = [[0 for jj in metlist] for ii in runlist]
siscores = [[0 for nn in metlist] for kk in runlist]
dbscores = [[0 for nn in metlist] for kk in runlist]
misclass = [[0 for nn in metlist] for kk in runlist]
counter = [[0 for nn in metlist] for kk in runlist]
error = [[0 for nn in metlist] for kk in runlist]
accur = [[0 for nn in metlist] for kk in runlist]
completeness = [[0 for nn in metlist] for kk in runlist]
homogen = [[0 for nn in metlist] for kk in runlist]
s_misclass = [[0 for nn in metlist] for kk in runlist]
s_error = [[0 for nn in metlist] for kk in runlist]
s_accur = [[0 for nn in metlist] for kk in runlist]
s_completeness = [[0 for nn in metlist] for kk in runlist]
s_homogen = [[0 for nn in metlist] for kk in runlist]
myerror = [[0 for nn in metlist] for kk in runlist]
mymisclass = [[0 for nn in metlist] for kk in runlist]
hamm = [[0 for nn in metlist] for kk in runlist]
eucli = [[0 for nn in metlist] for kk in runlist]
manhat = [[0 for nn in metlist] for kk in runlist]
euclid = [[0 for nn in metlist] for kk in runlist]
manhatt = [[0 for nn in metlist] for kk in runlist]

for irun in runlist:
    startirun=time.time()
    # Dynamic Folder Creation for FULL Version 
    if version == 0:
        if irun == 0:
            dir_path = os.path.join(base_path, "All_Outs")
            if isdir == True:
                shutil.rmtree(dir_path)
            os.makedirs(dir_path)
        if outfs == 0:
            os.chdir(dir_path)
            fmout = open('output_%s.txt' % str(irun+1), 'w+')
    else:
        dir_path = '{}/Run_{}'.format(base_path, irun+1)
        isdir = os.path.isdir(dir_path)  
        if isdir == True:
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
        if outfs == 0:
            os.chdir(dir_path)
            fout = open("output.txt", 'w+')
            
    os.chdir(base_path)       
    if irun == 0:
        if outfs == 0:
            sys.stdout = fsumm           
        else:
            sys.stdout = rstcon
        
        if selection == 0:
            #Genreration of Data and Writing in a file by you
            file = open('generated_data.txt','w')
            gendata_path = os.path.join(os.getcwd(),'generated_data.txt')
            x, y = make_blobs(n_samples = ds, centers = nicc, n_features=5, shuffle=True, random_state=state)
            x_raw = x
            targ = y
            pca = PCA(n_components=2)
            x_pca = pca.fit_transform(x_raw)
            for i in range(len(x_raw)):
                file.write(str(x[i,0])+"  "+str(x[i,1])+"  "+str(x[i,2])+"  "+str(x[i,3])+"  "+str(x[i,4]) +'\n')
            file.close()
            os.chdir(base_path)  
            print('--------------------------------------------')
            print("You are generating data randomly!")
            print('--------------------------------------------')
            print("The path of generated datafile is: ", gendata_path)
            print('')
            if version == 0:
                print('You are using the LITE version of the CODE')
            else:
                print('You are using the LITE version of the CODE')
            print('')
            print("Generated Data Size: ", ds)
            print("Maximum iteration numbers for circles finding:  ", cfi)
            print("Kmax for finding Silhouette Scores: ",kmax)
            print("Number of Proposed Cluster Centers for Generated Data: ",nicc)
            print("Desire Number of Clusters (Static): ", kstat)
            print("You have selected for Singleframe Fig for Comparison")
            if fignature == 0:
                print("You have selected for Multiple frame Fig for Comparison")
            else:
                print("You have selected for Single frame Fig for Comparison")
            print("Maximum iteration numbers for circles finding:  ", cfi)
            
        else:
            #Reading Data from an External Datafile
            if scdt == 0:
                #TXT FILE
                read_data = np.loadtxt(fname = datafilepath)
                #read_data = np.loadtxt(fname = datafilepath, usecols = (0,1))
                #tar_data = np.loadtxt(fname = datafilepath,usecols = (2))
                #tar_data = np.loadtxt(fname = targetfilepath)
                main_data = np.array(read_data)
                targ = scikitdata.target
                #targ= np.array(tar_data)-1

                #CSV FILE
                #read_data = pd.read_csv(datafilepath,sep=",", header=None)           
                #main_data=np.array(read_data.drop(axis=1, columns=[3]))
                #targ=(np.array((read_data[3])-1))

            else:
                main_data = scikitdata.data
                targ = scikitdata.target
            x_raw = main_data
            #scaler = StandardScaler()
            #scaler.fit(x)
            #scaled_data = scaler.fit_transform(x)
            #print(scaled_data)
            if pca == 0:
                x_pca = x_raw
            else:
                pca = PCA(n_components=2)
                x_pca = pca.fit_transform(x_raw)
            ds=len(x_raw)
            print('--------------------------------------------')
            if scdt == 0:
                print("You have read data from the file: ",readfile)
            else:
                print("You have read data from SCIKITLEARN Inbuilt")
            print('--------------------------------------------')
            if scdt == 0:
                print("The path of supplied datafile is: ", datafilepath)
            else:
                print("Find the file from load_datafilename")
            print('')
            print("The reduced 2D PCA components shape is: ",x_pca.shape)
            if version == 0:
                print('You are using the LITE version of the CODE')
            else:
                print('You are using the LITE version of the CODE')
            print('')
            print("Datafile Data Size: ", ds)
            print("Maximum iteration numbers for circles finding:  ", cfi)
            print("Kmax for finding Silhouette Scores: ",kmax)
            print("Desire Number of Clusters (Static): ", kstat)
            if fignature == 0:
                print("You have selected for Multiple frame Fig for Comparison")
            else:
                print("You have selected for Single frame Fig for Comparison")
            print("Maximum iteration numbers for circles finding:  ", cfi)

        #Scattering plot of Dataset  
        os.chdir(base_path)
        fig, ax = plt.subplots()
        #ax.plot(x_raw[:,0], x_raw[:,1],"k.")
        ax.scatter(x_pca[:,0], x_pca[:,1]) 
        ax.set(xlabel='X', ylabel='Y', title='Dataset')
        fig.savefig("Data")
    
        points = x_pca
        X = x_pca
        target = targ
        s_target = np.sort(target)
        ctrue = Counter(target)
        ta = list(ctrue.values())
        
        #The Silhouette Method for finding K
        no_of_clusters = np.arange(2,kmax+1)
        sil = []
        print('-------------------------')
        print("K  :  Silhouette Scores")
        print('-------------------------')
        for n_clusters in no_of_clusters:   
            cluster = KMeans(n_clusters = n_clusters) 
            cluster_labels = cluster.fit_predict(X) 
            silhouette_avg = silhouette_score(X, cluster_labels)
            print(n_clusters, " : ", silhouette_avg)
            sil.append(silhouette_avg)
        s1 = np.array(no_of_clusters) 
        s2 = np.array(sil) 
        os.chdir(base_path)
        fig, ax = plt.subplots()
        ax.plot(s1, s2,"gs-")
        ax.set(xlabel='Values of k', ylabel='Silhouette Score', title='The Silhouette Diagram for Optimal k')
        fig.savefig("Initial_Silhouette_Test")
    
        max_value = max(sil)
        max_index = sil.index(max_value)
        opt_clus = no_of_clusters[max_index]
    
        #Selection of K and Subsequently number of Circles
        if seleck == 0:
            noc = kstat
            print('\n'+"Optimal k for this dataset is",opt_clus,',but here using static k = ',kstat)
        else:
            noc = opt_clus
            print('\n'+"Optimal k for the dataset is",opt_clus,',So, you are using this k=', opt_clus)
        nlarge = math.ceil(noc/3.0)   
        nsmall = math.ceil(noc/3.0)

        #Computational Geometry Part
        vor = Voronoi(points)
        hull = ConvexHull(points)
        myList1=vor.vertices
        myList= points[hull.vertices]                  
        outsideVor=[[] for i in range(0,len(vor.vertices))]
        nat=points.shape[1]
        insideVor=[]
        b= myList
        for i in range(0,len(myList1)):
         x=(np.append(b,myList1[i]))
         x=np.array(np.reshape(np.array([x]), (-1, nat)))
         hull2=ConvexHull(x)
         oldHull = np.array(points[hull.vertices])
         newHull = np.array(x[hull2.vertices])
         if np.array_equal(oldHull,newHull):
            insideVor.append(myList1[i])
         else:
             continue  
        in_vor1=np.array(insideVor)
        out_vor=np.array([elem for elem in myList1 if elem not in in_vor1])
        if io == 0:
            in_vor = in_vor1
        else:
            in_vor = myList1
        rad_dist=(distance.cdist(in_vor,points,metric='euclidean'))
        radlist=[]
        for ii in range(len(rad_dist)):
            radlist.append(min(rad_dist[ii]))
        radlist = np.array(radlist)
        
        #LECs and SECs
        circlefind = list(range(1,cfi))
        for loop in circlefind:
            rad_lrg=np.array(heapq.nlargest(nlarge,radlist))
            rad_small=np.array(heapq.nsmallest(nsmall,radlist))
            cen_idx=heapq.nlargest(nlarge,range(len(radlist)), radlist.take) 
            sm_cen_idx=heapq.nsmallest(nsmall,range(len(radlist)), radlist.take)
            lar_cen=in_vor[cen_idx]
            sm_cen=in_vor[sm_cen_idx]
            dst=np.array(distance.cdist(lar_cen,points,'euclidean'))
            sm_dst=np.array(distance.cdist(sm_cen,points,'euclidean'))
            ncrangel=np.arange(0,len(rad_lrg),1)
            ncranges=np.arange(0,len(rad_small),1)
            idx=[[] for i in ncrangel]
            sidx=[[] for i in ncranges]
            for ncount in ncrangel:
                idx[ncount]=((np.where((np.round([abs(dst[ncount]-rad_lrg[ncount])],3))< eps))[1].tolist())
            for ncount in ncranges:
                sidx[ncount]=((np.where((np.round([abs(sm_dst[ncount]-rad_small[ncount])],3))<eps))[1].tolist())
        
            #Indexing Part
            all_idx=[]
            sm_all_idx=[]
            xidx=[i for i in idx if i != []]
            xsidx=[i for i in sidx if i != []]
            for ncount in ncrangel:
                all_idx=all_idx + xidx[ncount]
            for ncount in ncranges:
                sm_all_idx=sm_all_idx + xsidx[ncount]
                
            #Feasibility Checking    
            res = [] 
            for i in all_idx: 
                if i not in res: 
                    res.append(i)         
            sm_res = [] 
            for i in sm_all_idx: 
                if i not in sm_res: 
                    sm_res.append(i)
            
            if len(res) > noc and len(sm_res) > noc:
                break
            elif len(res) < noc and len(sm_res) > noc:
                nlarge += 1
            elif len(res) > noc and len(sm_res) < noc:
                nsmall += 1
            else :
                nlarge += 1
                nsmall += 1 
        if len(res) < noc or len(sm_res) < noc:
            sys.exit("Sorry, Error is there, increase the the length of circlefind list")
        if irun == 0:    
            print("---------------------------------------------")
            print("LEC and SEC numbers and Clusters no. Details ")
            print("---------------------------------------------")
            print("Number of taken LECs: ", nlarge)
            print("For these LECs, maximum number of clusters allowed: ", len(res))
            print("Number of taken SECs: ", nsmall)
            print("For these SECs, maximum number of clusters allowed: ", len(sm_res))
            
    circumpts_all = points[res]
    sm_circumpts_all=points[sm_res]

    barl=[]
    bars=[]
    barch=[]
    barkm=[] 
    loopss = []
    loopdb = []
    loopcom = []
    loophom = []
    loopmisc = []
    loopac = []
    looper = []
    s_loopcom = []
    s_loophom = []
    s_loopmisc = []
    s_loopac = []
    s_looper = []
    loopham = []
    loopeuc = []
    loopman = []
    mymis = []
    myerr = []
    looped = []
    loopmh = []
    
    #LEC and SEC Method (M1)
    strt_l_m1=np.array(circumpts_all[:noc])
    strt_s_m1=np.array(sm_circumpts_all[:noc])
    kmeans1=KMeans(n_clusters=noc,init=strt_l_m1,n_init=1)
    sm_kmeans1=KMeans(n_clusters=noc,init=strt_s_m1,n_init=1)
    kmeans1.fit(points)
    sm_kmeans1.fit(points)
    labels_m1 = kmeans1.labels_
    end_l_m1 = kmeans1.cluster_centers_
    end_s_m1 = sm_kmeans1.cluster_centers_
    sort_labels_m1 = np.sort(labels_m1)
    centroidl = end_l_m1
    centroids = end_s_m1
    sm_labels_m1 = sm_kmeans1.labels_
    sort_sm_labels_m1 = np.sort(sm_labels_m1)
    sc_m1 = np.round(silhouette_score(X, labels_m1, metric='euclidean', random_state=None),2)
    sm_sc_m1 = np.round(silhouette_score(X, sm_labels_m1, metric='euclidean', random_state=None),2)
    db_m1 = np.round(davies_bouldin_score(X, labels_m1),2)
    sm_db_m1 = np.round(davies_bouldin_score(X, sm_labels_m1),2)
    barl.append(kmeans1.n_iter_)
    bars.append(sm_kmeans1.n_iter_)
    loopss.extend([sc_m1,sm_sc_m1])
    loopdb.extend([db_m1,sm_db_m1])
    clm1 = Counter(labels_m1)
    sm_clm1 = Counter(sm_labels_m1)
    
    if accal == 1:
        ccm1 = []
        bbm1 = []
        for ik in range(0,noc):
            indexm1=(np.where(labels_m1 == ik)[0].tolist())
            ccm1.append(indexm1)
        for jk in range(len(ccm1)):
            pp=X[ccm1[jk]]
            bbm1.append(pp)
        ccsm_m1 = []
        bbsm_m1 = []
        for ik in range(0,noc):
            indexsm_m1=(np.where(sm_labels_m1 == ik)[0].tolist())
            ccsm_m1.append(indexsm_m1)
        for jk in range(len(ccsm_m1)):
            pp=X[ccsm_m1[jk]]
            bbsm_m1.append(pp)
        dist1 = 0
        mdist1 = 0
        for j in range(noc):
            for i in range(len(bbm1[j])):
              dist1 = dist1 + distance.euclidean(centroidl[j],bbm1[j][i])
              mdist1 = mdist1 + distance.cityblock(centroidl[j],bbm1[j][i])
        dist2 = 0
        mdist2 = 0
        for j in range(noc):
            for i in range(len(bbsm_m1[j])):
              dist2 = dist2 + distance.euclidean(centroids[j],bbsm_m1[j][i])
              mdist2 = mdist2 + distance.cityblock(centroids[j],bbsm_m1[j][i])
        looped.extend([dist1,dist2])
        loopmh.extend([mdist1,mdist2])
    
    if comcal == 1: 
        completeness_m1 = completeness_score(target, labels_m1)
        sm_completeness_m1 = completeness_score(target, sm_labels_m1)
        s_completeness_m1 = completeness_score(s_target, sort_labels_m1)
        s_sm_completeness_m1 = completeness_score(s_target, sort_sm_labels_m1)
        com1 = np.round(completeness_m1,2)
        s_com1 = np.round(s_completeness_m1,2)
        sm_com1 = np.round(sm_completeness_m1,2)
        s_sm_com1 = np.round(s_sm_completeness_m1,2)
        loopcom.extend([com1,sm_com1])
        s_loopcom.extend([s_com1,s_sm_com1])

    if homcal == 1:
        homogeneity_m1 = homogeneity_score(target, labels_m1)
        sm_homogeneity_m1 = homogeneity_score(target, sm_labels_m1)
        s_homogeneity_m1 = homogeneity_score(s_target, sort_labels_m1)
        s_sm_homogeneity_m1 = homogeneity_score(s_target, sort_sm_labels_m1)
        hom1 = np.round(homogeneity_m1,2)
        s_hom1 = np.round(homogeneity_m1,2)
        sm_hom1 = np.round(sm_homogeneity_m1,2)
        s_sm_hom1 = np.round(s_sm_homogeneity_m1,2)
        loophom.extend([hom1,sm_hom1])
        s_loophom.extend([s_hom1,s_sm_hom1])
        
    if accal == 1:
        accuracy_m1 = accuracy_score(target, labels_m1)
        sm_accuracy_m1 = accuracy_score(target, sm_labels_m1)
        s_accuracy_m1 = accuracy_score(s_target, sort_labels_m1)
        s_sm_accuracy_m1 = accuracy_score(s_target, sort_sm_labels_m1)
        ac1 = np.round(accuracy_m1,2)
        sm_ac1 = np.round(sm_accuracy_m1,2)
        s_ac1 = np.round(s_accuracy_m1,2)
        s_sm_ac1 = np.round(s_sm_accuracy_m1,2)
        loopac.extend([ac1,sm_ac1])
        s_loopac.extend([s_ac1,s_sm_ac1])
        hamdist_m1 = np.round(hamming(target,labels_m1))
        eucdist_m1 = np.round(euclidean(target, labels_m1))
        mandist_m1 = np.round(cityblock(target, labels_m1))
        sm_hamdist_m1 = np.round(hamming(target,labels_m1))
        sm_eucdist_m1 = np.round(euclidean(target, sm_labels_m1))
        sm_mandist_m1 = np.round(cityblock(target, sm_labels_m1))
        loopham.extend([hamdist_m1,sm_hamdist_m1])
        loopman.extend([mandist_m1,sm_mandist_m1])
        loopeuc.extend([eucdist_m1,sm_eucdist_m1])
        
    if ercal == 1:
        misclass_m1 = 0
        for ii in range(len(target)):
            diff = target[ii] - labels_m1[ii]
            if diff != 0:
                misclass_m1  = misclass_m1+1
        sm_misclass_m1 = 0
        for ii in range(len(target)):
            diff = target[ii] - sm_labels_m1[ii]
            if diff != 0:
                sm_misclass_m1  = sm_misclass_m1+1
        s_misclass_m1 = 0
        for ii in range(len(target)):
            diff = s_target[ii] - sort_labels_m1[ii]
            if diff != 0:
                s_misclass_m1  = s_misclass_m1+1
        s_sm_misclass_m1 = 0
        for ii in range(len(target)):
            diff = s_target[ii] - sort_sm_labels_m1[ii]
            if diff != 0:
                s_sm_misclass_m1  = s_sm_misclass_m1+1
        er1 = np.round((misclass_m1/len(target)),2)
        sm_er1 = np.round((sm_misclass_m1/len(target)),2)
        s_er1 = np.round((s_misclass_m1/len(target)),2)
        s_sm_er1 = np.round((s_sm_misclass_m1/len(target)),2)
        loopmisc.extend([misclass_m1,sm_misclass_m1])
        looper.extend([er1,sm_er1])
        s_loopmisc.extend([s_misclass_m1,s_sm_misclass_m1])
        s_looper.extend([s_er1,s_sm_er1])
        

    if ermy == 1:
        tp = list(clm1.values())
        difference = []
        zip_object = zip(tp, ta)
        for i, j in zip_object:
            difference.append(abs(i-j))
        mis1 = np.sum(difference)
        myer1 = np.round((mis1/len(target)*100),2)
        tp = list(sm_clm1.values())
        difference = []
        zip_object = zip(tp, ta)
        for i, j in zip_object:
            difference.append(abs(i-j))
        mis2 = np.sum(difference)
        myer2 = np.round((mis2/len(target)*100),2)
        mymis.extend([mis1,mis2])
        myerr.extend([myer1,myer2])
    
    #LEC and SEC Method (M2)
    strt_l_m2=np.array(random.sample(list(circumpts_all),noc))
    strt_s_m2=np.array(random.sample(list(sm_circumpts_all),noc))
    kmeans2=KMeans(n_clusters=noc,init=strt_l_m2,n_init=1)
    sm_kmeans2=KMeans(n_clusters=noc,init=strt_s_m2,n_init=1)
    kmeans2.fit(points)
    sm_kmeans2.fit(points)
    labels_m2 = kmeans2.labels_
    sort_labels_m2 = np.sort(labels_m2)
    sm_labels_m2 = sm_kmeans2.labels_
    sort_sm_labels_m2 = np.sort(sm_labels_m2)
    end_l_m2 = kmeans2.cluster_centers_
    end_s_m2 = sm_kmeans2.cluster_centers_
    centroidl = end_l_m2
    centroids = end_s_m2
    sc_m2 = np.round(silhouette_score(X, labels_m2, metric='euclidean', random_state=None),2)
    sm_sc_m2 = np.round(silhouette_score(X, sm_labels_m2, metric='euclidean', random_state=None),2)
    db_m2 = np.round(davies_bouldin_score(X, labels_m2),2)
    sm_db_m2 = np.round(davies_bouldin_score(X, sm_labels_m2),2)
    barl.append(kmeans2.n_iter_)
    bars.append(sm_kmeans2.n_iter_)
    loopss.extend([sc_m2,sm_sc_m2])
    loopdb.extend([db_m2,sm_db_m2])
    clm2 = Counter(labels_m2)
    sm_clm2 = Counter(sm_labels_m2)

    if accal == 1:
        ccm2 = []
        bbm2 = []
        for ik in range(0,noc):
            indexm2=(np.where(labels_m2 == ik)[0].tolist())
            ccm2.append(indexm2)
        for jk in range(len(ccm2)):
            pp=X[ccm2[jk]]
            bbm2.append(pp)
        ccsm_m2 = []
        bbsm_m2 = []
        for ik in range(0,noc):
            indexsm_m2=(np.where(sm_labels_m2 == ik)[0].tolist())
            ccsm_m2.append(indexsm_m2)
        for jk in range(len(ccsm_m2)):
            pp=X[ccsm_m2[jk]]
            bbsm_m2.append(pp)
        dist3 = 0
        mdist3 = 0
        for j in range(noc):
            for i in range(len(bbm2[j])):
              dist3 = dist3 + distance.euclidean(centroidl[j],bbm2[j][i])
              mdist3 = mdist3 + distance.cityblock(centroidl[j],bbm2[j][i])
        dist4 = 0
        mdist4 = 0
        for j in range(noc):
            for i in range(len(bbsm_m2[j])):
              dist4 = dist4 + distance.euclidean(centroids[j],bbsm_m2[j][i])
              mdist4 = mdist4 + distance.cityblock(centroids[j],bbsm_m2[j][i])
        looped.extend([dist3,dist4])
        loopmh.extend([mdist3,mdist4])
    
    if comcal == 1: 
        completeness_m2 = completeness_score(target, labels_m2)
        sm_completeness_m2 = completeness_score(target, sm_labels_m2)
        s_completeness_m2 = completeness_score(s_target, sort_labels_m2)
        s_sm_completeness_m2 = completeness_score(s_target, sort_sm_labels_m2)
        com2 = np.round(completeness_m2,2)
        s_com2 = np.round(s_completeness_m2,2)
        sm_com2 = np.round(sm_completeness_m2,2)
        s_sm_com2 = np.round(s_sm_completeness_m2,2)
        loopcom.extend([com2,sm_com2])
        s_loopcom.extend([s_com2,s_sm_com2])

    if homcal == 1:
        homogeneity_m2 = homogeneity_score(target, labels_m2)
        sm_homogeneity_m2 = homogeneity_score(target, sm_labels_m2)
        s_homogeneity_m2 = homogeneity_score(s_target, sort_labels_m2)
        s_sm_homogeneity_m2 = homogeneity_score(s_target, sort_sm_labels_m2)
        hom2 = np.round(homogeneity_m2,2)
        s_hom2 = np.round(homogeneity_m2,2)
        sm_hom2 = np.round(sm_homogeneity_m2,2)
        s_sm_hom2 = np.round(s_sm_homogeneity_m2,2)
        loophom.extend([hom2,sm_hom2])
        s_loophom.extend([s_hom2,s_sm_hom2])
        
    if accal == 1:
        accuracy_m2 = accuracy_score(target, labels_m2)
        sm_accuracy_m2 = accuracy_score(target, sm_labels_m2)
        s_accuracy_m2 = accuracy_score(s_target, sort_labels_m2)
        s_sm_accuracy_m2 = accuracy_score(s_target, sort_sm_labels_m2)
        ac2 = np.round(accuracy_m2,2)
        sm_ac2 = np.round(sm_accuracy_m2,2)
        s_ac2 = np.round(s_accuracy_m2,2)
        s_sm_ac2 = np.round(s_sm_accuracy_m2,2)
        loopac.extend([ac2,sm_ac2])
        s_loopac.extend([s_ac2,s_sm_ac2])
        hamdist_m2 = np.round(hamming(target,labels_m2))
        eucdist_m2 = np.round(euclidean(target, labels_m2))
        mandist_m2 = np.round(cityblock(target, labels_m2))
        sm_hamdist_m2 = np.round(hamming(target,labels_m2))
        sm_eucdist_m2 = np.round(euclidean(target, sm_labels_m2))
        sm_mandist_m2 = np.round(cityblock(target, sm_labels_m2))
        loopham.extend([hamdist_m2,sm_hamdist_m2])
        loopman.extend([mandist_m2,sm_mandist_m2])
        loopeuc.extend([eucdist_m2,sm_eucdist_m2])
    
    if ercal == 1:
        misclass_m2 = 0
        for ii in range(len(target)):
            diff = target[ii] - labels_m2[ii]
            if diff != 0:
                misclass_m2  = misclass_m2+1
        sm_misclass_m2 = 0
        for ii in range(len(target)):
            diff = target[ii] - sm_labels_m2[ii]
            if diff != 0:
                sm_misclass_m2  = sm_misclass_m2+1
        s_misclass_m2 = 0
        for ii in range(len(target)):
            diff = s_target[ii] - sort_labels_m2[ii]
            if diff != 0:
                s_misclass_m2  = s_misclass_m2+1
        s_sm_misclass_m2 = 0
        for ii in range(len(target)):
            diff = s_target[ii] - sort_sm_labels_m2[ii]
            if diff != 0:
                s_sm_misclass_m2  = s_sm_misclass_m2+1
        er2 = np.round((misclass_m2/len(target)),2)
        sm_er2 = np.round((sm_misclass_m2/len(target)*100),2)
        s_er2 = np.round((s_misclass_m2/len(target)),2)
        s_sm_er2 = np.round((s_sm_misclass_m2/len(target)),2)
        loopmisc.extend([misclass_m2,sm_misclass_m2])
        looper.extend([er2,sm_er2])
        s_loopmisc.extend([s_misclass_m2,s_sm_misclass_m2])
        s_looper.extend([s_er2,s_sm_er2])

    if ermy == 1:
        tp = list(clm2.values())
        difference = []
        zip_object = zip(tp, ta)
        for i, j in zip_object:
            difference.append(abs(i-j))
        mis1 = np.sum(difference)
        myer1 = np.round((mis1/len(target)*100),2)
        tp = list(sm_clm2.values())
        difference = []
        zip_object = zip(tp, ta)
        for i, j in zip_object:
            difference.append(abs(i-j))
        mis2 = np.sum(difference)
        myer2 = np.round((mis2/len(target)*100),2)
        mymis.extend([mis1,mis2])
        myerr.extend([myer1,myer2])

                        
    #CH Method
    Y=myList.tolist()
    strt_ch=np.array(random.sample(Y,noc))
    kmeansch=KMeans(n_clusters=noc,init=strt_ch,n_init=1)
    kmeansch.fit(points)
    labels_ch = kmeansch.labels_
    sort_labels_ch = np.sort(labels_ch)
    end_ch = kmeans2.cluster_centers_
    centroid = end_ch
    sc_ch = np.round(silhouette_score(X, labels_ch, metric='euclidean', random_state=None),2)
    db_ch = np.round(davies_bouldin_score(X, labels_ch),2)
    barch.append(kmeansch.n_iter_)
    loopss.append(sc_ch)
    loopdb.append(db_ch)
    clch = Counter(labels_ch)

    if accal == 1:
        ccch = []
        bbch = []
        for ik in range(0,noc):
            indexch=(np.where(labels_ch == ik)[0].tolist())
            ccch.append(indexch)
        for jk in range(len(ccch)):
            pp=X[ccch[jk]]
            bbch.append(pp)
        dist6 = 0
        mdist6 = 0
        for j in range(noc):
            for i in range(len(bbch[j])):
              dist6 = dist6 + distance.euclidean(centroid[j],bbch[j][i])
              mdist6 = mdist6 + distance.cityblock(centroid[j],bbch[j][i])
        looped.extend([dist6])
        loopmh.extend([mdist6])
    
    if comcal == 1:
        completeness_ch = completeness_score(target, labels_ch)
        s_completeness_ch = completeness_score(s_target, sort_labels_ch)
        comch = np.round(completeness_ch,2)
        s_comch = np.round(s_completeness_ch,2)
        loopcom.extend([comch])
        s_loopcom.extend([s_comch])

    if homcal == 1:
        homogeneity_ch = homogeneity_score(target, labels_ch)
        s_homogeneity_ch = homogeneity_score(s_target, sort_labels_ch)           
        homch = np.round(homogeneity_ch,2)
        s_homch = np.round(s_homogeneity_ch,2)
        loophom.extend([homch])
        s_loophom.extend([s_homch])
        
    if accal == 1:
        accuracy_ch = accuracy_score(target, labels_ch)
        s_accuracy_ch = accuracy_score(s_target, sort_labels_ch)
        acch = np.round(accuracy_ch,2)
        loopac.extend([acch])
        s_acch = np.round(s_accuracy_ch,2)
        s_loopac.extend([s_acch])
        hamdist_ch = np.round(hamming(target,labels_ch))
        eucdist_ch = np.round(euclidean(target, labels_ch))
        mandist_ch = np.round(cityblock(target, labels_ch))
        loopham.extend([hamdist_ch])
        loopman.extend([mandist_ch])
        loopeuc.extend([eucdist_ch])
    
    if ercal == 1:
        misclass_ch = 0
        for ii in range(len(target)):
            diff = target[ii] - labels_ch[ii]
            if diff != 0:
                misclass_ch  = misclass_ch+1
        s_misclass_ch = 0
        for ii in range(len(target)):
            diff = s_target[ii] - sort_labels_ch[ii]
            if diff != 0:
                s_misclass_ch  = misclass_ch+1            
        erch = np.round((misclass_ch/len(target)),2)
        s_erch = np.round((s_misclass_ch/len(target)),2)
        loopmisc.extend([misclass_ch])
        looper.extend([erch])
        s_loopmisc.extend([s_misclass_ch])
        s_looper.extend([s_erch])

    if ermy == 1:
        tp = list(clch.values())
        difference = []
        zip_object = zip(tp, ta)
        for i, j in zip_object:
            difference.append(abs(i-j))
        mis1 = np.sum(difference)
        myer1 = np.round((mis1/len(target)*100),2)
        mymis.extend([mis1])
        myerr.extend([myer1])
    
    #AKM Method
    if keydim == 0:
        Z = x_raw
        points1 = x_raw
        X1 = x_raw
    else:
        Z = x_pca
        points1 = x_pca
        X1 = x_pca
    strt_akm=np.array(random.sample(list(Z),noc))
    kmeans=KMeans(n_clusters=noc,init=strt_akm,n_init=1)
    kmeans.fit(points1)
    labels_akm = kmeans.labels_
    end_akm = kmeans.cluster_centers_
    sort_labels_akm = np.sort(labels_akm)
    end_ch = kmeans2.cluster_centers_
    centroid = end_ch
    sc_akm = np.round(silhouette_score(X1, labels_akm, metric='euclidean', random_state=None),2)
    db_akm = np.round(davies_bouldin_score(X1, labels_akm),2)
    barkm.append(kmeans.n_iter_)
    loopss.append(sc_akm)
    loopdb.append(db_akm)
    clakm = Counter(labels_akm)

    if accal == 1:
        ccakm = []
        bbakm = []
        for ik in range(0,noc):
            indexakm=(np.where(labels_akm == ik)[0].tolist())
            ccakm.append(indexakm)
        for jk in range(len(ccakm)):
            pp=X[ccakm[jk]]
            bbakm.append(pp)
        dist6 = 0
        mdist6 = 0
        for j in range(noc):
            for i in range(len(bbakm[j])):
              dist6 = dist6 + distance.euclidean(centroid[j],bbakm[j][i])
              mdist6 = mdist6 + distance.cityblock(centroid[j],bbakm[j][i])
        looped.extend([dist6])
        loopmh.extend([mdist6])
    
    #MB Method
    kmeansmb=MiniBatchKMeans(n_clusters=noc,init=strt_akm,n_init=1)
    kmeansmb.fit(points1)
    labels_mb = kmeansmb.labels_
    end_mb = kmeansmb.cluster_centers_
    sort_labels_mb= np.sort(labels_mb)
    end_ch = kmeans2.cluster_centers_
    centroid = end_ch
    sc_mb = np.round(silhouette_score(X1, labels_mb, metric='euclidean', random_state=None),2)
    db_mb = np.round(davies_bouldin_score(X1, labels_mb),2)
    barkm.append(kmeansmb.n_iter_)
    loopss.append(sc_mb)
    loopdb.append(db_mb)
    clmb = Counter(labels_mb)

    if accal == 1:
        ccmb = []
        bbmb = []
        for ik in range(0,noc):
            indexmb=(np.where(labels_mb == ik)[0].tolist())
            ccmb.append(indexmb)
        for jk in range(len(ccmb)):
            pp=X[ccmb[jk]]
            bbmb.append(pp)
        dist7 = 0
        mdist7 = 0
        for j in range(noc):
            for i in range(len(bbmb[j])):
              dist7 = dist7 + distance.euclidean(centroid[j],bbmb[j][i])
              mdist7 = mdist7 + distance.cityblock(centroid[j],bbmb[j][i])
        looped.extend([dist7])
        loopmh.extend([mdist7])
    
    if comcal == 1:
        completeness_akm = completeness_score(target, labels_akm)
        s_completeness_akm = completeness_score(s_target, sort_labels_akm)
        comakm = np.round(completeness_akm,2)
        s_comakm = np.round(s_completeness_akm,2)
        loopcom.extend([comakm])
        s_loopcom.extend([s_comakm])
        completeness_mb = completeness_score(target, labels_mb)
        s_completeness_mb = completeness_score(s_target, sort_labels_mb)
        commb = np.round(completeness_mb,2)
        s_commb = np.round(s_completeness_mb,2)
        loopcom.extend([commb])
        s_loopcom.extend([s_commb])

    if homcal == 1:
        homogeneity_akm = homogeneity_score(target, labels_akm)
        s_homogeneity_akm = homogeneity_score(s_target, sort_labels_akm)           
        homakm = np.round(homogeneity_akm,2)
        s_homakm = np.round(s_homogeneity_akm,2)
        loophom.extend([homakm])
        s_loophom.extend([s_homakm])
        homogeneity_mb = homogeneity_score(target, labels_mb)
        s_homogeneity_mb = homogeneity_score(s_target, sort_labels_mb)           
        hommb = np.round(homogeneity_mb,2)
        s_hommb = np.round(s_homogeneity_mb,2)
        loophom.extend([hommb])
        s_loophom.extend([s_hommb])
        
    if accal == 1:
        accuracy_akm = accuracy_score(target, labels_akm)
        s_accuracy_akm = accuracy_score(s_target, sort_labels_akm)
        acakm = np.round(accuracy_akm,2)
        loopac.extend([acakm])
        s_acakm = np.round(s_accuracy_akm,2)
        s_loopac.extend([s_acakm])
        accuracy_mb = accuracy_score(target, labels_mb)
        s_accuracy_mb = accuracy_score(s_target, sort_labels_mb)
        acmb = np.round(accuracy_mb,2)
        loopac.extend([acmb])
        s_acmb = np.round(s_accuracy_mb,2)
        s_loopac.extend([s_acmb])
        hamdist_akm = np.round(hamming(target,labels_akm))
        eucdist_akm = np.round(euclidean(target, labels_akm))
        mandist_akm = np.round(cityblock(target, labels_akm))
        loopham.extend([hamdist_akm])
        loopman.extend([mandist_akm])
        loopeuc.extend([eucdist_akm])
        hamdist_mb = np.round(hamming(target,labels_mb))
        eucdist_mb = np.round(euclidean(target, labels_mb))
        mandist_mb = np.round(cityblock(target, labels_mb))
        loopham.extend([hamdist_mb])
        loopman.extend([mandist_mb])
        loopeuc.extend([eucdist_mb])
    
    if ercal == 1:
        misclass_akm = 0
        for ii in range(len(target)):
            diff = target[ii] - labels_akm[ii]
            if diff != 0:
                misclass_akm  = misclass_akm+1
        s_misclass_akm = 0
        for ii in range(len(target)):
            diff = s_target[ii] - sort_labels_akm[ii]
            if diff != 0:
                s_misclass_akm = misclass_akm+1            
        erakm = np.round((misclass_akm/len(target)),2)
        s_erakm = np.round((s_misclass_akm/len(target)),2)
        loopmisc.extend([misclass_akm])
        looper.extend([erakm])
        s_loopmisc.extend([s_misclass_akm])
        s_looper.extend([s_erakm])

        misclass_mb = 0
        for ii in range(len(target)):
            diff = target[ii] - labels_mb[ii]
            if diff != 0:
                misclass_mb  = misclass_mb+1
        s_misclass_mb = 0
        for ii in range(len(target)):
            diff = s_target[ii] - sort_labels_mb[ii]
            if diff != 0:
                s_misclass_mb = misclass_mb+1            
        ermb = np.round((misclass_mb/len(target)),2)
        s_ermb = np.round((s_misclass_mb/len(target)),2)
        loopmisc.extend([misclass_mb])
        looper.extend([ermb])
        s_loopmisc.extend([s_misclass_mb])
        s_looper.extend([s_ermb])
        
    if ermy == 1:
        tp = list(clakm.values())
        difference = []
        zip_object = zip(tp, ta)
        for i, j in zip_object:
            difference.append(abs(i-j))
        mis1 = np.sum(difference)
        myer1 = np.round((mis1/len(target)*100),2)
        mymis.extend([mis1])
        myerr.extend([myer1])
        tp = list(clmb.values())
        difference = []
        zip_object = zip(tp, ta)
        for i, j in zip_object:
            difference.append(abs(i-j))
        mis1 = np.sum(difference)
        myer1 = np.round((mis1/len(target)*100),2)
        mymis.extend([mis1])
        myerr.extend([myer1])
                         
        
    iters[irun] = barl + bars + barch + barkm
    siscores[irun] = loopss
    dbscores[irun] = loopdb
    counter[irun] = [clm1, sm_clm1 , clm2, sm_clm2,  clch , clakm, clmb]
                         
    if comcal == 1:
        completeness[irun] = loopcom
        s_completeness[irun] = s_loopcom
    if homcal == 1:
        homogen[irun] = loophom
        s_homogen[irun] = s_loophom
    if accal == 1:
        accur[irun] = loopac
        s_accur[irun] = s_loopac
        hamm[irun] = loopham
        eucli[irun] = loopeuc
        manhat[irun] = loopman
        euclid[irun] = looped
        manhatt[irun] = loopmh
    if ercal == 1:
        misclass[irun] =loopmisc
        error[irun] = looper
        s_misclass[irun] = s_loopmisc
        s_error[irun] = s_looper
    if ermy == 1:
        myerror[irun] = myerr
        mymisclass[irun] = mymis
                        
    #Summary of Iterations
    if outfs == 0:
        if version == 0:
            os.chdir(dir_path)
            sys.stdout = fmout
        else:
            os.chdir(dir_path)
            sys.stdout = fout
    else:
        sys.stdout = rstcon
    print('--------------------------------------------')
    print('----------Results of Run_', irun+1,'-------------')
    print('--------------------------------------------')
    print(" ")
    print("circumpts_all",circumpts_all)
    print(" ")
    print("strt_l_m1:",strt_l_m1)
    print(" ")
    print("end_l_m1:",end_l_m1)
    print(" ")
    print("strt_akm:",strt_akm)
    print(" ")
    print("end_akm:",end_akm)
    print(" ")
    print("end_mb:",end_mb)
    print(" ")
    print("target lebel",target)
    print(" ")
    print("labels_m1",labels_m1)
    print(" ")
    print("labels_akm",labels_akm)
    print(" ")
    print("labels_mb",labels_mb)
    print(" ")
    print('                Counter                   ')
    print(counter[irun])
    print(" ")
    print('--------------------------------------------')
    print('--------------------------------------------')
    print(" ")
    print('--------------------------------------------')
    print("               Iteration Nos.               ")
    print('--------------------------------------------')
    print(iters[irun])
    print(" ")
    print('--------------------------------------------')
    print("              Silloute Scores               ")
    print('--------------------------------------------')
    print(siscores[irun])      
    print(" ")
    print('--------------------------------------------')
    print("                 DB Scores                  ")
    print('--------------------------------------------')
    print(dbscores[irun])      
    print(" ")
    print('--------------------------------------------')
    print("               Completeness                 ")
    print('--------------------------------------------')    
    if comcal == 1:
        print(completeness[irun])
        print(s_completeness[irun])
    print(" ")
    print('--------------------------------------------')
    print("                Homogenity                  ")
    print('--------------------------------------------')
    if homcal == 1:
        print(homogen[irun])
        print(s_homogen[irun])
    print(" ")
    print('--------------------------------------------')
    print("                Accuracy                    ")
    print('--------------------------------------------')
    if accal == 1:
        print(accur[irun])
        print(s_accur[irun])
    print(" ")
    print('--------------------------------------------')
    print("              Hamming Distance              ")
    print('--------------------------------------------')
    if accal == 1:
        print(hamm[irun])
    print(" ")
    print('--------------------------------------------')
    print("              Euclidean Distance              ")
    print('--------------------------------------------')
    if accal == 1:
        print(eucli[irun])
        print(euclid[irun])
    print(" ")
    print('--------------------------------------------')
    print("              Manhattan Distance              ")
    print('--------------------------------------------')
    if accal == 1:
        print(manhat[irun])
        print(manhatt[irun])
    print(" ")
    print('--------------------------------------------')
    print("             Misclasses  OLD                ")
    print('--------------------------------------------')
    if ercal == 1:
        print(misclass[irun])
        print(s_misclass[irun])
        print('--------------------------------------------')
        print("             Misclasses MY                  ")
        print('--------------------------------------------')
        print(mymisclass[irun])
    print('--------------------------------------------')
    print("                  Errors Old                ")
    print('--------------------------------------------')
    if ercal == 1:
        print(error[irun])
        print(s_error[irun])
        print('--------------------------------------------')
        print("                  Errors MY                ")
        print('--------------------------------------------')
        print(myerror[irun])
        
    
    if version != 0:
        #Bar Diagram of Iterations 
        os.chdir(dir_path)
        l=['LEC_M1','LEC_M2']
        s=['SEC_M1','SEC_M2']
        ch=['CH']
        km=['AKM']
        fig, ax = plt.subplots()
        ax.bar(l,barl,label="LEC_M1 & LEC_M2")
        ax.bar(s,bars,label="SEC_M1 & SEC_M2", color='g')
        ax.bar(ch,barch,label="CH", color ='k')
        ax.bar(km,barkm,label="AKM", color='r')
        ax.set(xlabel='Various Methods', ylabel='Iteration Numbers', title='Comparison')
        plt.savefig('Bar_Diagram')
    if version != 0:
        #PLOTTING PART
        r5=np.array(rad_lrg)
        sm_r5=np.array(rad_small)
        c5=np.array(lar_cen)
        sm_c5=np.array(sm_cen)
        vorxx=[[] for i in range(0,len(c5))]
        voryy=[[] for i in range(0,len(c5))]
        radxx=[[] for i in range(0,len(r5))]
        vorxx=c5[:,0]
        voryy=c5[:,1]
        svorxx=[[] for i in range(0,len(sm_c5))]
        svoryy=[[] for i in range(0,len(sm_c5))]
        sradxx=[[] for i in range(0,len(sm_r5))]
        svorxx=sm_c5[:,0]
        svoryy=sm_c5[:,1]
        
        if fignature == 1:
            ax1= plt.subplot(3,1,1)
            k5_cen=kmeans1.cluster_centers_
            k5_label=kmeans1.labels_
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k-')    
            #plt.scatter(points[:, 0], points[:, 1],label='True Position',s=50)
            plt.scatter(k5_cen[:, 0], k5_cen[:, 1],marker="x", color='r', s=200, alpha=1);
            plt.scatter(points[:, 0], points[:, 1], c=k5_label, cmap='rainbow');
            for i in range(len(c5)):
                circle5 = plt.Circle((vorxx[i], voryy[i]),  r5[i] , fill= False, color='g')
                ax1.add_artist(circle5)
            voronoi_plot_2d(vor, ax1,show_vertices=False, line_colors='orange',line_width=1, 
                            line_alpha=0.8,show_points=False)
            plt.title("LEC_M1") 
                        
            axkm = plt.subplot(3,1,2)
            k4_cen=kmeans.cluster_centers_
            k4_label=kmeans.labels_
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k-')    
            #plt.scatter(points[:, 0], points[:, 1],label='True Position',s=50)
            plt.scatter(k4_cen[:, 0], k4_cen[:, 1],marker="x", color='r', s=500, alpha=1);
            plt.scatter(points[:, 0], points[:, 1], c=k4_label, cmap='rainbow');
            plt.title("AKM") 
                
            #plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9)
            plt.subplots_adjust(hspace=0.66)
            #plt.tight_layout()
            plt.savefig('Compare')

            axkm = plt.subplot(3,1,3)
            k4_cen=kmeansmb.cluster_centers_
            k4_label=kmeansmb.labels_
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k-')    
            #plt.scatter(points[:, 0], points[:, 1],label='True Position',s=50)
            plt.scatter(k4_cen[:, 0], k4_cen[:, 1],marker="x", color='r', s=500, alpha=1);
            plt.scatter(points[:, 0], points[:, 1], c=k4_label, cmap='rainbow');
            plt.title("MB") 
                
            #plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9)
            plt.subplots_adjust(hspace=0.66)
            #plt.tight_layout()
            plt.savefig('Compare')
           
        else:
            
            fig, ax1 = plt.subplots()
            k5_cen=kmeans1.cluster_centers_
            k5_label=kmeans1.labels_        
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k-')    
            #plt.scatter(points[:, 0], points[:, 1],label='True Position',s=50)
            plt.scatter(k5_cen[:, 0], k5_cen[:, 1],marker="x", color='r', s=200, alpha=1);
            plt.scatter(points[:, 0], points[:, 1], c=k5_label, cmap='rainbow');
            for i in range(len(c5)):
                circle5 = plt.Circle((vorxx[i], voryy[i]),  r5[i] , fill= False, color='g')
                ax1.add_artist(circle5)
            voronoi_plot_2d(vor, ax1,show_vertices=False, line_colors='orange',line_width=1, 
                            line_alpha=0.8,show_points=False)
            plt.title("LEC_M1") 
            plt.savefig("LEC_M1")
            
            fig, sm_ax1 = plt.subplots()
            sm_k5_cen=sm_kmeans1.cluster_centers_
            sm_k5_label=sm_kmeans1.labels_
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k-')    
            plt.scatter(sm_k5_cen[:, 0], sm_k5_cen[:, 1],marker="x", color='r', s=200, alpha=1);
            plt.scatter(points[:, 0], points[:, 1], c=sm_k5_label, cmap='rainbow');
            for i in range(len(sm_c5)):
                sm_circle5 = plt.Circle((svorxx[i], svoryy[i]),  sm_r5[i] , fill= False, color='g',alpha=1)
                sm_ax1.add_artist(sm_circle5)
            voronoi_plot_2d(vor,sm_axshow_vertices=False, line_colors='orange',line_width=1, 
                            line_alpha=0.8,show_points=False)
            plt.title('SEC_M1')    
            plt.savefig("SEC_M1")
            
            fig, ax2 = plt.subplots()
            k5_cen=kmeans2.cluster_centers_
            k5_label=kmeans2.labels_
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k-')    
            #plt.scatter(points[:, 0], points[:, 1],label='True Position',s=50)
            plt.scatter(k5_cen[:, 0], k5_cen[:, 1],marker="x", color='r', s=200, alpha=1);
            plt.scatter(points[:, 0], points[:, 1], c=k5_label, cmap='rainbow');
            for i in range(len(c5)):
                circle5 = plt.Circle((vorxx[i], voryy[i]),  r5[i] , fill= False, color='g')
                ax2.add_artist(circle5)
            voronoi_plot_2d(vor, ax2,show_vertices=False, line_colors='orange',line_width=1, 
                            line_alpha=0.8,show_points=False)
            plt.title("LEC_M2") 
            plt.savefig("LEC_M2")        
           
            fig, sm_ax2 = plt.subplots()
            sm_k5_cen=sm_kmeans2.cluster_centers_
            sm_k5_label=sm_kmeans2.labels_
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k-')    
            plt.scatter(sm_k5_cen[:, 0], sm_k5_cen[:, 1],marker="x", color='r', s=200, alpha=1);
            plt.scatter(points[:, 0], points[:, 1], c=sm_k5_label, cmap='rainbow');
            for i in range(len(sm_c5)):
                sm_circle5 = plt.Circle((svorxx[i], svoryy[i]),  sm_r5[i] , fill= False, color='g',alpha=1)
                sm_ax2.add_artist(sm_circle5)
            voronoi_plot_2d(vor,sm_ax2,show_vertices=False, line_colors='orange',line_width=1, 
                            line_alpha=0.8,show_points=False)
            plt.title('SEC_M2')    
            plt.savefig("SEC_M2")
                    
            fig, axch = plt.subplots()
            k2_cen=kmeansch.cluster_centers_
            k2_label=kmeansch.labels_
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k-')    
            #plt.scatter(points[:, 0], points[:, 1],label='True Position',s=50)
            plt.scatter(k2_cen[:, 0], k2_cen[:, 1],marker="x", color='r', s=200, alpha=1);
            plt.scatter(points[:, 0], points[:, 1], c=k2_label, cmap='rainbow');
            for i in range(len(c5)):
                circle2 = plt.Circle((vorxx[i], voryy[i]),  r5[i] , fill= False, color='g')
                axch.add_artist(circle2)
            voronoi_plot_2d(vor, axch,show_vertices=False, line_colors='orange',line_width=1, 
                            line_alpha=0.8,show_points=False)
            plt.title("CH") 
            plt.savefig('CH')
                   
            fig, axkm = plt.subplots()
            k4_cen=kmeans.cluster_centers_
            k4_label=kmeans.labels_ 
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k-')    
            #plt.scatter(points[:, 0], points[:, 1],label='True Position',s=50)
            plt.scatter(k4_cen[:, 0], k4_cen[:, 1],marker="x", color='r', s=500, alpha=1);
            plt.scatter(points[:, 0], points[:, 1], c=k4_label, cmap='rainbow');
            plt.title("AKM") 
            plt.savefig('AKM')
        
    endirun=time.time()
    timeirun = np.round((endirun-startirun),2)
    if outfs == 0:
        if version == 0:
            os.chdir(dir_path)
            sys.stdout = fmout
            print(" ")
            print("Total execution time for Run_",irun+1,"is =",timeirun,"Second/s")
            print('-----------------------------------------------------------------')
            fmout.close()
        else:
            os.chdir(dir_path)
            sys.stdout = fout
            print(" ")
            print("Total execution time for Run_",irun+1,"is =",timeirun,"Second/s")
            print('-----------------------------------------------------------------')
            fout.close()
    else:
        sys.stdout = rstcon
        print("")
        print("Total execution time for Run_",irun+1,"is =",timeirun,"Second/s")
        print('-----------------------------------------------------------------')
        print('-----------------------------------------------------------------')
        print("")
        
#Result Analysis
a1 = np.round(np.mean([nn[0] for nn in iters]),2)
a2 = np.round(np.mean([nn[1] for nn in iters]),2)
a3 = np.round(np.mean([nn[2] for nn in iters]),2)
a4 = np.round(np.mean([nn[3] for nn in iters]),2)
a5 = np.round(np.mean([nn[4] for nn in iters]),2)
a6 = np.round(np.mean([nn[5] for nn in iters]),2)
a7 = np.round(np.mean([nn[6] for nn in iters]),2)
aar = [a1,a2,a3,a4,a5,a6,a7]
minas = np.min([a1,a2,a3,a4,a5,a6,a7])
maxas = np.max([a1,a2,a3,a4,a5,a6,a7])
r1 = np.round(a1/a6,2)
r2 = np.round(a2/a6,2)
r3 = np.round(a3/a6,2)
r4 = np.round(a4/a6,2)
r5 = np.round(a5/a6,2)
r6 = np.round(a6/a6,2)
r7 = np.round(a7/a6,2)
p1 = round(((a6-a1)/a1)*100)
p2 = round(((a6-a2)/a2)*100)
p3 = round(((a6-a3)/a3)*100)
p4 = round(((a6-a4)/a4)*100)
p5 = round(((a6-a5)/a5)*100)
p6 = round(((a6-a6)/a6)*100)
p7 = round(((a6-a7)/a7)*100)
ss1 = np.round(np.mean([nn[0] for nn in siscores]),2)
ss2 = np.round(np.mean([nn[1] for nn in siscores]),2)
ss3 = np.round(np.mean([nn[2] for nn in siscores]),2)
ss4 = np.round(np.mean([nn[3] for nn in siscores]),2)
ss5 = np.round(np.mean([nn[4] for nn in siscores]),2)
ss6 = np.round(np.mean([nn[5] for nn in siscores]),2)
ss7 = np.round(np.mean([nn[6] for nn in siscores]),2)
maxss = np.max([ss1,ss1,ss3,ss4,ss5,ss6,ss7])
db1 = np.round(np.mean([nn[0] for nn in dbscores]),2)
db2 = np.round(np.mean([nn[1] for nn in dbscores]),2)
db3 = np.round(np.mean([nn[2] for nn in dbscores]),2)
db4 = np.round(np.mean([nn[3] for nn in dbscores]),2)
db5 = np.round(np.mean([nn[4] for nn in dbscores]),2)
db6 = np.round(np.mean([nn[5] for nn in dbscores]),2)
db7 = np.round(np.mean([nn[6] for nn in dbscores]),2)
mindb = np.min([db1,db2,db3,db4,db5,db6,db7])
if comcal ==1:
    com1 = np.round(np.mean([nn[0] for nn in completeness]),2)
    com2 = np.round(np.mean([nn[1] for nn in completeness]),2)
    com3 = np.round(np.mean([nn[2] for nn in completeness]),2)
    com4 = np.round(np.mean([nn[3] for nn in completeness]),2)
    com5 = np.round(np.mean([nn[4] for nn in completeness]),2)
    com6 = np.round(np.mean([nn[5] for nn in completeness]),2)
    com7 = np.round(np.mean([nn[6] for nn in completeness]),2)
    s_com1 = np.round(np.mean([nn[0] for nn in s_completeness]),2)
    s_com2 = np.round(np.mean([nn[1] for nn in s_completeness]),2)
    s_com3 = np.round(np.mean([nn[2] for nn in s_completeness]),2)
    s_com4 = np.round(np.mean([nn[3] for nn in s_completeness]),2)
    s_com5 = np.round(np.mean([nn[4] for nn in s_completeness]),2)
    s_com6 = np.round(np.mean([nn[5] for nn in s_completeness]),2)
    s_com7 = np.round(np.mean([nn[6] for nn in s_completeness]),2)
    maxcom = np.max([com1,com2,com3,com4,com5,com6,com7])
    s_maxcom = np.max([s_com1,s_com2,s_com3,s_com4,s_com5,s_com6,s_com7])
if homcal ==1:
    hom1 = np.round(np.mean([nn[0] for nn in homogen]),2)
    hom2 = np.round(np.mean([nn[1] for nn in homogen]),2)
    hom3 = np.round(np.mean([nn[2] for nn in homogen]),2)
    hom4 = np.round(np.mean([nn[3] for nn in homogen]),2)
    hom5 = np.round(np.mean([nn[4] for nn in homogen]),2)
    hom6 = np.round(np.mean([nn[5] for nn in homogen]),2)
    hom7 = np.round(np.mean([nn[6] for nn in homogen]),2)
    s_hom1 = np.round(np.mean([nn[0] for nn in s_homogen]),2)
    s_hom2 = np.round(np.mean([nn[1] for nn in s_homogen]),2)
    s_hom3 = np.round(np.mean([nn[2] for nn in s_homogen]),2)
    s_hom4 = np.round(np.mean([nn[3] for nn in s_homogen]),2)
    s_hom5 = np.round(np.mean([nn[4] for nn in s_homogen]),2)
    s_hom6 = np.round(np.mean([nn[5] for nn in s_homogen]),2)
    s_hom7 = np.round(np.mean([nn[6] for nn in s_homogen]),2)
    maxhom = np.max([hom1,hom2,hom3,hom4,hom5,hom6,hom7])
    s_maxhom = np.max([s_hom1,s_hom2,s_hom3,s_hom4,s_hom5,s_hom6,s_hom7])
if accal ==1:
    ac1 = np.round(np.mean([nn[0] for nn in accur]),2)
    ac2 = np.round(np.mean([nn[1] for nn in accur]),2)
    ac3 = np.round(np.mean([nn[2] for nn in accur]),2)
    ac4 = np.round(np.mean([nn[3] for nn in accur]),2)
    ac5 = np.round(np.mean([nn[4] for nn in accur]),2)
    ac6 = np.round(np.mean([nn[5] for nn in accur]),2)
    ac7 = np.round(np.mean([nn[6] for nn in accur]),2)
    s_ac1 = np.round(np.mean([nn[0] for nn in s_accur]),2)
    s_ac2 = np.round(np.mean([nn[1] for nn in s_accur]),2)
    s_ac3 = np.round(np.mean([nn[2] for nn in s_accur]),2)
    s_ac4 = np.round(np.mean([nn[3] for nn in s_accur]),2)
    s_ac5 = np.round(np.mean([nn[4] for nn in s_accur]),2)
    s_ac6 = np.round(np.mean([nn[5] for nn in s_accur]),2)
    s_ac7 = np.round(np.mean([nn[6] for nn in s_accur]),2)
    maxac = np.max([ac1,ac2,ac3,ac4,ac5,ac6,ac7])
    s_maxac = np.max([s_ac1,s_ac2,s_ac3,s_ac4,s_ac5,s_ac6,s_ac7])
    ham1 = np.round(np.mean([nn[0] for nn in hamm]),2)
    ham2 = np.round(np.mean([nn[1] for nn in hamm]),2)
    ham3 = np.round(np.mean([nn[2] for nn in hamm]),2)
    ham4 = np.round(np.mean([nn[3] for nn in hamm]),2)
    ham5 = np.round(np.mean([nn[4] for nn in hamm]),2)
    ham6 = np.round(np.mean([nn[5] for nn in hamm]),2)
    ham7 = np.round(np.mean([nn[6] for nn in hamm]),2)
    euc1 = np.round(np.mean([nn[0] for nn in eucli]),2)
    euc2 = np.round(np.mean([nn[1] for nn in eucli]),2)
    euc3 = np.round(np.mean([nn[2] for nn in eucli]),2)
    euc4 = np.round(np.mean([nn[3] for nn in eucli]),2)
    euc5 = np.round(np.mean([nn[4] for nn in eucli]),2)
    euc6 = np.round(np.mean([nn[5] for nn in eucli]),2)
    euc7 = np.round(np.mean([nn[6] for nn in eucli]),2)
    eucd1 = np.round(np.mean([nn[0] for nn in euclid]),2)
    eucd2 = np.round(np.mean([nn[1] for nn in euclid]),2)
    eucd3 = np.round(np.mean([nn[2] for nn in euclid]),2)
    eucd4 = np.round(np.mean([nn[3] for nn in euclid]),2)
    eucd5 = np.round(np.mean([nn[4] for nn in euclid]),2)
    eucd6 = np.round(np.mean([nn[5] for nn in euclid]),2)
    eucd7 = np.round(np.mean([nn[6] for nn in euclid]),2)
    man1 = np.round(np.mean([nn[0] for nn in manhat]),2)
    man2 = np.round(np.mean([nn[1] for nn in manhat]),2)
    man3 = np.round(np.mean([nn[2] for nn in manhat]),2)
    man4 = np.round(np.mean([nn[3] for nn in manhat]),2)
    man5 = np.round(np.mean([nn[4] for nn in manhat]),2)
    man6 = np.round(np.mean([nn[5] for nn in manhat]),2)
    man7 = np.round(np.mean([nn[6] for nn in manhat]),2)
    manh1 = np.round(np.mean([nn[0] for nn in manhatt]),2)
    manh2 = np.round(np.mean([nn[1] for nn in manhatt]),2)
    manh3 = np.round(np.mean([nn[2] for nn in manhatt]),2)
    manh4 = np.round(np.mean([nn[3] for nn in manhatt]),2)
    manh5 = np.round(np.mean([nn[4] for nn in manhatt]),2)
    manh6 = np.round(np.mean([nn[5] for nn in manhatt]),2)
    manh7 = np.round(np.mean([nn[6] for nn in manhatt]),2)
    minham = np.min([ham1,ham2,ham3,ham4,ham5,ham6,ham7])
    mineuc = np.min([euc1,euc2,euc3,euc4,euc5,euc6,euc7])
    minman = np.min([man1,man2,man3,man4,man5,man6,man7])
    mineucd = np.min([eucd1,eucd2,eucd3,eucd4,eucd5,eucd6,eucd7])
    minmann = np.min([manh1,manh2,manh3,manh4,manh5,manh6,manh7])

if ercal ==1:
    mc1 = np.round(np.mean([nn[0] for nn in misclass]),2)
    mc2 = np.round(np.mean([nn[1] for nn in misclass]),2)
    mc3 = np.round(np.mean([nn[2] for nn in misclass]),2)
    mc4 = np.round(np.mean([nn[3] for nn in misclass]),2)
    mc5 = np.round(np.mean([nn[4] for nn in misclass]),2)
    mc6 = np.round(np.mean([nn[5] for nn in misclass]),2)
    mc7 = np.round(np.mean([nn[6] for nn in misclass]),2)
    minmc = np.min([mc1,mc2,mc3,mc4,mc5,mc6,mc7])
    er1 = np.round(np.mean([nn[0] for nn in error]),2)
    er2 = np.round(np.mean([nn[1] for nn in error]),2)
    er3 = np.round(np.mean([nn[2] for nn in error]),2)
    er4 = np.round(np.mean([nn[3] for nn in error]),2)
    er5 = np.round(np.mean([nn[4] for nn in error]),2)
    er6 = np.round(np.mean([nn[5] for nn in error]),2)
    er7 = np.round(np.mean([nn[6] for nn in error]),2)
    miner = np.min([er1,er2,er3,er4,er5,er6,er7])
    s_mc1 = np.round(np.mean([nn[0] for nn in s_misclass]),2)
    s_mc2 = np.round(np.mean([nn[1] for nn in s_misclass]),2)
    s_mc3 = np.round(np.mean([nn[2] for nn in s_misclass]),2)
    s_mc4 = np.round(np.mean([nn[3] for nn in s_misclass]),2)
    s_mc5 = np.round(np.mean([nn[4] for nn in s_misclass]),2)
    s_mc6 = np.round(np.mean([nn[5] for nn in s_misclass]),2)
    s_mc7 = np.round(np.mean([nn[6] for nn in s_misclass]),2)
    s_minmc = np.min([s_mc1,s_mc2,s_mc3,s_mc4,s_mc5,s_mc6,s_mc7])
    s_er1 = np.round(np.mean([nn[0] for nn in s_error]),2)
    s_er2 = np.round(np.mean([nn[1] for nn in s_error]),2)
    s_er3 = np.round(np.mean([nn[2] for nn in s_error]),2)
    s_er4 = np.round(np.mean([nn[3] for nn in s_error]),2)
    s_er5 = np.round(np.mean([nn[4] for nn in s_error]),2)
    s_er6 = np.round(np.mean([nn[5] for nn in s_error]),2)
    s_er7 = np.round(np.mean([nn[6] for nn in s_error]),2)
    s_miner = np.min([s_er1,s_er2,s_er3,s_er4,s_er5,s_er6,s_er7])

if ermy == 1 :
    mism1 = np.round(np.mean([nn[0] for nn in mymisclass]),2)
    mism2 = np.round(np.mean([nn[1] for nn in mymisclass]),2)
    mism3 = np.round(np.mean([nn[2] for nn in mymisclass]),2)
    mism4 = np.round(np.mean([nn[3] for nn in mymisclass]),2)
    mism5 = np.round(np.mean([nn[4] for nn in mymisclass]),2)
    mism6 = np.round(np.mean([nn[5] for nn in mymisclass]),2)
    mism7 = np.round(np.mean([nn[6] for nn in mymisclass]),2)
    minmism = np.min([mism1,mism2,mism3,mism4,mism5,mism6,mism7])
    erm1 = np.round(np.mean([nn[0] for nn in myerror]),2)
    erm2 = np.round(np.mean([nn[1] for nn in myerror]),2)
    erm3 = np.round(np.mean([nn[2] for nn in myerror]),2)
    erm4 = np.round(np.mean([nn[3] for nn in myerror]),2)
    erm5 = np.round(np.mean([nn[4] for nn in myerror]),2)
    erm6 = np.round(np.mean([nn[5] for nn in myerror]),2)
    erm7 = np.round(np.mean([nn[6] for nn in myerror]),2)
    minerm = np.min([erm1,erm2,erm3,erm4,erm5,erm6,erm7])  

os.chdir(base_path)
l=['LEC_M1','LEC_M2']
s=['SEC_M1','SEC_M2']
ch=['CH']
km=['AKM']
fig, ax = plt.subplots()
ax.bar(l,[a1,a3],label="LEC_M1 & LEC_M2")
ax.bar(s,[a2,a4],label="SEC_M1 & SEC_M2", color='g')
ax.bar(ch,[a5],label="CH", color ='k')
ax.bar(km,[a6],label="AKM", color='r')
ax.set(xlabel='Various Methods', ylabel='Iteration Numbers', title='Comparison of Iteration Numbers')
plt.savefig('Iterations')
fig, ax = plt.subplots()
ax.bar(l,[ss1,ss3],label="LEC_M1 & LEC_M2")
ax.bar(s,[ss2,ss4],label="SEC_M1 & SEC_M2", color='g')
ax.bar(ch,[ss5],label="CH", color ='k')
ax.bar(km,[ss6],label="AKM", color='r')
#plt.ylim(0.43,0.45)
ax.set(xlabel='Various Methods', ylabel='Silhouette Scores', title='Comparison of Silhouette Scores')
plt.savefig('Silhouette_Scores')
fig, ax = plt.subplots()
ax.bar(l,[db1,db3],label="LEC_M1 & LEC_M2")
ax.bar(s,[db2,db4],label="SEC_M1 & SEC_M2", color='g')
ax.bar(ch,[db5],label="CH", color ='k')
ax.bar(km,[db6],label="AKM", color='r')
#plt.ylim(0.78,0.80)
ax.set(xlabel='Various Methods', ylabel='DB Scores', title='Comparison of DB Scores')
plt.savefig('DB_Scores')
end=time.time()
totaltime = np.round((end-start), 2)

# Writing Outpur Summary in a File
if outfs == 0:
    os.chdir(base_path)
    sys.stdout = fsumm
    print("")
    print("--------------------Results SUMMARY--------------------------------")
    print("")
    print("Total numbers of run you have selected: ", runcount)
    print("")
    print('--------------------------------------------------------------')
    print("                     Average Parameters                            ")
    print('--------------------------------------------------------------')
    print("Parameters  LEC_M1  LEC_M2  SEC_M1  SEC_M2  CH  AKM  MB")
    print("")
    print('-----------------------------------------------------------------------------')
    print("Iterations  : ",a1,"    ",a2,"    ",a3,"    ",a4,"    ",a5,"    ",a6,"    ",a7)
    print("")
    print("Ratios      : ",r1,"    ",r2,"     ",r3,"     ",r4,"    ",r5,"    ",r6,"    ",r7)
    print("") 
    print("Percen      : ",p1,"     ",p2,"    ",p3,"    ",p4,"    ",p5,"    ",p6,"    ",p7)
    print(" ")
    print("Sh Scores   : ",ss1,"    ",ss2,"    ",ss3,"    ",ss4,"    ",ss5,"    ",ss6,"    ",ss7)
    print(" ") 
    print("DB Scores   : ",db1,"    ",db2,"    ",db3,"    ",db4,"    ",db5,"    ",db6,"    ",db7)
    if comcal ==1:
        print("") 
        print("Compltns    : ",com1,"    ",com2,"    ",com3,"    ",com4,"    ",com5,"    ",com6,"    ",com7)
        print("Compltns(S) : ",s_com1,"    ",s_com2,"    ",s_com3,"    ",s_com4,"    ",s_com5,"    ",s_com6,"    ",s_com7)
    if homcal ==1:
        print("")
        print("Homogty     : ",hom1,"    ",hom2,"    ",hom3,"    ",hom4,"    ",hom5,"    ",hom6,"    ",hom7)
        print("Homogty(S)  : ",s_hom1,"    ",s_hom2,"    ",s_hom3,"    ",s_hom4,"    ",s_hom5,"    ",s_hom6,"    ",s_hom7)
    if accal ==1:
        print("")
        print("Accuracy    : ",ac1,"    ",ac2,"    ",ac3,"    ",ac4,"    ",ac5,"    ",ac6,"    ",ac7)
        print("Accuracy(S) : ",s_ac1,"    ",s_ac2,"    ",s_ac3,"    ",s_ac4,"    ",s_ac5,"    ",s_ac6,"    ",s_ac7)
        print(" ")
        print("Hamming Dst : ",ham1,"    ",ham2,"    ",ham3,"    ",ham4,"    ",ham5,"    ",ham6,"    ",ham7)
        print(" ")
        print("Eucli Dst   : ",euc1,"    ",euc2,"    ",euc3,"    ",euc4,"    ",euc5,"    ",euc6,"    ",euc7)
        print("Euclid Dst  : ",eucd1,"    ",eucd2,"    ",eucd3,"    ",eucd4,"    ",eucd5,"    ",eucd6,"    ",eucd7)
        print(" ")
        print("Manhat Dst  : ",man1,"    ",man2,"    ",man3,"    ",man4,"    ",man5,"    ",man6,"    ",man7)
        print("Manhatt Dst : ",manh1,"    ",manh2,"    ",manh3,"    ",manh4,"    ",manh5,"    ",manh6,"    ",manh7)
    if ercal ==1:
        print("")
        print("Misclass    : ",mc1,"    ",mc2,"    ",mc3,"    ",mc4,"    ",mc5,"    ",mc6,"    ",mc7)
        print("Misclass(S) : ",s_mc1,"    ",s_mc2,"    ",s_mc3,"    ",s_mc4,"    ",s_mc5,"    ",s_mc6,"    ",s_mc7)
        print("")
        print("Error Old   : ",er1,"    ",er2,"    ",er3,"    ",er4,"    ",er5,"    ",er6,"    ",er7)
        print("ErrorOld(S) : ",s_er1,"    ",s_er2,"    ",s_er3,"    ",s_er4,"    ",s_er5,"    ",s_er6,"    ",s_er7)
    
    if ermy == 1:
        print("")
        print("My Misclass : ",mism1,"    ",mism2,"    ",mism3,"    ",mism4,"    ",mism5,"    ",mism6,"    ",mism7)
        print("My Error    : ",erm1,"    ",erm2,"    ",erm3,"    ",erm4,"    ",erm5,"    ",erm6,"    ",erm7)
        
    print('-----------------------------------------------------------------------------')
    print("")
    print("The minimum iteration number = ", minas)
    print("The maximum Silhouette score = ", maxss)
    print("The minimum DB score = ", mindb)
    print("The maximum completeness = ", maxcom)
    print("The maximum completeness(S) = ", s_maxcom)
    print("The maximum homogeneity = ", maxhom)
    print("The maximum homogeneity(S) = ", s_maxhom)
    print("The maximum accuracy = ", maxac)
    print("The maximum accuracy(S) = ", s_maxac)
    print("The minimum Hamming = ", minham)
    print("The minimum Euclidean = ", mineuc)
    print("The minimum Manhattan = ", minman)
    print("The minimum Euclidean = ", mineucd)
    print("The minimum Manhattan = ", minmann)
    print("The minimum misclass  = ", minmc)
    print("The minimum misclass(S)  = ", s_minmc)
    print("The minimum MY misclass  = ", minmism)
    print("The minimum errors  =  ", miner)
    print("The minimum errors(S)  =  ", s_miner)
    print("The minimum MY errors  =  ", minerm)
    print(" ")
    print(" ")
    print("Congratulations! Your program has been executed succesfully")
    print(" ")
    print("Total execution time =",totaltime,"Second/s")
    print(" ")
    print('-----------------------------------------------------------------')
    print("                  Thank You for Running My CODE                  ")
    print("                 @Kinsuk : kinsuk@nitttrkol.ac.in                ")
    print('-----------------------------------------------------------------')
    fsumm.close()
    sys.stdout.close() 
    
# End Terminal Messages with short summary 
sys.stdout = rstcon
print("--------------------Results SUMMARY---------------------------")
print(" ")
print("--------------Average Iterations--------------")
print('----------------------------------------------')
print("LEC_M1  LEC_M2   SEC_M1  SEC_M2     CH    AKM  MB")
print('----------------------------------------------')
print("",a1,"   ",a2,"    ",a3,"    ",a4,"  ",a5,"  ",a6,"  ",a7)
print("",r1,"  ",r2,"   ",r3,"  ",r4,"  ",r5,"  ",r6,"    ",r7)
print("",p1,"  ",p2,"    ",p3,"  ",p4,"  ",p5,"  ",p6,"    ",p7)
print('----------------------------------------------')
print(" ")
print(" ")
print("Congratulations! Your program has been executed succesfully")
print(" ")
print("Total execution time =",totaltime,"Second/s")
print(" ")
print('-----------------------------------------------------------------')
print("                  Thank You for Running My CODE                  ")
print("                 @Kinsuk : kinsuk@nitttrkol.ac.in                ")
print('-----------------------------------------------------------------')

