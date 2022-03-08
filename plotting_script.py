'''
This file plots from the pickle files generated by "compute_marginal.py".
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc, font_manager
import matplotlib as mpl
sizeOfFont_X = 30
sizeOfFont_Y = 25
plt.style.use('seaborn-paper')
mpl.rcParams['axes.linewidth'] = 5 #set the value globally

if __name__ == "__main__":
    ### Set kord = "k" to plot effect of varying k
    ### Set kord = "d" to plot effect of varying d.
    kord = "d"
    if kord == "d":
        marg_df = pd.read_pickle("data/all_mechanisms_vary_d.pkl")
    else:
        d = 9
        marg_df = pd.read_pickle("data/all_mechanisms_vary_"+str(d)+".pkl")
        #marg_df.rename(columns={"ihash_l1_std":"iolh_l1_std"},inplace=True)
        #marg_df.to_pickle("all_mechanisms_vary_"+str(d)+".pkl")

    print (marg_df.columns)
    marg_df = marg_df[marg_df.d >=6]

    marg_df= marg_df.sort_values(by=["k"],ascending=True)
    marg_df= marg_df.sort_values(by=["d"],ascending=True)

    marg_df["k"] =marg_df["k"].astype("int").astype("str")
    marg_df["d"] =marg_df["d"].astype("int").astype("str")

    if kord == "k":
        k_list = marg_df.k.tolist()
    else:
        k_list = marg_df.d.tolist()

    #print(k_list)

    #print (marg_df)
    color_bars = ["magenta","blue","green","red","turquoise","black","indigo","brown"]
    cols =[u'irr_l1_std', u'mrr_l1_std', u'mht_l1_std', u'mps_l1_std',u'iht_l1_std',u'iolh_l1_std',u"icmsht_l1_std"]
    colspaper =[u'InpRR', u'MargRR',  u'MargHT', u'MargPS',u'InpHT',u'InpOLH',"InpCMSHT"]
    colsmeandf =[u'irr_l1', u'mrr_l1',  u'mht_l1', u'mps_l1',u'iht_l1','iolh_l1',u"icmsht_l1"]

    population = int(marg_df["population"].iloc[0])

    ticks_font_X = font_manager.FontProperties(style='normal',  size=sizeOfFont_X, weight='bold', stretch='normal')
    ticks_font_Y = font_manager.FontProperties(style='normal',  size=sizeOfFont_Y, weight='bold', stretch='normal')

    df_std = marg_df[cols].rename(columns=dict(zip(cols,colspaper)))
    df_mean= marg_df[colsmeandf].rename(columns=dict(zip(colsmeandf,colspaper)))

    df_std.index= k_list
    df_mean.index = k_list
    ### If some counts are too low to be recovered by the corrections, we normalize them to 2.0, since the maxium L1 between two vectors is <=2.0.
    df_mean[df_mean > 2.0] = 2.0

    ## Convert L1 to total variational distance.
    df_mean/=2.0
    df_std/=2.0

    fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(15,12))

    df_mean.plot.bar(ax=axes,color=color_bars,yerr=df_std,rot="20",width=.8,capsize=10,ylim=(0,0.5))

    #df_mean.plot(ax=axes,color=color_bars,yerr=df_std,rot="20",ylim=(0,0.4),linewidth=7)


    axes.legend(loc="best",prop={'size':30,"weight":"bold"})

    plt.ylabel("Total Variation Distance" , fontsize=32,fontweight='bold')
    if kord == "d":
        plt.xlabel("d",fontsize=30,fontweight='bold')
        axes.set_title("N="+str(population)+ ", " + r" $e^{\epsilon}$="+str(3)+ ", k=3",fontsize=35,fontweight='bold')

    else:
        plt.xlabel("k",fontsize=40,fontweight='bold')
        axes.set_title("N="+str(population)+ ", " + r" $e^{\epsilon}$="+str(3)+ ", d="+str(d),fontsize=42,fontweight='bold')

    for label in axes.get_xticklabels():
        label.set_fontproperties(ticks_font_X)
    for label in axes.get_yticklabels():
        label.set_fontproperties(ticks_font_Y)

    plt.rcParams['errorbar.capsize']=24
    if kord == "d":
        plt.savefig("plots/vary_d")
    else:
        plt.savefig("plots/vary_k")
    plt.show()
