"""=====================================================

                        import

====================================================="""
import pandas as pd
from sklearn.cluster import KMeans
import scipy
import numpy as np
from numpy.random import *

"""=====================================================

                        X means

====================================================="""
class Xmeans:
    
    def __init__(self,table,k_zero=2):
        self.table=table
        #初期クラスター数
        self.k_zero=k_zero
        #クラスターを置くデッキ
        self.clusters=[]
        #どのクラスターに属するかの帰属
        self.attribute=[]
        
    #ハンドラーとなる関数
    def xmeans(self,cluster_id,data_id):
        #初期idが受け渡される
        self.cluster_id=cluster_id
        cluster=Cluster(self,data_id)
        #クラスターを二つに分割
        split1,split2=cluster.split()
        c1,c2=Cluster(self,split1),Cluster(self,split2)
        #基準化定数を算出
        norm_const=self.get_norm_const(c1,c2)
        #BICを算出し、分割前後で比較
        if cluster.get_BIC()<self.get_BIC_(c1,c2,norm_const):
            #アトリビュートに結果を反映し、終了
            self.attribute[data_id]=cluster_id
        else:
            #デッキの最後尾に二番目の分割クラスターを追加
            self.clusters.append(c2.data_id)
            #再帰的に、次の分割へ
            self.xmeans(cluster_id,c1.data_id)

    #あらかじめ指定された数に分割
    def first_split(self):
        #分割後、アトリビュートに反映
        array=KMeans(n_clusters=self.k_zero,random_state=None).fit_predict(self.table)
        self.attribute=array
        #各クラスターの初期idをデッキに渡す
        for num in range(self.k_zero):
            self.clusters.append(np.where(array==num)[0])
            
    #初期idからデータを取得
    def get_data(self,data_id):
        return self.table[data_id]
    
    #基準化定数を取得
    def get_norm_const(self,c1,c2):
        separation=np.linalg.norm(c1.Lmean-c2.Lmean)/np.sqrt(np.linalg.det(c1.Lvariance)+np.linalg.det(c2.Lvariance))
        norm_const=1/(2*scipy.stats.norm.cdf(separation))
        return norm_const
    
    #BICを取得
    def get_BIC_(self,c1,c2,norm_const):
        length=len(c1.data_id)+len(c2.data_id)
        dimension=self.table.shape[1]
        BIC_=-2*(length*np.log(norm_const)+c1.get_Lprob()+c2.get_Lprob())+dimension*(dimension+3)*np.log(length)
        return BIC_

class Cluster:
    
    def __init__(self,x,data_id):
        #データは初期idのみを受け渡す
        self.data_id=data_id
        #初期idからデータ取得
        self.data=x.get_data(data_id)
        #平均値
        self.Lmean=np.mean(self.data, axis=0)
        #分散共分散行列
        self.Lvariance=np.cov(self.data,rowvar=False)
        
    #対数尤度を算出する
    def get_Lprob(self):
        mean=self.Lmean
        cov=self.Lvariance
        log_Lprob=[]
        for num in range(len(self.data)):
            log_Lprob.append(scipy.stats.multivariate_normal.logpdf(self.data[num],mean,cov))
        return np.sum(log_Lprob)
    
    #BICを算出する
    def get_BIC(self):
        n,dimension=self.data.shape
        BIC=-2*self.get_Lprob()+dimension*(dimension+3)*np.log(n)/2
        return BIC
    
    #クラスターを2meansで分割する
    def split(self):
        array=KMeans(n_clusters=2,random_state=1).fit_predict(self.data)
        return self.data_id[np.where(array==0)[0]],self.data_id[np.where(array==1)[0]]

"""=====================================================

                        main

====================================================="""    
def main(table):
    #tableはnumpy配列
    x=Xmeans(table)
    x.first_split()

    for cluster_id,data_id in enumerate(x.clusters):
        x.xmeans(cluster_id,data_id)
    
    return x.attribute

