
import numpy as np
import warnings


class BestInterval:

    def __init__(self, depth=5, beam_size=1, add_iter=0):
        self.beam_size = beam_size
        self.depth = depth
        self.add_iter = add_iter
        
    def find(self, dx, dy):
        if np.logical_or(dy.min() < 0, dy.max() > 1):
            warnings.warn("The target variable takes values from outside [0,1]")
        dim = dx.shape[1]
        if self.depth > dim:
            warnings.warn("Restricting depth parameter to the number of atributes in data")
        depth = min(self.depth, dim)
        
        box_init = self._get_initial_restrictions(dx)
        res_box = []
        res_tab = np.empty([0,3])
        
        for i in range(0, dim):
            tmp = self._refine(dx, dy, box_init, i, 0)
            res_box.append(tmp[0])
            res_tab = np.concatenate((res_tab, np.array([[tmp[1], tmp[2], i]])), axis = 0)
        
        if depth > 1:
            for _ in range(0, depth - 1):
                if res_tab.shape[0] > self.beam_size:
                    retain = res_tab[:,0] >= np.sort(res_tab[:,0])[::-1][self.beam_size - 1]
                    res_tab = res_tab[retain]
                    res_box = [res_box[i] for i in np.where(retain)[0]]
                for k in range(0, len(res_tab)):
                    if res_tab[k, 1] == 1:
                        res_tab[k, 1] = 0
                        inds_r = np.arange(0, dim, 1)[np.arange(0, dim, 1) != res_tab[k, 2]]
                        for i in inds_r:
                            tmp = self._refine(dx, dy, res_box[k], i, res_tab[k, 0])
                            res_box.append(tmp[0])
                            res_tab = np.concatenate((res_tab, np.array([[tmp[1], tmp[2], i]])), axis = 0)
                             
            # additional iterations (refining dimensions which have been formerly refined)
            if res_tab.shape[0] > self.beam_size:
                retain = res_tab[:,0] >= np.sort(res_tab[:,0])[::-1][self.beam_size - 1]
                res_tab = res_tab[retain]
                res_box = [res_box[i] for i in np.where(retain)[0]]
                
            while res_tab[:,1].sum() != 0 and self.add_iter > 0:
                self.add_iter = self.add_iter - 1
                for k in range(0, len(res_tab)):
                    if res_tab[k, 1] == 1:
                        res_tab[k, 1] = 0
                        inds_r = np.where((box_init - res_box[k]).sum(axis = 0) != 0)[0]
                        inds_r = inds_r[inds_r != res_tab[k, 2]]
                        for i in inds_r:
                            tmp = self._refine(dx, dy, res_box[k], i, res_tab[k, 0])
                            res_box.append(tmp[0])
                            res_tab = np.concatenate((res_tab, np.array([[tmp[1], tmp[2], i]])), axis = 0)
                if res_tab.shape[0] > self.beam_size:
                    retain = res_tab[:,0] >= np.sort(res_tab[:,0])[::-1][self.beam_size - 1]
                    res_tab = res_tab[retain]
                    res_box = [res_box[i] for i in np.where(retain)[0]]
        
        winner = np.where(res_tab[:,0] == max(res_tab[:,0]))[0][0]
        return res_box[winner]

    def _refine(self, dx, dy, box, ind, start_q):
        # below numbers correspond to the row numbers in the pseudo-code description
        # from "Efficient algorithms for finding richer subgroup descriptions in 
        # numeric and nominal data" (Algorithm 3)
        N = len(dy)
        Np = sum(dy)
        
        ind_in_box = np.ones(N, dtype = bool)
        for i in range(0, dx.shape[1]):
            if not i == ind:
                ind_in_box = np.logical_and(ind_in_box, np.logical_and(dx[:,i] >= box[0,i], dx[:,i] <= box[1,i]))
        in_box = np.vstack((dx[ind_in_box,ind], dy[ind_in_box])).T
        in_box = in_box[in_box[:,1].argsort()]

        t_m, h_m = float("-inf"), float("-inf") # 3-4
        l, r = box[0,ind], box[1,ind]           # 1
        n = in_box.shape[0]
        npos = in_box[:,1].sum()
        wracc_m = start_q                       # 2
        
        t = np.unique(in_box[:,0])              # define T 
        for i in range(0,len(t)):               # 5
            if i != 0:
                tmp = in_box[in_box[:,0] == t[i-1]]
                n = n - tmp.shape[0]            # 6
                npos = npos - tmp[:,1].sum()    # 6
            h = self._wracc(n, npos, N, Np)     # 7
            if h > h_m:                         # 8
                h_m = h                         # 9
                t_m = t[i]                      # 10 
            tmp = in_box[np.logical_and(in_box[:,0] >= t_m, in_box[:,0] <= t[i])]
            n_i = tmp.shape[0]
            npos_i = tmp[:,1].sum()
            wracc_i = self._wracc(n_i, npos_i, N, Np)
            if wracc_i > wracc_m:               # 11 
                l = t_m                         # 12 
                r = t[i]                        # 12 
                wracc_m = wracc_i               # 13 
        box_new = box.copy()
        box_new[:,ind] = [l,r]    
        return [box_new, wracc_m, int(not wracc_m == start_q)]   
    
    def _wracc(self, n, npos, N, Np):
        return (n/N)*(npos/n - Np/N)
    
    def _get_initial_restrictions(self, data):
        maximum = data.max(axis=0)
        minimum = data.min(axis=0)
        return np.vstack((minimum, maximum))


# =============================================================================
# # Test
# 
# import pandas as pd
# df = pd.read_csv("src\\main\\generators\\testdata.csv")
# dx = df.iloc[:,[0]].copy().to_numpy()
# dy = df.iloc[:,6].copy().to_numpy()
# 
# bi = BestInterval()
# bi.find(dx, dy)
# 
# dy[0] = 0
# bi.find(dx, dy)
# 
# dy = 1 - dy
# bi.find(dx, dy)
# 
# dy[1:5] = 1
# bi.find(dx, dy)
# 
# import time
# dx = df.iloc[:,0:6].copy().to_numpy()
# dy = df.iloc[:,6].copy().to_numpy()
# bi = BestInterval(depth = 3)
# start = time.time()
# bi.find(dx, dy)
# end = time.time()
# print(end - start) # 0.12s
#     
# box = bi._get_initial_restrictions(dx)
# start_q = 0
# bi._refine(dx, dy, box, 0, start_q)
# bi._refine(dx, dy, box, 1, start_q)
# 
# # generated data 
# 
# np.random.seed(seed=1)
# dx = np.random.random((1000,4)) 
# dy = ((dx > 0.3).sum(axis = 1) == 4) - 0
# dx[:,1] = dx[:,1]*2
# bi = BestInterval(depth = 4, beam_size = 1)
# bi.find(dx, dy)
# bi = BestInterval(depth = 4, beam_size = 4)
# bi.find(dx, dy)
# bi = BestInterval(depth = 4, beam_size = 1, add_iter = 4)
# bi.find(dx, dy)
# 
# dx = dx[:,[3,1,2,0]]
# bi = BestInterval(depth = 3, beam_size = 1)
# bi.find(dx, dy)
# bi = BestInterval(depth = 3, beam_size = 4)
# bi.find(dx, dy)
# bi = BestInterval(depth = 3, beam_size = 1, add_iter = 3)
# bi.find(dx, dy)
# =============================================================================


'''
from typing import List

import numpy as np

import pandas as pd
from rpy2.robjects import pandas2ri  # install any dependency package if you get error like "module not found"
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage


pandas2ri.activate()
base = importr('base')

class BestInterval:

    def __init__(self, beam_size=1, depth=5):
        self.beam_size = beam_size
        self.depth = depth

    def find(self, X, y, regression=True) -> List[pd.DataFrame]:
        bi = SignatureTranslatedAnonymousPackage(self.get_rstring(), "bi")
        X_cols = X.columns
        result = bi.beam_refine(X, y, beam_size=self.beam_size, depth=self.depth)
        result = pd.DataFrame(result, columns=X_cols)
        return [_get_initial_restrictions(X), result]

    def get_rstring(self):
        return open("src/main/R/Refinement.R", mode="r").read()
    
    def _get_initial_restrictions(data: pd.DataFrame) -> pd.DataFrame:
        maximum: pd.DataFrame = data.max(axis=0)
        minimum: pd.DataFrame = data.min(axis=0)
        return pd.DataFrame(data=[minimum, maximum], columns=data.columns)

'''