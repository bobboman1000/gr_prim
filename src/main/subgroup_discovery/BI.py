
import numpy as np
import pandas as pd
import warnings


class BestInterval:

    def __init__(self, beam_size=1, depth=5):
        self.beam_size = beam_size
        self.depth = depth
        
    def find(self, dx, dy):
        if np.logical_or(min(dy) < 0, max(dy) > 1):
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
                    retain = res_tab[:,0] >= np.sort(res_tab)[::-1][self.beam_size - 1,0]
                    res_tab = res_tab[retain]
                    res_box = [res_box[i] for i in np.where(retain)[0]]
                for k in range(0, len(res_tab)):
                    if res_tab[k, 1] == 1:
                        res_tab[k, 1] == 0
                        inds_r = np.arange(0, dim, 1)[np.arange(0, dim, 1) != res_tab[k, 2]]
                        for i in inds_r:
                            tmp = self._refine(dx, dy, res_box[k], i, res_tab[k, 0])
                            res_box.append(tmp[0])
                            res_tab = np.concatenate((res_tab, np.array([[tmp[1], tmp[2], i]])), axis = 0)
        
        winner = np.where(res_tab[:,0] == max(res_tab[:,0]))[0][0]
        return res_box[winner]

    def _refine(self, dx, dy, box, ind, start_q):
        # below numbers correspond to the row numbers in the pseudo-code description
        # from "Efficient algorithms for finding richer subgroup descriptions in 
        # numeric and nominal data" (Algorithm 3)
        N = len(dx)
        Np = sum(dy)
        
        ind_in_box = np.ones(N, dtype = bool)
        for i in range(0,dx.shape[1]):
            if not i == ind:
                ind_in_box = np.logical_and(ind_in_box, np.logical_and(dx.iloc[:,i] >= box.iloc[0,i], dx.iloc[:,i] <= box.iloc[1,i]))
        in_box = pd.concat([dx[ind_in_box].iloc[:,ind], dy[ind_in_box]], axis=1)
        in_box.columns = ['x', 'y']
        in_box = in_box.sort_values(by = 'x')
        
        t_m, h_m = float("-inf"), float("-inf") # 3-4
        l, r = box.iloc[0,ind], box.iloc[1,ind] # 1
        n = len(in_box)
        npos = sum(in_box['y'])
        wracc_m = start_q                       # 2
        
        t = in_box['x'].unique()                # define T 
        for i in range(1,len(t)):               # 5
            tmp = in_box[in_box['x'] == t[i-1]]
            n = n - len(tmp)                    # 6
            npos = npos - sum(tmp['y'])         # 6
            h = self._wracc(n, npos, N, Np)     # 7
            if h > h_m:                         # 8
                h_m = h                         # 9
                t_m = t[i]                      # 10 
            tmp = in_box[np.logical_and(in_box['x'] >= t_m, in_box['x'] <= t[i])]
            n_i = len(tmp)
            npos_i = sum(tmp['y'])
            wracc_i = self._wracc(n_i, npos_i, N, Np)
            if wracc_i > wracc_m:               # 11 
                l = t_m                         # 12 
                r = t[i]                        # 12 
                wracc_m = wracc_i               # 13 
        box_new = box.copy()
        box_new.iloc[:,ind] = [l,r]    
        return [box_new, wracc_m, int(not wracc_m == start_q)]   
    
    def _wracc(self, n, npos, N, Np):
        return (n/N)*(npos/n - Np/N)
    
    def _get_initial_restrictions(self, data) -> pd.DataFrame:
        maximum = data.max(axis=0)
        minimum = data.min(axis=0)
        return pd.DataFrame(data = [minimum, maximum], columns = data.columns)




df = pd.read_csv("src\\main\\generators\\testdata.csv")
dx = df.iloc[:,[0]]
dy = df.iloc[:,6]

bi = BestInterval()
bi.find(dx, dy)

box = bi._get_initial_restrictions(dx)
start_q = 0
bi._refine(dx, dy, box, 0, start_q)
bi._refine(dx, dy, box, 1, start_q)



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