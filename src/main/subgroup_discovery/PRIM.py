import numpy as np
import sys

class Prim:

    def __init__(self, alpha = 0.05, mass_min = 20, target = 'precision'):
        self.alpha = alpha
        self.mass_min = mass_min
        self.target = target
        
        self.dx = None
        self.dy = None
        self.box = None
        self.N = None
        self.Np = None

    def find(self, X, Y):
        self.dx = X.copy()
        self.dy = Y.copy()
        self.box = self._get_initial_restrictions(self.dx)
        self.N = len(self.dy)
        self.Np = sum(self.dy)
        
        highest = self._target_fun(sum(self.dy), len(self.dy))
        ret_ind = 0        
        boxes = [self.box.copy()]
        i = 1
        while self.dx.shape[0] > self.mass_min and i < 100:
            i = i + 1
            hgh = self._peel_one()
            boxes.append(self.box.copy())
            if hgh > highest:
                highest = hgh
                ret_ind = i
        
        self.dx = None
        self.dy = None
        self.box = None
        self.N = None
        self.Np = None
        
        return boxes[0:ret_ind]
    
    def _peel_one(self):
        ndel = max(round(self.dx.shape[0]*self.alpha), 1)
        hgh, bnd = float("-inf"), float("-inf")
        rn, cn = -1, -1
        if ndel > 0:
            for i in range(0, self.dx.shape[1]):
                bound = np.sort(self.dx[:,i])[(ndel-1):(ndel+1)].sum()/2
                retain = self.dx[:,i] > bound
                tar = self._target_fun(sum(self.dy[retain]), sum(retain))
                if tar > hgh:
                    hgh = tar
                    inds = retain
                    rn = 0
                    cn = i
                    bnd = bound
                bound = np.sort(self.dx[:,i])[::-1][(ndel-1):(ndel+1)].sum()/2
                retain = self.dx[:,i] < bound
                tar = self._target_fun(sum(self.dy[retain]), sum(retain))
                if tar > hgh:
                    hgh = tar
                    inds = retain
                    rn = 1
                    cn = i
                    bnd = bound
        
        self.dx = self.dx[inds]
        self.dy = self.dy[inds]
        self.box[rn,cn] = bnd
        
        return hgh

    def _get_initial_restrictions(self, data):
        maximum = data.max(axis=0)
        minimum = data.min(axis=0)
        return np.vstack((minimum, maximum))
    
    def _target_fun(self, npos, n):
        if self.target == 'precision':
            tar = npos/n
        elif self.target == 'wracc':
            tar = (n/self.N)*(npos/n - self.Np/self.N)
        else:
            sys.exit("The target function is unknown. It should be either wracc or precision")
        return tar


# =============================================================================
# # generated data 
# 
# np.random.seed(seed=1)
# dx = np.random.random((1000,4))
# dy = ((dx > 0.3).sum(axis = 1) == 4) - 0
# 
# import time
# pr_new = Prim()
# start = time.time()
# bp_new = pr_new.find(dx,dy)  
# end = time.time()
# print(end - start)   # ~ 1.1 s
# # the boxes are a bit different from inplementation in "ema_workbench" since
# # we are using "round" operation (i.e. more patient). In case one replaces
# # "round" with "math.ceil" (from "math" module), the results coincide
# 
# # real data 
# 
# import pandas as pd
# df = pd.read_csv("src\\main\\generators\\testdata.csv")
# dx = df.iloc[:,[0]].copy().to_numpy()
# dy = df.iloc[:,6].copy().to_numpy()
# 
# pr_new = Prim()
# bp_new = pr_new.find(dx,dy)
# 
# dy[0] = 0
# bp_new = pr_new.find(dx,dy)
# 
# dy = 1 - dy
# bp_new = pr_new.find(dx,dy)
# 
# dy[1:6] = 1
# bp_new = pr_new.find(dx,dy)
# 
# dx = df.iloc[:,0:6].copy().to_numpy()
# dy = df.iloc[:,6].copy().to_numpy()
# start = time.time()
# bp_new = pr_new.find(dx,dy)
# end = time.time()
# print(end - start) # ~0.3s
# # the boxes are again bit different from inplementation in "ema_workbench".
# # The source of difference here is that for some reason reference implementation
# # does not take bounds average "np.sort(self.dx[:,i])[::-1][(ndel-1):(ndel+1)].sum()/2"
# # ¯\_(ツ)_/¯
# 
# 
# # compare different target functions
# 
# import pandas as pd
# df = pd.read_csv("src\\main\\generators\\testdata.csv")
# dx = df.iloc[:,[0]].copy().to_numpy()
# dy = np.linspace(0, 1, num = dx.shape[0])
# 
# pr_prec = Prim()
# pr_wracc = Prim(target = 'wracc')
# 
# bp_prec = pr_prec.find(dx,dy)
# bp_wracc = pr_wracc.find(dx,dy)
# =============================================================================

'''
import pandas as pd
from ema_workbench.analysis import prim as p, prim_util as pu

class PRIM:

    def __init__(self, threshold=1, mass_min=0.05, wracc: bool=False):
        self.threshold = threshold
        self.mass_min = mass_min
        self.wracc = wracc

    def find(self, X: pd.DataFrame, y: np.ndarray, regression=True):
        if regression:
            loc_mode = p.sdutil.RuleInductionType.REGRESSION
        else:
            loc_mode = p.sdutil.RuleInductionType.BINARY

        if self.wracc:
            obj_function = pu.PRIMObjectiveFunctions.WRACC
        else:
            obj_function = pu.PRIMObjectiveFunctions.ORIGINAL

        prim = p.Prim(x=X, y=y, threshold=self.threshold, mode=loc_mode, obj_function=obj_function, mass_min=self.mass_min)
        box_pred = prim.find_box()

        return box_pred.box_lims
    

import time
np.random.seed(seed=1)
dx = np.random.random((1000,4))
dy = ((dx > 0.3).sum(axis = 1) == 4) - 0
dx = pd.DataFrame(dx, columns = ['x1', 'x2', 'x3' , 'x4']) 
pr = PRIM(threshold = 10)
start = time.time()
bp = pr.find(dx, dy)
end = time.time()
print(end - start) # ~2.8s

import pandas as pd
df = pd.read_csv("src\\main\\generators\\testdata.csv")
dy = df.iloc[:,6].copy().to_numpy()
dx = df.iloc[:,0:6].copy()
start = time.time()
bp = pr.find(dx,dy)
end = time.time()
print(end - start) # ~1.2s
'''