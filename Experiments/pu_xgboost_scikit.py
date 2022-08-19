import xgboost as xgb
from scipy.special import expit
import numpy as np

import math

from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_array, check_consistent_length
from sklearn.utils import check_random_state

class PUBoost:
    def __init__(self, obj, n_estimators=500, subsample=1.0, max_depth=5,
                 colsample_bytree=1.0, min_child_weight=1, learning_rate=0.01,
                 lambda1=0, lambda2=0, label_freq=None,
                 validation=False, random_state=None):
      
        self.obj = obj
        self.label_freq = label_freq
        self.random_state = random_state
        self.validation = validation
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.max_depth = max_depth
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.validation = validation
        self.learning_rate = learning_rate
        
        # alpha is l1, lambda is l2
        params = {'random_state': random_state, 'tree_method': 'exact',
                  'verbosity': 0, 'reg_alpha': lambda1, 'reg_lambda': lambda2,
                  'n_estimators': n_estimators, 'subsample':subsample,
                  'max_depth':max_depth, 'colsample_bytree':colsample_bytree,
                  'min_child_weight':min_child_weight, 'eta':learning_rate}
        
        if obj == 'ce' or obj == 'weightedce':
            params['objective'] = 'binary:logistic'
        elif obj == 'upu' or obj =='nnpu':
            params['disable_default_eval_metric'] = 1

        self.params = params

    def fit(self, x_train, y_train, x_val=None, y_val=None):
      if self.obj == 'ce':
          dtrain = xgb.DMatrix(x_train, label=y_train)
          dval = xgb.DMatrix(x_val, label=y_val)

          xgboost = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=self.n_estimators, early_stopping_rounds=50,
                          evals=[(dval, 'eval')], verbose_eval=False)
      elif self.obj == 'upu':
        
          dtrain = xgb.DMatrix(x_train, label=y_train)  
          if self.validation == True:
            dval = xgb.DMatrix(x_val, label=y_val)
            
          label_freq_ = self.label_freq
          eps = 1e-4 # small value to avoid log(0)

          def upu_train(raw_scores, y_true):
               y = y_true.get_label()
               scores = expit(raw_scores)
               scores = np.clip(scores, eps, 1 - eps)
               # Gradient
               grad = -y*(1.0/label_freq_)*(1-scores) + y*(1.0-1.0/label_freq_)*scores + (1-y)*scores
 
               # Hessian
               hess = scores*(1-scores)
 
               return grad, hess
 
          def upu_val(raw_scores, y_true):
              y = y_true.get_label()
              scores = expit(raw_scores)
              scores = np.clip(scores, eps, 1 - eps)
              # print('scores: ', scores)
              print('number of infs in scores: ', np.sum(np.isinf(scores)))
              loss = np.empty_like(scores)
              pos = y == 1
             
              loss[pos] = -1.0*((1.0/label_freq_)*np.log(scores[pos]) + (1.0-1.0/label_freq_)*np.log(1.0 - scores[pos]))
              # print('Loss of positives: ', loss[pos])
              print('number of infs in loss pos: ', np.sum(np.isinf(loss[pos])))

              loss[~pos] = -1.0*(np.log(1.0 - scores[~pos]))
              # print('Loss of negatives: ', loss[~pos])
              print('number of infs in loss neg: ', np.sum(np.isinf(loss[~pos])))


              upu = loss.mean()
              return 'uPU', upu
                          
          if self.validation == True:
            xgboost = xgb.train(params=self.params, dtrain=dtrain, obj=upu_train, 
                                feval=upu_val, num_boost_round=500,
                                early_stopping_rounds=50, evals=[(dval, 'eval')], 
                                verbose_eval=False)
          else:
            
            xgboost = xgb.train(params=self.params, dtrain=dtrain, obj=upu_train, 
                                num_boost_round=100)


      elif self.obj == 'nnpu':
        
          dtrain = xgb.DMatrix(x_train, label=y_train)
          if self.validation == True:
            dval = xgb.DMatrix(x_val, label=y_val)
          
          label_freq_ = self.label_freq
          eps = 1e-9  # small value to avoid log(0)

          def nnpu_train(raw_scores, y_true):
               y = y_true.get_label()
               scores = expit(raw_scores) + eps
               zero_col_train = np.zeros_like(y)
               pu_col_train = 1 - y*(1/label_freq_)
               
               nnpu_col_train = np.amax(np.vstack((zero_col_train, pu_col_train)),0)
                                
               # Gradient
               grad = -y*(1.0-scores)*(1/label_freq_) + nnpu_col_train*scores
               # Hessian
               hess = (y*(1/label_freq_) + nnpu_col_train) * scores * (1.0 - scores)
 
               return grad, hess
 
          def nnpu_val(raw_scores, y_true):
              y = y_true.get_label()
              scores = expit(raw_scores) + eps
              
              loss = np.empty_like(scores)

              pos = y == 1
              
              zero_col = np.zeros_like(scores[pos])
              pos_col = -1.0*np.log(1.0-scores[pos])*(1.0-(1/label_freq_))
              max_col = np.amax(np.vstack((zero_col, pos_col)),0)
             
              loss[pos] = -1.0*np.log(scores[pos])*(1/label_freq_) + max_col
              
              
              loss[~pos] = -1.0*np.log(1.0-scores[~pos])

              nnpu = loss.mean()

              return 'nnPU', nnpu

          if self.validation == True:
            xgboost = xgb.train(params=self.params, dtrain=dtrain, obj=nnpu_train, 
                                feval=nnpu_val, num_boost_round=500,
                                early_stopping_rounds=50, evals=[(dval, 'eval')], 
                                verbose_eval=False)
          else:
            
            xgboost = xgb.train(params=self.params, dtrain=dtrain, obj=nnpu_train, 
                                num_boost_round=100)
              
      return xgboost



      
                      
def make_noisy_negatives(y, 
                         X = None, 
                         flip_ratio = None, 
                         label_noise = 'uniform',
                         n_neighbors = 'auto',
                         true_prop_pos = None, 
                         pollution_ratio = None,
                         random_state = None):
  
  """ Flipping Negatives into Positives
  
    Generating noisy negatives by flipping. It can generate noisy negatives 
    using a flip ratio (percentage of flipped positives) or a pollution ratio 
    (percentage of hidden positives in negatives)
    
    Parameters
    ----------
    y                 : array-like, compulsory
    X                 : array-like, sparse matrix of shape (n_samples, n_features)
    flip_ratio        : float, optional (default = None)
        Percentage of positives into noisy negatives.
    true_prop_pos     : float, optional (default = None)
        Percentage of true positives in the dataset.
    pollution_ratio   : float, optional (default = None)
        Percentage of hidden positives (noisy negs) among negatives.
    random_seed       : int, optional (default = 123)
      
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.

    """
  if X is not None:  
    X = check_array(X, accept_sparse='csr', dtype=np.float64)
  
  if X is None and label_noise == 'knn':
    raise ValueError("Label noise by Nearest Neighbors requires features matrix ``X`` ")
  
  random_state = check_random_state(random_state)
  
  y_ = y.copy()
  y_ = check_array(y_, ensure_2d=False, dtype=None)
  check_consistent_length(X, y_)
  n_pos = np.sum(y_ == 1)
  
  true_prop_pos = np.mean(y_)
  
  if n_neighbors == 'auto':
    n_neighbors = math.floor(math.sqrt(len(y_)))
    
  if flip_ratio is None and pollution_ratio is None:
    raise ValueError("Either flip rate (`flip_ratio`) or pollution ratio (`pollution_ratio`) must be known")
 
  if flip_ratio is not None and true_prop_pos is not None:
    flip_ratio_ = flip_ratio
       
  else:
    raise ValueError("If flip ratio is not None, true proportion of positives must be known")

  ## Types of Label Noise
  
  # Uniform Label Noise
  if label_noise == 'uniform':
    p_ = None

  # KNN Label Noise
  elif label_noise == 'knn':
    

    dist_matrix = pairwise_distances(X, n_jobs = -1)
    ix_pos = np.where(y == 1)[0]
    ix_neg = np.where(y == 0)[0]

    pos_dist_mt = dist_matrix[ix_pos][:,ix_neg]
    
    pos_dist_mt.sort(axis = 1)
    k_mean_dist = np.mean(pos_dist_mt[:,:n_neighbors], axis = 1)
    
    sample_weight = np.array([k_mean_dist[i]/np.sum(k_mean_dist) for i,_ in enumerate(k_mean_dist)])
    p_ = sample_weight.reshape(-1,)
    
    
  else: 
    raise ValueError('labeling is not valid; Use knn or uniform instead')
    
  ## Size of Flipped Negatives
  if label_noise == 'knn':
    size_ = min(int(np.ceil(flip_ratio_*n_pos)), len(np.nonzero(p_)[0]))  
    
  else: 
    size_ = int(np.ceil(flip_ratio_*n_pos))
  
  ## Index to be Flipped Positives
  indices = random_state.choice(range(n_pos), 
                                size_, 
                                replace = False, 
                                p = p_)
    
  print('Number Samples First Stage ', len(indices))

  if len(indices) < int(np.ceil(flip_ratio_*n_pos)):

      print('Sampling Second Stage - Random Sampling on Complement Index Set')

      indices_comp = np.setdiff1d(range(n_pos), indices)
      size_extra_ = int(np.ceil(flip_ratio_*n_pos)) - len(indices)
      indices_extra = random_state.choice(indices_comp,
                                          size_extra_,
                                          replace=False)
      
      print('Number Samples Second Stage ', len(indices_extra))

      indices = np.hstack((indices, indices_extra))
    
  
  pos_y = y_[y_ == 1].copy()
  pos_y[indices] = 0
  
  y_[y_ == 1] = pos_y

  mislabeled_rate = 1 - np.mean(pos_y)
  pol_ratio = (mislabeled_rate*true_prop_pos)/(1 - true_prop_pos*(1 - mislabeled_rate))

  print("Flipping ratio (pos. to neg.): {:.3f}".format(mislabeled_rate))
  print("Pollution ratio (noisy neg. among observed neg.): {:.3f}".format(pol_ratio))
  
  return y_
