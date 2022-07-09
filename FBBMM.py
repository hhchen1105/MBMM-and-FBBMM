from numpy import log
import numpy as np
from scipy.stats import beta
from scipy.special import gamma
import math
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.special import logsumexp
from scipy import integrate

class FBBMM:
    def __init__(self, n_components, n_runs, param, tol = 1e-6, verbose=0, verbose_interval=1):
        self.n_components = n_components # number of Guassians/clusters
        self.tol = tol
        self.n_runs = n_runs
        self.param = param
        self.pi = np.array([1./self.n_components for i in range(self.n_components)])
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        
    def get_param(self):
        return (self.param, self.pi)
    
    def get_pi(self):
        return self.pi

    def fit(self, X):
        '''
        Parameters:
        -----------
        X: (N x d), data 
        '''
        N = X.shape[0]
        d = X.shape[1]
                    
        lower_bound  = -np.infty
        try:
            for run in range(self.n_runs):  
                prev_lower_bound  = lower_bound
           
                log_prob_norm, log_resp = self.e_step(X)
              
                self.m_step(X, log_resp)                     
              
                lower_bound = log_prob_norm
                if self.verbose != 0:
                    if run % self.verbose_interval == 0:
                        print('run: ', run, ', lower_bound: ', lower_bound)  
                        
                if abs(prev_lower_bound-lower_bound) < self.tol:          
                    break

        except Exception as e:
            print(e)
    
    def log_gamma_function(self, z):
        """
        Stirling’s formula of gamma function
        z : float
        """
        return 0.5*(np.log(2*np.pi)-np.log(z))+z*(np.log(z+(1/(12*z-(1/10*z))))-1)
        
    
    def estimate_log_weights(self):
        return np.log(self.pi)
    
    def estimate_log_prob(self, X, param):
        """Estimate the log Multivariate Beta probability.
        
        Parameters    
        X: (n_samples, n_features)   
        
        Returns
        log_prob: (n_samples, n_components)
        """
        n_samples, n_features = X.shape
        n_components = self.n_components
        
        log_prob = np.empty((n_samples, n_components))
         
        for n, x in enumerate(X):

            for c, para in enumerate(param): 
            
                a11 = para[0]
                a10 = para[1]
                a01 = para[2]
                a00 = para[3]
                
                def f(u11,x,y):
                    return pow(u11,a11-1)*pow(x-u11,a10-1)*pow(y-u11,a01-1)*pow(1-x-y+u11,a00-1)
                        
                lower_bound = max(0, x[0]+x[1]-1)
                upper_bound = min(x[0],x[1])
              
                val = integrate.quad(f, lower_bound, upper_bound, args=(x[0],x[1]))[0]
              
                log_prob[n,c] = np.log(val)+self.log_gamma_function(np.sum(para))-np.sum(self.log_gamma_function(para))
  
        return log_prob
    
    def estimate_weighted_log_prob(self, X, param):
        """
        Returns
        (n_samples, n_components)
        """
        
        return self.estimate_log_weights() + self.estimate_log_prob(X, param)
    
    def estimate_log_prob_resp(self, X, param):
        """log pi + log P(X | Z)
        
        Parameters    
        X: (n_samples, n_features)   
        ----------    
            
        Returns    
        weighted_log_prob: (n_samples, n_component)
        """
        
        weighted_log_prob = self.estimate_weighted_log_prob(X, self.param)
        
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
  
        
        with np.errstate(under='ignore'):
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
            
        return log_prob_norm, log_resp
    
    def e_step(self, X):
        '''
        X: (n_samples, n_features)
        
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X
        log_responsibility: (n_samples, n_components)
        '''
        
        log_prob_norm, log_resp = self.estimate_log_prob_resp(X, self.param)
         
        return np.mean(log_prob_norm), log_resp
                
    def m_step(self, X, log_resp):
        """
        Parameters
        X: (n_samples, n_features)
        
        log_resp: (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.:
            
        """
        n_samples, n_features = X.shape
        n_components = self.n_components
        

        resp = np.exp(log_resp)    
        
        for c in range(n_components):
            self.pi[c] = np.sum(resp[:,c]) / n_samples

        param_num = n_components*(n_features+2) # total parameters num
        x_guess = np.array([])
        lower = np.array([])
        upper = np.array([])
        
        #initialize optimizatin parameters and boundary
        for i in range(param_num):
            x_guess = np.append(x_guess,self.param[i//(n_features+2)][i%(n_features+2)])
            lower = np.append(lower, 1e-1) #lower
            upper = np.append(upper, 100) #upper
        
        def new_loglikeli(para):
            
            para = para.reshape(n_components, n_features+2)
  
            op_weighted_log_prob = self.estimate_weighted_log_prob(X, para)
            op_weighted_log_prob = np.multiply(resp, op_weighted_log_prob)
            op_weighted_log_prob = np.sum(op_weighted_log_prob)
            
            return -op_weighted_log_prob
        
 
        bounds = Bounds(lower, upper)
        res = minimize(new_loglikeli, x_guess, method='SLSQP',options={'ftol': 1e-4}, bounds=bounds)
        
        
        for i in range(param_num):
            self.param[i//(n_features+2)][i%(n_features+2)] = res.x[i]    
    
    def predict(self, X):   
        """Predict the labels for the data samples in X using trained model.
        
        Parameters
        ----------
        X: (n_samples, n_features)
        

        Returns 
        ----------
        labels: (n_samples,)
            Component labels.
        """       
        return self.estimate_weighted_log_prob(X, self.param).argmax(axis=1)
   
    
       