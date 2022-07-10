from numpy import log
import numpy as np
from scipy.special import gamma
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.special import logsumexp

class MBMM:
    def __init__(self, n_components, n_runs, param, tol = 1e-6, verbose=0, verbose_interval=1):
        """
        n_components: int, default=None
          The number of mixture components.

        n_runs: int, default=None
          The number of EM iterations to perform.

        param: array-like of shape (n_features+1, n_components), default=None
          The parameters of Multivariate Beta distribution.

        tol: float, default=1e-6
          The convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.

        verbose: int, default=0
          Enable verbose output. If not equal to 0 then it prints each iteration step and log probability.

        verbose_interval: int, default=1
          Number of iteration done before the next print.
         
        """
        self.n_components = n_components 
        self.n_runs = n_runs
        self.param = param
        self.tol = tol
        self.pi = np.array([1./self.n_components for i in range(self.n_components)]) #The weights of each mixture components.
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        
    def get_param(self):
        return (self.param, self.pi)
    
    def fit(self, X):
        '''Estimate model parameters with the EM algorithm.

        Parameters  
        ----------  
        X: (n_samples, n_features) 
            List of n_features-dimensional data points. Each row
            corresponds to a single data point. 
        '''

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
        """Stirling’s formula of gamma function 
        log((z-1)!)

        Parameters  
        ----------     
        z : float
        """

        return 0.5*(np.log(2*np.pi)-np.log(z))+z*(np.log(z+(1/(12*z-(1/10*z))))-1)
        
    
    def estimate_log_weights(self):
        return np.log(self.pi)
    
    def estimate_log_prob(self, X, param):
        """Estimate the log Multivariate Beta probability.
        
        Parameters  
        ----------  
        X: (n_samples, n_features)   
        
        param: (n_component, n_features+1)
        

        Returns
        ----------
        log_prob: (n_samples, n_components)
        """
        n_samples, n_features = X.shape
        n_components = self.n_components
        
        log_prob = np.empty((n_samples, n_components))
        
        
        for c, para in enumerate(param):        
            log_prob[:,c] = self.log_gamma_function(np.sum(para))-np.sum(self.log_gamma_function(para)) \
                       + np.sum((para[:-1]-1.)*np.log(X)-(para[:-1]+1.)*np.log(1.-X), axis=1) \
                       - np.sum(para)*np.log(1.+np.sum(X/(1.-X), axis=1))
   
        return log_prob
    
    def estimate_weighted_log_prob(self, X, param):
        """Estimate the weighted log-probabilities, log weights + log P(X | Z).

        Parameters  
        ----------  
        X: (n_samples, n_features)   

        param: (n_component, n_features+1)
        

        Returns
        ----------
        weighted_log_prob: (n_samples, n_components)
        """
        return self.estimate_log_weights() + self.estimate_log_prob(X, param)
    
    def estimate_log_prob_resp(self, X, param):
        """Estimate log probabilities and responsibilities for each sample X.
        
        Parameters 
        ----------     
        X: (n_samples, n_features)   

        param: (n_component, n_features+1)
          
            
        Returns
        ----------      
        log_prob_norm: (n_samples,)
            log p(x)

        log_responsibilities(log_resp): (n_samples, n_components)
            Logarithm of the posterior probabilities
        """

        weighted_log_prob = self.estimate_weighted_log_prob(X, self.param)
        
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
         
        with np.errstate(under='ignore'):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        
        return log_prob_norm, log_resp
    
    def e_step(self, X):
        '''
        Parameters 
        ----------
        X: (n_samples, n_features)
        

        Returns 
        ----------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X
        
        log_responsibility: (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of each sample in X
        '''
        
        log_prob_norm, log_resp = self.estimate_log_prob_resp(X, self.param)

        return np.mean(log_prob_norm), log_resp
    
    def m_step(self, X, log_resp):
        """
        Parameters
        ----------
        X: (n_samples, n_features)
        
        log_resp: (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.:
            
        """
        resp = np.exp(log_resp)
        
        n_samples, n_features = X.shape
        n_components = self.n_components
                   
        #Compute param: (n_components*(n_features+1))
        param_num = n_components*(n_features+1) # total parameters num
        
        #Initialize optimization parameters and boundary
        x_guess = np.array([self.param[i//(n_features+1)][i%(n_features+1)] for i in range(param_num)])
        lower = np.array([1e-8 for _ in range(param_num)])
        upper = np.array([np.inf for _ in range(param_num)])

        def loglikeli(para):       
            para = para.reshape(n_components, n_features+1)
 
            op_weighted_log_prob = self.estimate_weighted_log_prob(X, para)
            op_weighted_log_prob = np.multiply(resp, op_weighted_log_prob)
            op_weighted_log_prob = logsumexp(op_weighted_log_prob, axis=1)
            op_weighted_log_prob = np.sum(op_weighted_log_prob)
            
            return -op_weighted_log_prob

        bounds = Bounds(lower, upper)
        opti_param = minimize(loglikeli, x_guess, method='SLSQP',options={'ftol': 1e-8}, bounds=bounds)

        for i in range(param_num):
            self.param[i//(n_features+1)][i%(n_features+1)] = opti_param.x[i]
            
        

        #Compute weights(pi): (n_components,)
        for c in range(n_components):
            self.pi[c] = np.sum(resp[:,c]) / n_samples 

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
        
