import numpy as np
from time import time
import psutil
import resource
from numpy.linalg import norm
from scipy.sparse import diags
from help_functions import kernel_cost, gradient, residue


class SSN_numpy:
    """
            Solve the conic optimization model using a semi-smooth Newton method.

        Input:
        - data: dict, contient les données nécessaires pour le modèle
        - reg1: float, régularisation 1
        - reg2: float, régularisation 2
    """
    def __init__(self, data, alph_1, alph_2, beta_0, beta_1, beta_2, theta_upper,
                 theta_lower, reg1, reg2, EG_rate, nIter):
        self.data = data
        self.alph_1 = alph_1
        self.alph_2 = alph_2
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.theta_upper = theta_upper
        self.theta_lower = theta_lower
        self.reg1 = reg1
        self.reg2 = reg2
        self.EG_rate = EG_rate
        self.nIter = nIter

     # One  itteration  of the SSN algorithme
       
    def SSN_main(r_gamma, r_X, gamma, X, mu, Q, Phi, reg1, reg2):
        m = len(gamma)

        # The first step
        Z = X - (Phi @ np.diag(gamma) @ Phi.T + reg1 * np.eye(m))
        Sigma, P = np.linalg.eigh(Z)
        Sigma = Sigma[::-1]
        P = P[:, ::-1]
        alpha = np.where(Sigma > 0)[0]
        beta = np.where(Sigma <= 0)[0]
        Omega = np.zeros((m, m))
        Omega[np.ix_(alpha, alpha)] = np.ones(len(alpha))
        eta = 1 - np.outer(Sigma[beta], 1 / Sigma[alpha])
        eta = 1 / eta
        Omega[np.ix_(alpha, beta)] = eta.T
        Omega[np.ix_(beta, alpha)] = eta
        L = Omega / (mu + 1 - Omega)

        T = r_X + P @ (L * (P.T @ r_X @ P)) @ P.T
        H = Phi.T @ T @ Phi
        d_gamma = -r_gamma - np.diag(H) / (1 + mu)
        d_X = -r_X

        # The second step (CG)
        y = d_gamma
        K = P.T @ Phi
        H = K.T @ (L * (K @ np.diag(y) @ K.T)) @ K
        r = d_gamma - ((0.5 / reg2) * Q @ y + mu * y + np.diag(H))
        p = r
        rr = r.T @ r
        for i in range(min(m // 5, 50)):
            H = K.T @ (L * (K @ np.diag(p) @ K.T)) @ K
            Ap = (0.5 / reg2) * Q @ p + mu * p + np.diag(H)
            ss1 = rr / (p.T @ Ap)
            y += ss1 * p
            r -= ss1 * Ap
            if np.linalg.norm(r) < 1e-6:
                break
            ss2 = r.T @ r / rr
            p = r + ss2 * p
            H = K.T @ (L * (K @ np.diag(y) @ K.T)) @ K
            r = d_gamma - ((0.5 / reg2) * Q @ y + mu * y + np.diag(H))
            rr = r.T @ r
        d_gamma = y
        d_X = (d_X + P @ (L * (P.T @ d_X @ P)) @ P.T) / (1 + mu)

        # The third step
        d_X -= P @ (L * (K @ np.diag(d_gamma) @ K.T)) @ P.T

        return d_gamma, d_X
    def fit(self,verbose,usage=False,Windows=True):
        """

        - verbose: bool, indique si les détails doivent être imprimés
        -usage: bool, indique si les détails sur l'utilisation du CPU doivent être affichés
        -Windows: bool, indiquez True si vous êtes sur Windows afin que le calcul de l'utilisation du CPU soient calculée
        Output:
        - gamma: array, vecteur de taille m
        - c: float, coût
        - t: float, temps de résolution
        - res_time: list, historique du temps de résolution
        - res_norm: list, historique de la norme du résidu
        -usage_lst: une liste avec l'utilisation par seconde du CPU pour chaque ittération
        -details: une liste qui contient plus d'informations sur l'utilisation des ressources pour chaque ittération
        """
        data=self.data
        alph_1=self.alph_1
        alph_2=self.alph_2
        beta_0=self.beta_0
        beta_1=self.beta_1
        beta_2=self.beta_2
        theta_upper=self.theta_upper
        theta_lower=self.theta_lower
        reg1=self.reg1
        reg2=self.reg2
        EG_rate=self.EG_rate
        nIter=self.nIter


        # Input data
        M = data['M']
        Phi = data['Phi']
        KX1 = data['KX1']
        KY1 = data['KY1']
        KX2 = data['KX2']
        KY2 = data['KY2']

        # Initialization
        m = len(M)
        Q = KX1 + KY1
        z = np.mean(KX2, axis=0) + np.mean(KY2, axis=0) - 2 * reg2 * M

        gamma = np.ones(m) /m
        v_gamma = gamma 
        X = np.ones((m, m)) /(m*m)
        v_X = X
        theta = 1.0    
        r_gamma, r_X = residue(gamma, X, Phi, Q, z, reg1, reg2)
        mu = theta * (norm(r_gamma) + norm(r_X, 'fro'))
        res_time = [0]
        res_norm = [mu]

        if verbose:
            print('\n-------------- SSNEG ---------------')
            print('iter |  cost  |  residue  |  time')
        
        tstart = time()
        if Windows:
            start_usage=psutil.cpu_times()
        else:
            start_usage = resource.getrusage(resource.RUSAGE_SELF)
        
        usage_lst=[(0,0)]
        details=[]
        # Main loop
        for iter in range(1, nIter + 1):

            # Compute EG step
            g_gamma, g_x = gradient(v_gamma, v_X, Phi, Q, z, reg1, reg2)
            v_gamma_mid = v_gamma - EG_rate * g_gamma
            v_X_mid = v_X - EG_rate * g_x
            g_gamma, g_x = gradient(v_gamma_mid, v_X_mid, Phi, Q, z, reg1, reg2)
            v_gamma = v_gamma - EG_rate * g_gamma
            v_X = v_X - EG_rate * g_x


            # Compute the residue function
            mu = norm(r_gamma) + norm(r_X, 'fro')

            # Compute SSN step
            d_gamma, d_X = SSN_numpy.SSN_main(r_gamma, r_X, gamma, X,  (m / 5) * theta * mu, Q, Phi, reg1, reg2)

            
            # Compute the next iterate
            r_gamma_, r_X_ = residue(gamma + d_gamma, X + d_X, Phi, Q, z, reg1, reg2)
            r_gamma_v, r_X_v = residue(v_gamma, v_X, Phi, Q, z, reg1, reg2)
            if (norm(r_gamma_) + norm(r_X_, 'fro')) < (norm(r_gamma_v) + norm(r_X_v, 'fro')):
                gamma += d_gamma
                X += d_X
            else:
                gamma = v_gamma
                X = v_X

            # Update the parameter theta
            r_gamma, r_X = residue(gamma, X, Phi, Q, z, reg1, reg2)
            rho_div_norm = -(np.dot(r_gamma, d_gamma) + np.trace(np.dot(r_X.T, d_X))) / (norm(d_gamma) ** 2 + norm(d_X, 'fro') ** 2)
            if rho_div_norm >= alph_2:
                theta = max(beta_0 * theta, theta_lower)
            elif rho_div_norm >= alph_1:
                theta = beta_1 * theta
            else:
                theta  = min(theta_upper, beta_2 * theta)

            if mu < 5e-3:  # 5e-3
                c = kernel_cost(gamma, data, reg2)
                t = time() - tstart
                res_time.append(t)
                res_norm.append(mu)
                if verbose:
                    print(f'{iter:5}|{c:3.2e}|{mu:3.2e}|{t:3.2e}')
                break

            if iter % 50 == 0:
                c = kernel_cost(gamma, data, reg2)
                t = time() - tstart
                res_time.append(t)
                res_norm.append(mu)
                if verbose:
                    print(f'{iter:5}|{c:3.2e}|{mu:3.2e}|{t:3.2e}')
            
            if usage:
                print(f'{iter:5}|{cpu_time:3.2e}')
            
            
            c = kernel_cost(gamma, data, reg2)
            t = time() - tstart
            if iter==1:
                if Windows:
                    current_usage = psutil.cpu_times()
                    details.append((iter,psutil.cpu_stats(),psutil.cpu_percent(),psutil.cpu_freq()))
                    cpu_time= current_usage.user - start_usage.user
                else:
                    current_usage = resource.getrusage(resource.RUSAGE_SELF)
                    details.append((iter,current_usage))
                    cpu_time = current_usage.ru_utime -start_usage.ru_utime
            else:
                if Windows:
                    current_usage = psutil.cpu_times()
                    details.append((iter,psutil.cpu_stats(),psutil.cpu_percent(),psutil.cpu_freq()))
                    cpu_time = current_usage.user  - usage_lst[-1][1]
                else:    
                    current_usage = resource.getrusage(resource.RUSAGE_SELF)
                    details.append((iter,current_usage))
                    cpu_time = current_usage.ru_utime - usage_lst[-1][1]
            usage_lst.append((iter,cpu_time))
        return gamma, c, t, res_time, res_norm,usage_lst,details
