'''
Ici on va mettre toutes les fonctions qui nous seront utiles pour le projet et qui sont pas directement liées à la partie implimentation.
'''

import numpy as np

def diag_matrice_croissant(X):
    '''
    Fonction qui prend en entrée une matrice carrée X et qui calcule les valeurs propres et vecteurs propres de X, 
    puis renvoie la matrice diagonale avec les valeurs propores classées par ordre croissant et la matrice
    des vecteurs propres associée.
    '''
    # On trie les valeurs propres dans l'ordre croissant
    sigma, P = np.linalg.eigh(X)
    Sigma_decroissant = sigma[::-1]
    P_decroissant = P[:, ::-1]
    Sigma=np.diag(Sigma_decroissant)
    return Sigma, P_decroissant

def frobenius_scalar_product(A, B):
    return np.trace(A@ B.T)

def Phi_operator(X, A):
    '''
    Cette focntion permet de calculer l'opérateur Phi comme défini dans la page 6 du papier.
    '''
    n = A.shape[1]  # Nombre de colonnes dans A
    L = np.zeros(n)
    for i in range(n):
        Ai = A[:, i:i+1]  # Sélectionner la colonne i de A
        L[i] = frobenius_scalar_product(X, Ai@Ai.T)
    return L.reshape(-1, 1)

def Operateur_A(Phi):
    '''
    Étant donné une matrice Phi, cette fonction construit la matrice A donnée dans la page 8 de l'article.
    '''
    n = Phi.shape[1]

    # Initialiser la matrice A avec une taille appropriée
    A = np.zeros((n, n*n))

    # Construire la matrice L
    for i in range(n):
        col_i = Phi[:, i:i+1]  # Sélectionner la colonne i de Phi
        A[i*n:(i+1)*n, :] = np.kron(col_i.T, col_i.T)
    return A

def Operateur_T(P,sigma,S,mu,epsilon):

    '''
    Étant donnée une matrice S, cette fonction retourne l'opérateur T évalué en S, en utilisant 
    la  décomposition donnée dans la page 9.
    '''
    indices_positifs = np.where(sigma > 0)[0]
    # Sélectionner les vecteurs propres correspondants
    P_positifs = P[:, indices_positifs]

    U=P_positifs.T@S
    G=P_positifs@(1/(2*mu)*(U@P_positifs)@P_positifs.T + epsilon*(U@P_positifs)@P_positifs.T)
    return G+G.T


def moving_average(data, window_size):
    smoothed_data = []
    for i in range(len(data)):
        if i < window_size // 2:
            smoothed_data.append(sum(x[1] for x in data[:i+window_size//2+1]) / (i + window_size//2 + 1))
        elif i >= len(data) - window_size // 2:
            smoothed_data.append(sum(x[1] for x in data[i-window_size//2:]) / (len(data) - i + window_size//2))
        else:
            smoothed_data.append(sum(x[1] for x in data[i-window_size//2:i+window_size//2+1]) / window_size)
    return [(data[i][0], smoothed_data[i]) for i in range(len(data))]
    

def kernel_cost(gamma, data, reg):

    KX2 = data['KX2']
    KY2 = data['KY2']
    KX3 = data['KX3']
    KY3 = data['KY3']

    tmp1 = np.mean(KX3) + np.mean(KY3)
    tmp2 = (np.mean(KX2, axis=0) + np.mean(KY2, axis=0)) @ gamma
    c = (tmp1 - tmp2) / (2 * reg)

    return c


def gradient(gamma, X, Phi, Q, z, reg1, reg2):
    """
    Compute the gradient of the objective function.

    Args:
        gamma (np.ndarray): m*1 vector.
        X (np.ndarray): m*m matrix.
        Phi (np.ndarray): m*m matrix.
        Q (np.ndarray): m*m matrix.
        z (np.ndarray): m*1 vector.
        reg1 (float): Regularization parameter.
        reg2 (float): Regularization parameter.

    Returns:
        g_gamma (np.ndarray): m*1 vector.
        g_X (np.ndarray): m*m matrix.
    """
    m = len(z)
    H = Phi.T @ X @ Phi
    g_gamma = (Q @ gamma - z) / (2 * reg2) - np.diag(H)
    g_X = Phi @ np.diag(gamma) @ Phi.T + reg1 * np.eye(m)
    return g_gamma, g_X


def residue(gamma, X, Phi, Q, z, reg1, reg2):
    """
    Compute the residue of the objective function.

    Args:
        gamma (np.ndarray): m*1 vector.
        X (np.ndarray): m*m matrix.
        Phi (np.ndarray): m*m matrix.
        Q (np.ndarray): m*m matrix.
        z (np.ndarray): m*1 vector.
        reg1 (float): Regularization parameter.
        reg2 (float): Regularization parameter.

    Returns:
        r_gamma (np.ndarray): m*1 vector.
        r_X (np.ndarray): m*m matrix.
    """
    # Compute the gradient
    g_gamma, g_X = gradient(gamma, X, Phi, Q, z, reg1, reg2)  
    X_new = X - g_X
    V, D = np.linalg.eigh(X_new)
    X_new = V @ np.maximum(D, 0) @ V.T
    r_X = X - X_new
    
    return g_gamma, r_X

