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
    sigma,P=np.linalg.eig(X)
    indices_tri = np.argsort(sigma)[::-1]
    Sigma_decroissant = sigma[indices_tri]
    P_decroissant = P[:, indices_tri]
    Sigma=np.diag(Sigma_decroissant)
    return Sigma,P_decroissant

def frobenius_scalar_product(A, B):
    return np.trace(A@ B.T)

def Phi_operator(X, A):
    n = A.shape[1]  # Nombre de colonnes dans A
    L = np.zeros(n)
    for i in range(n):
        Ai = A[:, i:i+1]  # Sélectionner la colonne i de A
        L[i] = frobenius_scalar_product(X, Ai@Ai.T)
    return L.reshape(-1, 1)
