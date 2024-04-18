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