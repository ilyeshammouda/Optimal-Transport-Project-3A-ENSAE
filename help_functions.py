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

    
