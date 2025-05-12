# Sbírka všech modelů pro lepší import

import math
import numpy as np
import matplotlib.pylab as plt
np.set_printoptions(precision=2)
import scipy.io as sio

from math import erf
from typing import Tuple

# Klasické modely ze scikitlearnu s leave-one-out cross-validací
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV

# ------------

# následující kód byl převzat ze cvičení z NI-BML - původní byl vytvořen O. Chladek
# rozšířen o komentáře
# ---
def low_truncated_normal_distribution(means: np.ndarray, covariance_matrix_diagonal: np.ndarray, lower_bound: float) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    low_truncated_normal_distribution funkce provádí (numericky) stabilní převod momentů normálního rozdělení na ořezané normální rozdělení
    konkrétně: N(means,covariance) -> tN(means,diag(covariance),lower_bound,+inf)
    
    pro tyto účely využívá vektor alf, kterým jsou rozděleny jednotlivé části náhodného vektoru na stabilní a nestabilní indexy (dle toho se mění výpočet)
    """
    # vypočítání poměru lower_bound a kovariance - představuje korekční faktor
    alf = np.divide((lower_bound - means).T, np.sqrt(2) * covariance_matrix_diagonal)
    
    # připravení polí pro výpočty momentů (přesněji: střední hodnoty a variance)
    mv = np.zeros(len(means))
    variance = np.copy(mv)
    nk = np.copy(mv)

    # nalezení stabilních indexů (kde je korekční faktor malý)
    stable = np.where(alf <= 3)[1]
    
    # výpočet pro nestabilní indexy
    for index in stable:
        al = alf[0,index]
        pom = (1 - erf(al)) * np.sqrt(np.pi / 2)
        gam = (-np.exp(-al ** 2)) / pom
        delt = (- lower_bound * np.exp(-al ** 2)) / pom

        covariance = covariance_matrix_diagonal[index]
        mv[index] = means[index] - covariance * gam
        variance[index] = covariance ** 2 + means[index] * mv[index] - covariance * delt
        nk[index] = covariance * pom

    # nalezení stabilních indexů (kde je korekční faktor malý)
    unstable = np.where(alf > 3)[1]
    
    # výpočet pro nestabilní indexy
    for index in unstable:
        ma = means[index,0]
        sa = covariance_matrix_diagonal[index]

        mv[index] = lower_bound - sa ** 2 / ma
        variance[index] = lower_bound ** 2 - (2 * lower_bound * sa ** 2) / ma + (
                2 * sa ** 4 - 2 * lower_bound ** 2 * sa ** 2) / ma ** 2
        nk[index] = -(sa ** 2) / ma * np.exp(-(lower_bound - ma) ** 2 / (2 * sa ** 2))

    # detekce chyby: 
    fault = np.where(mv < lower_bound)[0]
    if fault.size > 0:
        print('tN:(mv<a)')

    return mv.T.astype(np.double), variance
# ---

# ------------


def bayesRegression(X,y,iterCount=101):
    """
    Základní model Bayesovské lineární regrese
    """
    c0 = 1e-10
    d0 = 1e-10
    omegaHat = 1/np.ndarray.max(X.T @ X)
    hist_omega = []

    n = X.shape[1]
    p = X.shape[0]
    for i in range(1,iterCount):
        # update beta
        sigmaBeta = np.linalg.inv(omegaHat * X.T @ X  + np.eye(n))
        muBeta = sigmaBeta @ (omegaHat * X.T @ y)
        betaHat = muBeta
        betaSMHat = muBeta @ muBeta.T + sigmaBeta

        # update omega
        c = c0 + p/2
        d = d0 + (1/2) * ((y.T @ y) - 2 * (y.T @ X @ betaHat) + (np.trace(betaSMHat @ X.T @ X)))
        omegaHat = c/d
        hist_omega = np.append(hist_omega,omegaHat)
    return betaHat, sigmaBeta, hist_omega

def sparseBayesRegression(X,y,iterCount=101):
    """
    Model s diagonální kovarianční maticí V - odpovídá řídkosti řešení (sparsity)
    """
    c0 = 1e-10
    d0 = 1e-10
    a0 = 1e-10
    b0 = 1e-10
    omegaHat = 1/np.ndarray.max(X.T @ X)
    hist_omega = []

    n = X.shape[1]
    p = X.shape[0]

    V = np.diag(np.ones(n))

    for i in range(1,iterCount):
        # update beta
        sigmaBeta = np.linalg.inv(omegaHat * X.T @ X  + V)
        muBeta = sigmaBeta @ (omegaHat * X.T @ y)
        betaHat = muBeta
        betaSMHat = muBeta @ muBeta.T + sigmaBeta

        # update V
        for j in range(0,len(muBeta)): 
            aj = a0 + 1/2
            bj = b0 + 1/2 * betaSMHat[j,j]
            V[j,j] = aj/bj

        # update omega
        c = c0 + p/2
        d = d0 + (1/2) * ((y.T @ y) - 2 * (y.T @ X @ betaHat) + (np.trace(betaSMHat @ X.T @ X)))
        omegaHat = c/d
        hist_omega = np.append(hist_omega,omegaHat)
    return betaHat, sigmaBeta, hist_omega

def positiveSparseBayesRegression(X,y,iterCount=101):
    """
    Model s diagonální kovarianční maticí V, kde betaTrue odpovídá truncated normal distribution
    """
    c0 = 1e-10
    d0 = 1e-10
    a0 = 1e-10
    b0 = 1e-10
    omegaHat = 1/np.ndarray.max(X.T @ X)
    hist_omega = []

    n = X.shape[1]
    p = X.shape[0]

    V = np.diag(np.ones(n))

    for i in range(1,iterCount):
        # update beta
        sigmaBetaPrep = np.linalg.inv(omegaHat * X.T @ X  + V)
        muBetaPrep = sigmaBetaPrep @ (omegaHat * X.T @ y)
        betaHat, b_b_estimate = low_truncated_normal_distribution(
                means=muBetaPrep,
                covariance_matrix_diagonal=np.sqrt(sigmaBetaPrep.diagonal()),
                lower_bound=0
            )
        betaHat = betaHat.reshape(-1,1)
        b_b_estimate = b_b_estimate.reshape(-1,1)
        bbt_variance = b_b_estimate - betaHat ** 2  # vector[N]
        betaSMHat = np.outer(betaHat, betaHat) + np.diag(bbt_variance.reshape(-1))  # matrix[N,N]
        sigmaBetaDiag = bbt_variance
        
        # update V
        for j in range(0,len(betaHat)): 
            aj = a0 + 1/2
            bj = b0 + 1/2 * betaSMHat[j,j]
            V[j,j] = aj/bj
            
        # update omega
        c = c0 + p/2
        d = d0 + (1/2) * ((y.T @ y) - 2 * (y.T @ X @ betaHat) + (np.trace(betaSMHat @ X.T @ X)))
        omegaHat = c/d
        hist_omega = np.append(hist_omega,omegaHat)
    return betaHat, sigmaBetaDiag, hist_omega


def bayesLVLRegression(X, y, l0, iterCount=101):
    """
    Regrese se strukturou kovarianční matice odpovídající LVL^T, kde L má parametry přímo pod diagonálou (a na diagonále 1) a V je diagonální matice (s parametry na diagonále)
    
    Vychází z LS-APC
    """
    c0 = 1e-10
    d0 = 1e-10
    a0 = 1e-10
    b0 = 1e-10
    
    g0 = 1e-2
    h0 = 1e-2
    
    omegaHat = 1/np.ndarray.max(X.T @ X)
    hist_omega = []
    
    n = X.shape[1]
    p = X.shape[0]
    
    V = np.diag(np.ones(n))
    L = np.diag(np.ones(n))
    lowerDiagonalL = l0 * np.diag(np.ones(n-1))
    L[1:,:n-1] += lowerDiagonalL
    tmpLULt = np.diag(np.zeros(n))
    
    psi = np.ones(n-1)
    mu_lj = np.zeros(n-1)
    sigma_lj = np.zeros(n-1)
    
    for i in range(1,iterCount):
        # update beta
        sigmaBeta = np.linalg.inv(omegaHat * X.T @ X  + L @ V @ L.T + tmpLULt)
        muBeta = sigmaBeta @ (omegaHat * X.T @ y)
        betaHat = muBeta
        betaSMHat = np.outer(betaHat, betaHat) + sigmaBeta
        
        # pomocné výpočty pro následující updaty
        tmpPrep = np.zeros((n, n))
        tmpPrep[:n-1,:n-1] = np.multiply(np.diag(sigma_lj), np.diag(betaSMHat[1:,1:]))
        bPrep = np.diag(L.T @ betaSMHat @ L + tmpPrep)

        # update V
        for j in range(0,len(betaHat)): 
            aj = a0 + 1/2
            bj = b0 + 1/2 * bPrep[j]
            V[j,j] = aj/bj
    
        for j in range(0,len(betaHat)-1): 
            # update l_j
            sigma_lj[j] = 1/(V[j,j] * betaSMHat[j+1,j+1] + psi[j])
            mu_lj[j] = sigma_lj[j] * (-V[j,j] * betaSMHat[j,j+1] + l0 * psi[j])
            L[j+1,j] = mu_lj[j]
    
            # update psi_j
            gj = g0 + 1/2
            hj = h0 + 1/2 * (mu_lj[j] * mu_lj[j] + sigma_lj[j] - 2*l0 * mu_lj[j] + l0**2)
            psi[j] = gj/hj
    
        tmpLULt = np.zeros((n, n))
        tmpLULt[1:,1:] = np.multiply(np.diag(sigma_lj), V[:n-1,:n-1])
    
        # update omega
        c = c0 + p/2
        d = d0 + (1/2) * ((y.T @ y) - 2 * (y.T @ X @ betaHat) + (np.trace(betaSMHat @ X.T @ X)))
        omegaHat = c/d
        hist_omega = np.append(hist_omega,omegaHat)
    return betaHat, sigmaBeta, hist_omega

def smooth(betaHat, sigmaOld, newSigma):
    """
    Hladící funkce pro odhad druhého momentu u pozitivních momentů
    (Přesněji:z diagonální kovarianční matice udělá kovarianční matici, která není čistě diagonální)
    
    Odhad kovariance pomocí ořezaného normálního rozdělení vrátí pouze variance (diagonálu kovarianční matice), tj. newSigma
    Tato funkce představuje transformaci s využitím kovarianční matice normálního rozdělení, tj. sigmaOld
    """
    multiplicationFactor = np.diag(np.sqrt(np.divide(newSigma, np.diag(sigmaOld))))
    new_si_x = multiplicationFactor @ sigmaOld @ multiplicationFactor
    return np.outer(betaHat, betaHat) + new_si_x

def bayesLVLPositiveRegression(X,y,l0,to_smooth=False, iterCount=101):
    """
    Pozitivní varianta bayesLVLRegression
    """
    c0 = 1e-10
    d0 = 1e-10
    a0 = 1e-10
    b0 = 1e-10
    
    g0 = 1e-2
    h0 = 1e-2
    
    omegaHat = 1/np.ndarray.max(X.T @ X)
    hist_omega = []
    
    n = X.shape[1]
    p = X.shape[0]
    
    V = np.diag(np.ones(n))
    L = np.diag(np.ones(n))
    lowerDiagonalL = l0 * np.diag(np.ones(n-1))
    L[1:,:n-1] += lowerDiagonalL
    tmpLULt = np.diag(np.zeros(n))
    
    psi = np.ones(n-1)
    mu_lj = np.zeros(n-1)
    sigma_lj = np.zeros(n-1)
    
    for i in range(1,iterCount):
        # update beta
        sigmaBetaPrep = np.linalg.inv(omegaHat * X.T @ X  + L @ V @ L.T + tmpLULt)
        muBetaPrep = sigmaBetaPrep @ (omegaHat * X.T @ y)
    
        betaHat, b_b_estimate = low_truncated_normal_distribution(
                means=muBetaPrep,
                covariance_matrix_diagonal=np.sqrt(sigmaBetaPrep.diagonal()),
                lower_bound=0
            )
        betaHat = betaHat.reshape(-1,1)
        b_b_estimate = b_b_estimate.reshape(-1,1)
        sigmaBeta = b_b_estimate - betaHat ** 2
        if to_smooth:
            betaSMHat = smooth(betaHat,sigmaBetaPrep,sigmaBeta.reshape(-1))
        else:
            betaSMHat = np.outer(betaHat,betaHat) + np.diag(sigmaBeta.reshape(-1))
     
        tmpPrep = np.zeros((n, n))
        tmpPrep[:n-1,:n-1] = np.multiply(np.diag(sigma_lj), np.diag(betaSMHat[1:,1:]))
        bPrep = np.diag(L.T @ betaSMHat @ L + tmpPrep)
    
        # update V
        for j in range(0,len(betaHat)): 
            aj = a0 + 1/2
            bj = b0 + 1/2 * bPrep[j]
            V[j,j] = aj/bj
    
        for j in range(0,len(betaHat)-1): 
            # update l_j
            sigma_lj[j] = 1/(V[j,j] * betaSMHat[j+1,j+1] + psi[j])
            mu_lj[j] = sigma_lj[j] * (-V[j,j] * betaSMHat[j,j+1] + l0 * psi[j])
            L[j+1,j] = mu_lj[j]
            
            # update psi_j
            gj = g0 + 1/2
            hj = h0 + 1/2 * (mu_lj[j] * mu_lj[j] + sigma_lj[j] - 2*l0 * mu_lj[j] + l0**2)
            psi[j] = gj/hj
    
        tmpLULt = np.zeros((n, n))
        tmpLULt[1:,1:] = np.multiply(np.diag(sigma_lj), V[:n-1,:n-1])
    
        # update omega
        c = c0 + p/2
        d = d0 + (1/2) * ((y.T @ y) - 2 * (y.T @ X @ betaHat) + (np.trace(betaSMHat @ X.T @ X)))
        omegaHat = c/d
        hist_omega = np.append(hist_omega,omegaHat)
    return betaHat, sigmaBeta, hist_omega


# Pomocné funkce pro výpočet modelu s obecnou kovarianční strukturou
# ---

def padTopLeft(matIn, shapeToPadTo):
    startInd = matIn.shape[0]
    newMat = np.zeros((shapeToPadTo,shapeToPadTo))
    newMat[-startInd:,-startInd:] = matIn
    return newMat

def trimZerosMat(matIn, indexMat):
    nonZeroRows = np.any(indexMat != 0, axis=1)
    nonZeroCols = np.any(indexMat != 0, axis=0)
    return matIn[np.ix_(nonZeroRows,nonZeroCols)]

def trimZerosMatAndPsi(matIn, psiDiagIn, indexMat):
    nonZeroRows = np.any(indexMat != 0, axis=1)
    nonZeroCols = np.any(indexMat != 0, axis=0)
    return trimZerosMat(matIn,indexMat),trimZerosMat(psiDiagIn,indexMat)

def expandTrimmedMat(matIn,indexMat):
    outMat = np.zeros_like(indexMat)
    nonZeroRows = np.any(indexMat != 0, axis=1)
    nonZeroCols = np.any(indexMat != 0, axis=0)
    outMat[np.ix_(nonZeroRows,nonZeroCols)] = matIn
    return outMat
# ---

def bayesGeneralLVLRegression(X, y, l0, activeL, iterCount = 101):
    """
    Zobecněná varianta bayesLVLRegression, kde matice L je dolní trojúhelníková matice s 1 na diagonálami. Zapnutí parametrů lze určit přes strukturu activeL
    """
    c0 = 1e-10
    d0 = 1e-10
    a0 = 1e-10
    b0 = 1e-10
    
    g0 = 1e-2
    h0 = 1e-2
    
    omegaHat = 1/np.ndarray.max(X.T @ X)
    hist_omega = []
    
    n = X.shape[1]
    p = X.shape[0]
    
    V = np.diag(np.ones(n))
    psi_j = np.diag(np.zeros(n))
    
    indexL = np.diag(np.zeros(n))
    listOfLIndices = []
    for col in range(0,len(activeL)):
        for row in activeL[col]:
            indexL[row,col] = 1
            listOfLIndices.append((col,row))
    psi_j = indexL.copy() # tohle se mění na psi_j, indexace odpovídá mu_lj

    # zde již proběhla změna
    L = np.diag(np.ones(n)) + indexL * l0
    mu_lj = indexL * l0 # <=> L2 = L bez jedniček na diagonále
    arr_sigma_lj = [np.diag(np.zeros(n)) for j in range(n-1)] # sigma_l_j se nachází na indexu j a její matice je indexech od [j+1,j+1] do [n,n]

    tmpLVLt = np.diag(np.zeros(n))
    
    for i in range(1,iterCount):
        # update beta
        sigmaBeta = np.linalg.inv(omegaHat * X.T @ X + L @ V @ L.T + tmpLVLt)
        muBeta = sigmaBeta @ (omegaHat * X.T @ y)
        betaHat = muBeta
        betaSMHat = np.outer(betaHat, betaHat) + sigmaBeta

        # update V
        for j in range(0,n-1):
            aj = a0 + 1/2
            L_jL_jT = (np.matrix(L[j:,j]).T @ np.matrix(L[j:,j])) + padTopLeft(arr_sigma_lj[j][j+1:,j+1:], n-j)
            bj = b0 + 1/2 * np.sum(np.multiply(L_jL_jT, betaSMHat[j:,j:]))
            V[j,j] = aj/bj
        j = n-1
        an = a0 + 1/2
        L_nL_nT = np.matrix(L[j:,j]).T @ np.matrix(L[j:,j])
        bn = b0 + 1/2 * np.sum(np.multiply(L_nL_nT, betaSMHat[j:,j:]))
        V[j,j] = an/bn

        for col in range(0,n-1):
            # update l_j 
            indexSelect = indexL[col:,col].reshape(-1,1) @ indexL[col:,col].reshape(-1,1).T
            betaSMHatIndex_j, psi_jIndexed = trimZerosMatAndPsi(betaSMHat[col:,col:], np.diag(psi_j[col:,col]), indexSelect)
            if betaSMHatIndex_j.shape[0] == 0:
                continue
            sigma_lj = np.linalg.inv(psi_jIndexed + betaSMHatIndex_j * V[col,col])
            arr_sigma_lj[col] = padTopLeft(expandTrimmedMat(sigma_lj, indexSelect),n) # změna provedena
            
            posPartMu = l0 * (psi_j[col+1:,col].reshape(-1,1))
            negPartMu = V[col,col] * np.multiply(betaSMHat[col+1:,col], indexL[col+1:,col]).reshape(-1,1)
            mu_lj[col+1:, col] = (arr_sigma_lj[col][col+1:,col+1:]@(posPartMu - negPartMu)).reshape(-1) # změna provedena
            L[col+1:,col] = mu_lj[col+1:, col]

            # update psi_j
            j = col
            gj = g0 + (1/2) * np.ones((n-j,1))
            L_jL_jT = (np.matrix(L[j:,j]).T @ np.matrix(L[j:,j])) + padTopLeft(arr_sigma_lj[j][j+1:,j+1:], n-j)
            
            hj = h0 + (1/2) * (np.diag(L_jL_jT).reshape(-1,1) - 2 * l0 * mu_lj[col:,col].reshape(-1,1) + l0**2 * np.ones((n-j,1)))
            psi_j[col:,col] = np.divide(gj,hj).reshape(-1) * indexL[col:,col]
    
        # změna výpočtu, aby odpovídal závislosti v rámci lj
        tmpLVLt = np.sum([V[j,j] * arr_sigma_lj[j] for j in range(n-1)],0)
    
        # update omega
        c = c0 + p/2
        d = d0 + (1/2) * ((y.T @ y) - 2 * (y.T @ X @ betaHat) + (np.trace(betaSMHat @ X.T @ X)))
        omegaHat = c/d
        hist_omega = np.append(hist_omega,omegaHat)
    return betaHat, sigmaBeta, hist_omega

def bayesLVLGeneralPositiveRegression(X,y,l0,activeL,to_smooth=False, iterCount = 101):
    """
    Zobecněná varianta bayesLVLRegression, kde matice L je dolní trojúhelníková matice s 1 na diagonálami. Zapnutí parametrů lze určit přes strukturu activeL
    """
    c0 = 1e-10
    d0 = 1e-10
    a0 = 1e-10
    b0 = 1e-10
    
    g0 = 1e-2
    h0 = 1e-2
    
    omegaHat = 1/np.ndarray.max(X.T @ X)
    hist_omega = []
    
    n = X.shape[1]
    p = X.shape[0]
    
    V = np.diag(np.ones(n))
    psi_j = np.diag(np.zeros(n))
    
    indexL = np.diag(np.zeros(n))
    listOfLIndices = []
    for col in range(0,len(activeL)):
        for row in activeL[col]:
            indexL[row,col] = 1
            listOfLIndices.append((col,row))
    psi_j = indexL.copy() # tohle se mění na psi_j, indexace odpovídá mu_lj

    # zde již proběhla změna
    L = np.diag(np.ones(n)) + indexL * l0
    mu_lj = indexL * l0 # <=> L2 = L bez jedniček na diagonále
    arr_sigma_lj = [np.diag(np.zeros(n)) for j in range(n-1)] # sigma_l_j se nachází na indexu j a její matice je indexech od [j+1,j+1] do [n,n]

    tmpLVLt = np.diag(np.zeros(n))
    
    for i in range(1,iterCount):
        # update beta
        sigmaBetaPrep = np.linalg.inv(omegaHat * X.T @ X  + L @ V @ L.T + tmpLVLt)
        muBetaPrep = sigmaBetaPrep @ (omegaHat * X.T @ y)
        betaHat, b_b_estimate = low_truncated_normal_distribution(
                means=muBetaPrep,
                covariance_matrix_diagonal=np.sqrt(sigmaBetaPrep.diagonal()),
                lower_bound=0
            )
        betaHat = betaHat.reshape(-1,1)
        b_b_estimate = b_b_estimate.reshape(-1,1)
        sigmaBeta = b_b_estimate - betaHat ** 2
        betaSMHat = None
        
        if to_smooth:
            betaSMHat = smooth(betaHat,sigmaBetaPrep,sigmaBeta.reshape(-1))
        else:
            betaSMHat = np.outer(betaHat,betaHat) + np.diag(sigmaBeta.reshape(-1))

        # update V
        for j in range(0,n-1):
            aj = a0 + 1/2
            L_jL_jT = (np.matrix(L[j:,j]).T @ np.matrix(L[j:,j])) + padTopLeft(arr_sigma_lj[j][j+1:,j+1:], n-j)
            bj = b0 + 1/2 * np.sum(np.multiply(L_jL_jT, betaSMHat[j:,j:]))
            V[j,j] = aj/bj
        j = n-1
        an = a0 + 1/2
        L_nL_nT = np.matrix(L[j:,j]).T @ np.matrix(L[j:,j])
        bn = b0 + 1/2 * np.sum(np.multiply(L_nL_nT, betaSMHat[j:,j:]))
        V[j,j] = an/bn

        for col in range(0,n-1):
            # update l_j 
            indexSelect = indexL[col:,col].reshape(-1,1) @ indexL[col:,col].reshape(-1,1).T
            betaSMHatIndex_j, psi_jIndexed = trimZerosMatAndPsi(betaSMHat[col:,col:], np.diag(psi_j[col:,col]), indexSelect)
            if betaSMHatIndex_j.shape[0] == 0:
                continue
            sigma_lj = np.linalg.inv(psi_jIndexed + betaSMHatIndex_j * V[col,col])
            arr_sigma_lj[col] = padTopLeft(expandTrimmedMat(sigma_lj, indexSelect),n) # změna provedena
            
            posPartMu = l0 * (psi_j[col+1:,col].reshape(-1,1))
            negPartMu = V[col,col] * np.multiply(betaSMHat[col+1:,col], indexL[col+1:,col]).reshape(-1,1)
            mu_lj[col+1:, col] = (arr_sigma_lj[col][col+1:,col+1:]@(posPartMu - negPartMu)).reshape(-1) # změna provedena
            L[col+1:,col] = mu_lj[col+1:, col]

            # update psi_j
            j = col
            gj = g0 + (1/2) * np.ones((n-j,1))
            L_jL_jT = (np.matrix(L[j:,j]).T @ np.matrix(L[j:,j])) + padTopLeft(arr_sigma_lj[j][j+1:,j+1:], n-j)
            
            hj = h0 + (1/2) * (np.diag(L_jL_jT).reshape(-1,1) - 2 * l0 * mu_lj[col:,col].reshape(-1,1) + l0**2 * np.ones((n-j,1)))
            psi_j[col:,col] = np.divide(gj,hj).reshape(-1) * indexL[col:,col]
    
        # změna výpočtu, aby odpovídal závislosti v rámci lj
        tmpLVLt = np.sum([V[j,j] * arr_sigma_lj[j] for j in range(n-1)],0)
    
        # update omega
        c = c0 + p/2
        d = d0 + (1/2) * ((y.T @ y) - 2 * (y.T @ X @ betaHat) + (np.trace(betaSMHat @ X.T @ X)))
        omegaHat = c/d
        hist_omega = np.append(hist_omega,omegaHat)
    return betaHat, sigmaBeta, hist_omega

def sparseBayesRegressionWithBeta(X,y,betaIn,iterCount=101):
    """
    Model s diagonální kovarianční maticí V
    
    Apriorní rozdělení beta ~ N(betaIn,V)
    """
    c0 = 1e-10
    d0 = 1e-10
    a0 = 1e-10
    b0 = 1e-10
    omegaHat = 1/np.ndarray.max(X.T @ X)
    hist_omega = []

    n = X.shape[1]
    p = X.shape[0]

    V = np.diag(np.ones(n))
    
    for i in range(1,iterCount):
        # update beta
        sigmaBeta = np.linalg.inv(omegaHat * X.T @ X  + V)
        muBeta = sigmaBeta @ (V @ betaIn + omegaHat * X.T @ y) # zde změna -> V @ betaIn
        betaHat = muBeta
        betaSMHat = muBeta @ muBeta.T + sigmaBeta

        # update V
        for j in range(0,len(muBeta)): 
            aj = a0 + 1/2
            # zde změna 1/2 (betaIn @ betaIn.T)[j,j] - (betaIn @ muBeta.T)[j,j]
            bjInner = 1/2 * betaSMHat[j,j] - betaIn[j,0] * muBeta[j,0] + 1/2 * betaIn[j,0] * betaIn[j,0]
            bj = b0 + bjInner
            V[j,j] = aj/bj
        
        # update omega
        c = c0 + p/2
        d = d0 + (1/2) * ((y.T @ y) - 2 * (y.T @ X @ betaHat) + (np.trace(betaSMHat @ X.T @ X)))
        omegaHat = c/d
        hist_omega = np.append(hist_omega,omegaHat)
    return betaHat, sigmaBeta, hist_omega

def positiveSparseBayesRegressionWithBeta(X,y,betaIn,iterCount=101):
    """
    Model s diagonální kovarianční maticí V, kde betaTrue odpovídá truncated normal distribution
    
    Apriorní rozdělení beta ~ tN(betaIn,V,0,+inf)
    """
    c0 = 1e-10
    d0 = 1e-10
    a0 = 1e-10
    b0 = 1e-10
    omegaHat = 1/np.ndarray.max(X.T @ X)
    hist_omega = []

    n = X.shape[1]
    p = X.shape[0]

    V = np.diag(np.ones(n))
    
    for i in range(1,iterCount):
        # update beta
        sigmaBetaPrep = np.linalg.inv(omegaHat * X.T @ X  + V)
        muBetaPrep = sigmaBetaPrep @ (V @ betaIn + omegaHat * X.T @ y) # zde změna -> V @ betaIn
        betaHat, b_b_estimate = low_truncated_normal_distribution(
                means=muBetaPrep,
                covariance_matrix_diagonal=np.sqrt(sigmaBetaPrep.diagonal()),
                lower_bound=0
            )
        betaHat = betaHat.reshape(-1,1)
        b_b_estimate = b_b_estimate.reshape(-1,1)
        bbt_variance = b_b_estimate - betaHat ** 2  # vector[N]
        betaSMHat = np.outer(betaHat, betaHat) + np.diag(bbt_variance.reshape(-1))  # matrix[N,N]
        sigmaBetaDiag = bbt_variance

        # update V
        for j in range(0,n): 
            aj = a0 + 1/2
            # zde změna 1/2 (betaIn @ betaIn.T)[j,j] - (betaIn @ muBeta.T)[j,j]
            bjInner = 1/2 * betaSMHat[j,j] - betaIn[j,0] * betaHat[j,0] + 1/2 * betaIn[j,0] * betaIn[j,0]
            bj = b0 + bjInner
            V[j,j] = aj/bj
        
        # update omega
        c = c0 + p/2
        d = d0 + (1/2) * ((y.T @ y) - 2 * (y.T @ X @ betaHat) + (np.trace(betaSMHat @ X.T @ X)))
        omegaHat = c/d
        hist_omega = np.append(hist_omega,omegaHat)
    return betaHat,sigmaBetaDiag, hist_omega

def bayesGeneralLVLRegressionWithBeta(X, y, l0, activeL, betaIn, iterCount = 101):
    """
    Zobecněná varianta bayesLVLRegression, kde matice L je dolní trojúhelníková matice s 1 na diagonálami. Zapnutí parametrů lze určit přes strukturu activeL
    
    S beta ~ N(betaIn, (LVL^T)^-1)
    """
    c0 = 1e-10
    d0 = 1e-10
    a0 = 1e-10
    b0 = 1e-10
    
    g0 = 1e-2
    h0 = 1e-2
    
    omegaHat = 1/np.ndarray.max(X.T @ X) # změna z X na XTX
    hist_omega = []
    
    n = X.shape[1]
    p = X.shape[0]
    
    V = np.diag(np.ones(n))
    psi_j = np.diag(np.zeros(n)) # tohle se mění na psi_j, indexace odpovídá mu_lj - zde změna
    
    indexL = np.diag(np.zeros(n))
    listOfLIndices = []
    for col in range(0,len(activeL)):
        for row in activeL[col]:
            indexL[row,col] = 1
            listOfLIndices.append((col,row))
    psi_j = indexL.copy() # tohle se mění na psi_j, indexace odpovídá mu_lj - zde změna

    # zde již proběhla změna
    L = np.diag(np.ones(n)) + indexL * l0
    mu_lj = indexL * l0 # <=> L2 = L bez jedniček na diagonále
    # sigma_{l_j} se nachází na indexu j, matice sigma_{l_j} je připravena pro součet s l_j \cdot l_j^T
    arr_sigma_lj = [np.diag(np.zeros(n)) for j in range(n-1)]
    tmpLVLt = np.diag(np.zeros(n))
    
    for i in range(1,iterCount):        
        # update beta
        sigmaBeta = np.linalg.inv(omegaHat * X.T @ X + L @ V @ L.T + tmpLVLt)
        muBeta = sigmaBeta @ ((L @ V @ L.T + tmpLVLt) @ betaIn + omegaHat * X.T @ y) # betaIn zde
        betaHat = muBeta
        betaSMHat = np.outer(betaHat, betaHat) + sigmaBeta

        # update V
        for j in range(0,n-1):
            aj = a0 + 1/2
            LjLj_tmp = (np.matrix(L[j:,j]).T @ np.matrix(L[j:,j]))
            LjLjSigma_tmp = arr_sigma_lj[j][j:,j:]
            L_jL_jT = LjLj_tmp + LjLjSigma_tmp

            bb = np.sum(np.multiply(L_jL_jT, betaSMHat[j:,j:]/2))
            b0b0 = np.sum(np.multiply(L_jL_jT, (betaIn @ betaIn.T)[j:,j:]/2))
            b0b = - np.sum(np.multiply(L_jL_jT, (betaIn @ betaHat.T)[j:,j:]/2))
            bb0 = - np.sum(np.multiply(L_jL_jT, (betaHat @ betaIn.T)[j:,j:]/2))
            # bb0Andb0b = - np.sum(np.multiply(L_jL_jT, (betaHat @ betaIn.T)[j:,j:])) - alt. verze, která má horší přesnost
            
            longEquation = bb + b0b + bb0 + b0b0 # betaIn zde, 1/2 již před sumou
            
            bj = b0 + longEquation
            V[j,j] = aj/bj
        j = n-1
        an = a0 + 1/2
        L_nL_nT = np.matrix(L[j:,j]).T @ np.matrix(L[j:,j])
        longEquation = (1/2) * np.sum(np.multiply(L_nL_nT, betaSMHat[j:,j:])) - np.sum(np.multiply(L_nL_nT, (betaIn @ betaHat.T)[j:,j:])) + (1/2) * np.sum(np.multiply(L_nL_nT, (betaIn @ betaIn.T)[j:,j:])) 
        bn = b0 + longEquation
        V[j,j] = an/bn
        
        for col in range(0,n-1):        
            # update l_j
            indexSelect = indexL[col:,col].reshape(-1,1) @ indexL[col:,col].reshape(-1,1).T
            betaInIndex_j = trimZerosMat((betaIn @ betaIn.T)[col:,col:], indexSelect) # betaIn
            betaHatBetaInIndex_j = trimZerosMat((betaHat @ betaIn.T)[col:,col:], indexSelect) # betaIn
            betaSMHatIndex_j, psi_jIndexed = trimZerosMatAndPsi(betaSMHat[col:,col:], np.diag(psi_j[col:,col]), indexSelect)
            
            if betaSMHatIndex_j.shape[0] == 0:
                continue
            sigma_lj = np.linalg.inv(psi_jIndexed + (betaSMHatIndex_j - betaHatBetaInIndex_j - betaHatBetaInIndex_j.T + betaInIndex_j) * V[col,col]) # betaIn zde
            
            
            arr_sigma_lj[col] = padTopLeft(expandTrimmedMat(sigma_lj, indexSelect),n)
            
            posPartMu = (l0 * psi_j[col+1:,col]).reshape(-1,1)
            
            
            # reshape hotfix
            betaInTerms = np.multiply((-1)*betaIn[col,0]*betaHat[col+1:,0].reshape(-1,1)+(-1)*betaHat[col,0]*betaIn[col+1:,0].reshape(-1,1)+betaIn[col,0]*betaIn[col+1:,0].reshape(-1,1), indexL[col+1:,col].reshape(-1,1)).reshape(-1,1)
            negPartMu = V[col,col] * (np.multiply(betaSMHat[col+1:,col], indexL[col+1:,col]).reshape(-1,1) + betaInTerms) # mu betaIn

            mu_lj[col+1:, col] = (arr_sigma_lj[col][col+1:,col+1:]@(posPartMu - negPartMu)).reshape(-1) # betaIn zde
            L[col+1:,col] = mu_lj[col+1:, col]

            # update psi_j
            j = col
            gj = g0 + (1/2) * np.ones((n-j-1,1)) # optimalizace přes lin. alg. namísto iterace
            
            # druhý moment l_j (bez prvního sloupce a řádku)
            LjLj_tmp = (np.matrix(L[j:,j]).T @ np.matrix(L[j:,j]))[1:,1:]
            LjLjSigma_tmp = arr_sigma_lj[j][col+1:,col+1:]
            L_jL_jT = LjLj_tmp + LjLjSigma_tmp
            
            hj = h0 + (1/2) * (np.diag(L_jL_jT).reshape(-1,1) - 2 * l0 * mu_lj[col+1:,col].reshape(-1,1) + l0**2 * np.ones((n-j-1,1)))
            psi_j[col+1:,col] = np.divide(gj,hj).reshape(-1) * indexL[col+1:,col]
    
        # změna výpočtu, aby odpovídal závislosti v rámci lj
        tmpLVLt = np.sum([V[j,j] * arr_sigma_lj[j] for j in range(n-1)],0)
    
        # update omega
        c = c0 + p/2
        d = d0 + (1/2) * ((y.T @ y)[0,0] - 2 * (y.T @ X @ betaHat)[0,0] + (np.trace(betaSMHat @ X.T @ X)))
        omegaHat = c/d
        hist_omega = np.append(hist_omega,omegaHat)
    return betaHat, sigmaBeta, hist_omega

# funkční verze final
def bayesGeneralLVLRegressionWithBetaPositive(X, y, l0, activeL, betaIn, iterCount = 101, to_smooth = False):
    """
    Zobecněná varianta bayesLVLRegression, kde matice L je dolní trojúhelníková matice s 1 na diagonálami. Zapnutí parametrů lze určit přes strukturu activeL
    
    S beta ~ N(betaIn, (LVL^T)^-1)
    """
    betaIn = betaIn.astype("float64")
    
    c0 = 1e-10
    d0 = 1e-10
    a0 = 1e-10
    b0 = 1e-10
    
    g0 = 1e-2
    h0 = 1e-2
    
    omegaHat = 1/np.ndarray.max(X.T @ X) # změna z X na XTX
    hist_omega = []
    
    n = X.shape[1]
    p = X.shape[0]
    
    V = np.diag(np.ones(n))
    psi_j = np.diag(np.ones(n)) # tohle se mění na psi_j, indexace odpovídá mu_lj - zde změna
    
    indexL = np.diag(np.zeros(n))
    listOfLIndices = []
    for col in range(0,len(activeL)):
        for row in activeL[col]:
            indexL[row,col] = 1
            listOfLIndices.append((col,row))
    psi_j = indexL.copy() # tohle se mění na psi_j, indexace odpovídá mu_lj - zde změna

    # zde již proběhla změna
    L = np.diag(np.ones(n)) + indexL * l0
    mu_lj = indexL * l0 # <=> L2 = L bez jedniček na diagonále
    # sigma_{l_j} se nachází na indexu j, matice sigma_{l_j} je připravena pro součet s l_j \cdot l_j^T
    arr_sigma_lj = [np.diag(np.zeros(n)) for j in range(n-1)]
    tmpLVLt = np.diag(np.zeros(n))
    
    
    for i in range(1,iterCount):
        # update beta
        sigmaBetaPrep = np.linalg.inv(omegaHat * X.T @ X + (L @ V @ L.T + tmpLVLt))
        muBetaPrep = sigmaBetaPrep @ ((L @ V @ L.T + tmpLVLt) @ betaIn + omegaHat * X.T @ y) # betaIn zde
        betaHat, b_b_estimate = low_truncated_normal_distribution(
                means=muBetaPrep,
                covariance_matrix_diagonal=np.sqrt(sigmaBetaPrep.diagonal()),
                lower_bound=0
            )
        betaHat = betaHat.reshape(-1,1)
        b_b_estimate = b_b_estimate.reshape(-1,1)
        sigmaBeta = b_b_estimate - betaHat ** 2
        betaSMHat = None
        
        if to_smooth:
            betaSMHat = smooth(betaHat,sigmaBetaPrep,sigmaBeta.reshape(-1))
        else:
            betaSMHat = np.outer(betaHat,betaHat) + np.diag(sigmaBeta.reshape(-1))
        
        # update V
        for j in range(0,n-1):
            aj = a0 + 1/2
            LjLj_tmp = (np.matrix(L[j:,j]).T @ np.matrix(L[j:,j]))
            LjLjSigma_tmp = arr_sigma_lj[j][j:,j:]
            L_jL_jT = LjLj_tmp + LjLjSigma_tmp

            bb = np.sum(np.multiply(L_jL_jT, betaSMHat[j:,j:]/2))
            b0b0 = np.sum(np.multiply(L_jL_jT, (betaIn @ betaIn.T)[j:,j:]/2))
            b0b = - np.sum(np.multiply(L_jL_jT, (betaIn @ betaHat.T)[j:,j:]/2))
            bb0 = - np.sum(np.multiply(L_jL_jT, (betaHat @ betaIn.T)[j:,j:]/2))
            
            longEquation = bb + b0b + bb0 + b0b0 # betaIn zde, 1/2 již před sumou
            
            bj = b0 + longEquation
            V[j,j] = aj/bj
        j = n-1
        an = a0 + 1/2
        L_nL_nT = np.matrix(L[j:,j]).T @ np.matrix(L[j:,j]) # zde to nevadí
        bb = np.sum(np.multiply(L_nL_nT, betaSMHat[j:,j:]/2))
        b0b0 = np.sum(np.multiply(L_nL_nT, (betaIn @ betaIn.T)[j:,j:]/2))
        b0b = - np.sum(np.multiply(L_nL_nT, (betaIn @ betaHat.T)[j:,j:]/2))
        bb0 = - np.sum(np.multiply(L_nL_nT, (betaHat @ betaIn.T)[j:,j:]/2))
        longEquation = bb + b0b + bb0 + b0b0 # betaIn zde, 1/2 již před sumou
        
        bn = b0 + longEquation
        V[j,j] = an/bn
        
        for col in range(0,n-1):
            # update l_j
            indexSelect = indexL[col:,col].reshape(-1,1) @ indexL[col:,col].reshape(-1,1).T
            # indexL má 0 na diagonále -> zbydou jen čisté křížové a kvadratické členy
            betaInIndex_j = trimZerosMat((betaIn @ betaIn.T)[col:,col:], indexSelect) # betaIn
            betaHatBetaInIndex_j = trimZerosMat((betaHat @ betaIn.T)[col:,col:], indexSelect) # betaIn
            betaInBetaHatIndex_j = trimZerosMat((betaIn @ betaHat.T)[col:,col:], indexSelect) # betaIn
            betaSMHatIndex_j, psi_jIndexed = trimZerosMatAndPsi(betaSMHat[col:,col:], np.diag(psi_j[col:,col]), indexSelect)
            
            if betaSMHatIndex_j.shape[0] == 0:
                continue
            sigma_lj = np.linalg.inv(psi_jIndexed + (betaSMHatIndex_j - betaHatBetaInIndex_j - betaInBetaHatIndex_j + betaInIndex_j) * V[col,col]) # betaIn zde
            
            arr_sigma_lj[col] = padTopLeft(expandTrimmedMat(sigma_lj, indexSelect),n)
            
            posPartMu = (l0 * psi_j[col+1:,col]).reshape(-1,1)
            
            # reshape hotfix
            betaInTerms = np.multiply((-1)*betaIn[col,0]*betaHat[col+1:,0].reshape(-1,1)+(-1)*betaHat[col,0]*betaIn[col+1:,0].reshape(-1,1)+betaIn[col,0]*betaIn[col+1:,0].reshape(-1,1), indexL[col+1:,col].reshape(-1,1)).reshape(-1,1)
            negPartMu = V[col,col] * (np.multiply(betaSMHat[col+1:,col], indexL[col+1:,col]).reshape(-1,1) + betaInTerms) # mu betaIn

            mu_lj[col+1:, col] = (arr_sigma_lj[col][col+1:,col+1:]@(posPartMu - negPartMu)).reshape(-1) # betaIn zde
            L[col+1:,col] = mu_lj[col+1:, col].copy()

            # update psi_j
            j = col

            # verze přes lin. alg. namísto iterace
            gj = g0 + (1/2) * np.ones((n-j-1,1))
            
            # druhý moment l_j (bez prvního sloupce a řádku)
            LjLj_tmp = (np.matrix(L[j:,j]).T @ np.matrix(L[j:,j]))[1:,1:]
            LjLjSigma_tmp = arr_sigma_lj[j][col+1:,col+1:]
            L_jL_jT = LjLj_tmp + LjLjSigma_tmp
            
            hj = h0 + (1/2) * (np.diag(L_jL_jT).reshape(-1,1) - 2 * l0 * mu_lj[col+1:,col].reshape(-1,1) + l0**2 * np.ones((n-j-1,1)))
            psi_j[col+1:,col] = np.divide(gj,hj).reshape(-1) * indexL[col+1:,col]
    
        # změna výpočtu, aby odpovídal závislosti v rámci lj
        tmpLVLt = np.sum([V[j,j] * arr_sigma_lj[j] for j in range(n-1)],0)
    
        # update omega
        c = c0 + p/2
        d = d0 + (1/2) * ((y.T @ y)[0,0] - 2 * (y.T @ X @ betaHat)[0,0] + (np.trace(betaSMHat @ X.T @ X)))
        omegaHat = c/d
        hist_omega = np.append(hist_omega,omegaHat)
    return betaHat, sigmaBeta, hist_omega

# -------

# Rozhraní pro klasické metody, aby odpovídalo metodám VB

def LinearRegressionGetBeta(X, y,iterCount=None, **kwargs):
    linearRegressionModel = LinearRegression(fit_intercept=False, **kwargs).fit(X, y)
    betaHatLR = linearRegressionModel.coef_.reshape(-1,1)
    return betaHatLR, None, None

def RidgeGetBeta(X,y,alphasRange,iterCount=None, **kwargs):
    ridgeRegressionModelCrossValidation = RidgeCV(alphas = alphasRange, fit_intercept=False, **kwargs).fit(X, y.reshape(-1))
    betaHatRidge = ridgeRegressionModelCrossValidation.coef_.reshape(-1,1)
    return betaHatRidge, None, None

def LASSOGetBeta(X,y,alphasRange,iterCount=None, **kwargs):
    lassoModelCrossValidation = LassoCV(alphas = alphasRange, fit_intercept=False, **kwargs).fit(X, y.reshape(-1))
    betaHatLASSO = lassoModelCrossValidation.coef_.reshape(-1,1)
    return betaHatLASSO, None, None
