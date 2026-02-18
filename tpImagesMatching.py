import numpy as np
import cv2
import os
from skimage.feature import local_binary_pattern
from scipy.ndimage import uniform_filter
from PIL import Image

DOSSIER_SCRIPT = os.path.dirname(os.path.abspath(__file__))


######################## Exercice 1 : 

def glcm(window, nb_grays) :
    """
    Cette fonction calcule la matrice de co-occurrence GLCM
    pour une fenêtre d'image.

    Pour calculer cette matrice, on utilise un offset horizontal.
    On va parcourir la fenêtre horizontalement et pour chaque pixel actuel,
    on va regarder sont pixel voisin.
    
    :param window: fenêtre d'image
    :param nb_grays: nombre de niveaux de gris
    """

    glcm_matrix = np.zeros((nb_grays, nb_grays), dtype=int)

    height, width = window.shape

    #on choisit un offset horizontal
    for y in range(height) :
        for x in range(width-1) : #-1 car offset horizontal sur b
            a = window[y,x]
            b = window[y,x+1] #on regarde le voisin horizontal

            if 0<=a<nb_grays and 0<=b<nb_grays :
                glcm_matrix[a,b] += 1
    
    return glcm_matrix


def glcm_variance(glcm_mat) :
    """
    Cette fonction a pour but de calculer la variance
    de la matrice glcm.
    
    :param glcm_mat: matrice de co-occurrence faite par la fonction glcm
    """
    
    t = glcm_mat.sum()
    if t==0 : 
        return 0
    
    p = glcm_mat/t #matrice de probabilités P provenant des descripteurs d'Haralick

    i,j = np.indices(p.shape) #récupère les indices i et j de la matrice p

    moy = np.sum(i*p) #calcule la moyenne

    return np.sum(((i-moy)**2)*p)


def glcm_contrast(glcm_mat) :
    """
    Cette fonction a pour but de calculer le contraste
    de la matrice glcm.
    
    :param glcm_mat: matrice de co-occurrence faite par la fonction glcm
    """

    t = glcm_mat.sum()
    if t==0 : 
        return 0
    
    p = glcm_mat/t #matrice de probabilités P provenant des descripteurs d'Haralick

    i,j = np.indices(p.shape) #récupère les indices i et j de la matrice p

    return np.sum(((i-j)**2)*p)


#calculer mat de co-occurrence pour chaque partie de l'image
#et ensuite, pourra calculer un seuil avec qui permettra de faire ressortir objets de l'image

def glcm_entropy(glcm_mat) :
    """
    Cette fonction a pour but de calculer l'entropie
    de la matrice glcm.
    
    :param glcm_mat: matrice de co-occurrence faite par la fonction glcm
    """

    t = glcm_mat.sum()
    if t==0 : 
        return 0
    
    p = glcm_mat/t #matrice de probabilités P provenant des descripteurs d'Haralick

    return -np.sum(p[p>0] * np.log2(p[p>0]))



#calculer mat de co-occurrence pour chaque partie de l'image
#et ensuite, pourra calculer un seuil avec qui permettra de faire ressortir objets de l'image


def image_feature(image, window_size, nb_grays) :

    """
    Cette fonction calcule les cartes de features (variance, contraste et entropie)
    pour image en la scannant avec une fenêtre glissante d'une taille à définir.
    
    :param image: image à scanner
    :param window_size: taille de la fenêtre glissante
    :param nb_grays: nombre de niveaux de gris
    """

    height, width = image.shape
    half_window = window_size//2
    #feature_mat = np.zeros((height, width), dtype=float)
    #feature_mat = (image * (nb_grays - 1)).astype(int)
    feature_mat = np.pad(image, half_window, mode='constant', constant_values=0) #pour gérer les bords


    var_map = np.zeros((height, width))
    cont_map = np.zeros((height, width))
    ent_map = np.zeros((height, width))

    #on calcule la glcm sur chaque partie de l'image
    for y in range(half_window, height - half_window):
        for x in range(half_window, width - half_window):
            window = feature_mat[y-half_window : x+half_window+1, x-half_window : x+half_window+1]
            glcm_mat = glcm(window, nb_grays)

            #on calcule variance, contraste et entropie
            var = glcm_variance(glcm_mat)
            cont = glcm_contrast(glcm_mat)
            ent = glcm_entropy(glcm_mat)
            var_map[y, x] = var
            cont_map[y, x] = cont
            ent_map[y, x] = ent
    
    return var_map, cont_map, ent_map



def threshold_mask(feature_map, thresh, max_val):
    """
    Normalise et applique un seuillage pour créer un masque à partir de feature_map.
    
    :param feature_map: carte de feature
    :param thresh: seuil (après normalisation 0-255)
    """
    norm_map = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(norm_map, thresh, max_val, cv2.THRESH_BINARY)
    return mask



def safe_imread(path):
    """
    Lit une image TIFF en utilisant PIL pour gérer les formats non supportés par OpenCV.
    """
    img = Image.open(path)
    img = img.convert('L')  # Convertir en grayscale 8-bit
    return np.array(img)


zebra1 = safe_imread(os.path.join(DOSSIER_SCRIPT,"zebra_1.tif"))
zebra2 = safe_imread(os.path.join(DOSSIER_SCRIPT,"zebra_2.tif"))
zebra3 = safe_imread(os.path.join(DOSSIER_SCRIPT,"zebra_3.tif"))

nb_grays = 256

window_size = 5

# Test question 1 :
v_map, c_map, e_map = image_feature(zebra1, window_size, nb_grays)


# Test question 2 : 

cont_thresh = 100  # à ajuster si nécessaire
mask_glcm_cont = threshold_mask(c_map, cont_thresh) #test sur contraste


var_thresh = 50  # à ajuster si nécessaire
mask_glcm_var = threshold_mask(v_map, cont_thresh) #test sur variance


cv2.imshow("Var_map avec zebra 1", mask_glcm_var)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Cont_map avec zebra 1", mask_glcm_cont)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Test question 4 : 

lbp_map = local_binary_pattern(zebra1, P=50, R=25, method="uniform")
lbp_var = uniform_filter(lbp_map.astype(float) ** 2, size=window_size) - uniform_filter(lbp_map.astype(float), size=window_size) ** 2
lbp_thresh = 20
mask_lbp = threshold_mask(lbp_var, lbp_thresh)

cv2.imshow("LBP avec zebra 1", mask_lbp)
cv2.waitKey(0)
cv2.destroyAllWindows()
