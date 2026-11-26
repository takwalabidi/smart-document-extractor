import os  # Module standard pour gérer les chemins de fichiers et créer des dossiers
import glob  # Module standard pour rechercher automatiquement les fichiers images selon un motif
import cv2  # Bibliothèque OpenCV pour charger, transformer et sauvegarder des images
import numpy as np  # Bibliothèque NumPy pour manipuler les pixels d'une image sous forme de tableau


def load_image_from_path(image_path: str) -> np.ndarray:
    """Charge une image depuis un chemin sur le disque au format BGR (OpenCV)."""
    # Lit l'image en couleur depuis le disque (renvoie None si le chemin est faux ou illisible)
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Vérifie que l'image a bien été chargée avant de continuer
    if img_bgr is None:
        # Lève une erreur explicite si le fichier ne peut pas être ouvert
        raise ValueError(f"Impossible de lire l'image : {image_path}")
    # Retourne le tableau de pixels de l'image
    return img_bgr


def load_image_from_upload(uploaded_file) -> np.ndarray:
    """Charge une image envoyée en mémoire (par exemple depuis Streamlit)."""
    # Lit le contenu binaire (les octets) du fichier importé
    file_bytes = uploaded_file.read()
    # Transforme les octets bruts en un tableau 1D d'entiers non signés sur 8 bits (0 à 255)
    np_arr = np.frombuffer(file_bytes, np.uint8)
    # Décode le tableau en une image couleur OpenCV
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # Vérifie que le décodage a réussi
    if img_bgr is None:
        # Lève une erreur si les données ne correspondent pas à une image valide
        raise ValueError("Impossible de décoder l'image importée.")
    # Retourne l'image décodée
    return img_bgr


def denoise(gray: np.ndarray, strength: int) -> np.ndarray:
    """Réduit le bruit. strength = 0 -> ne rien faire."""
    # Si la force est nulle ou négative, on retourne l'image telle quelle
    if strength <= 0:
        return gray
    # Médian : supprime les points noirs/blancs isolés (bruit poivre et sel)
    denoised = cv2.medianBlur(gray, 3)
    # Bilatéral : lisse le grain tout en conservant les bords des lettres
    denoised = cv2.bilateralFilter(denoised, d=9, sigmaColor=strength, sigmaSpace=strength)
    # Retourne l'image débruitée
    return denoised


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Fait pivoter l'image autour de son centre sans couper les bords (fond blanc)."""
    # Si l'angle est nul, aucune rotation n'est nécessaire
    if angle == 0.0:
        return image
    # Récupère les dimensions (hauteur, largeur) de l'image
    h, w = image.shape[:2]
    # Calcule le point central de l'image
    center = (w / 2.0, h / 2.0)
    # Génère la matrice de rotation 2D autour du centre
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calcule le cosinus et le sinus absolus pour déterminer les nouvelles dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Ajuste la translation dans la matrice pour centrer l'image dans le nouveau cadre
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]

    # Applique la rotation avec interpolation cubique et remplissage blanc (255)
    return cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def estimate_skew_angle(gray: np.ndarray, max_angle: float = 15.0) -> float:
    """
    Estime l'angle de rotation à appliquer pour redresser le texte (en degrés).
    Méthode : profil de projection. Quand les lignes de texte sont horizontales,
    la somme des pixels d'encre par ligne alterne fortement (lignes pleines /
    lignes vides) -> variance maximale. On teste plusieurs angles et on garde
    celui qui maximise cette variance.
    """
    # 1. Réduire l'image pour aller vite
    h, w = gray.shape[:2]
    scale = 600.0 / max(h, w)
    small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 2. Corriger l'éclairage inégal : on divise l'image par son fond (flou très large)
    background = cv2.GaussianBlur(small, (0, 0), sigmaX=25)
    flat = cv2.divide(small, background, scale=255)

    # 3. Binariser : l'encre (texte, traits) devient blanche (255)
    flat = cv2.GaussianBlur(flat, (3, 3), 0)  # adoucit le bruit avant le seuil
    _, bw = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Score d'un angle = variance de la somme des pixels par ligne
    sh, sw = bw.shape
    center = (sw / 2.0, sh / 2.0)

    def score(angle: float) -> float:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(bw, M, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
        return float(np.var(rotated.sum(axis=1)))

    # 5. Recherche grossière (pas de 1°) puis fine (pas de 0.1°)
    coarse_angles = np.arange(-max_angle, max_angle + 1.0, 1.0)
    best_angle = max(coarse_angles, key=score)

    fine_angles = np.arange(best_angle - 1.0, best_angle + 1.05, 0.1)
    best_angle = max(fine_angles, key=score)

    return float(best_angle)


def preprocess_image(
    img_bgr: np.ndarray,
    denoise_strength: int = 55,
    angle: float | None = 0.0,
) -> np.ndarray:
    """
    Niveaux de gris + débruitage + CLAHE + (optionnel) redressement.

    denoise_strength : 0 = pas de débruitage, ~50-75 = efficace
    angle            : 0.0  -> aucune rotation
                       None -> angle détecté automatiquement
                       autre valeur -> rotation manuelle en degrés
    """
    # Convertit l'image couleur (3 canaux : Bleu, Vert, Rouge) en niveaux de gris (1 seul canal)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Applique la réduction de bruit (médian + bilatéral)
    gray = denoise(gray, denoise_strength)

    # Crée l'algorithme CLAHE (contraste adaptatif par petites zones) pour corriger ombres et reflets
    # clipLimit=2.0 évite d'amplifier trop le bruit ; tileGridSize=(8, 8) définit la taille des zones
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Applique l'amélioration de contraste sur l'image en niveaux de gris
    enhanced = clahe.apply(gray)

    # Détecte automatiquement l'angle d'inclinaison si angle est None
    if angle is None:
        angle = estimate_skew_angle(gray)
    # Pivote l'image si un angle non nul est défini ou détecté
    if angle != 0.0:
        enhanced = rotate_image(enhanced, angle)

    # Retourne l'image nettoyée, plus nette pour la lecture OCR
    return enhanced


def run_batch_preprocessing(data_dir: str = "data", output_dir: str = "outputs/cleaned_images") -> list[str]:
    """Parcourt toutes les images de data_dir et enregistre les versions nettoyées."""
    # Affiche un message de début dans le terminal
    print("=== DÉMARRAGE DU PRÉTRAITEMENT (OPENCV) ===")
    # Crée le dossier de sortie (et les dossiers parents si besoin) sans erreur s'il existe déjà
    os.makedirs(output_dir, exist_ok=True)

    # Extensions d'images acceptées
    extensions = ("*.jpg", "*.jpeg", "*.png")
    # Liste qui contiendra tous les chemins d'images trouvés
    image_paths = []
    # Cherche récursivement (dans tous les sous-dossiers) les fichiers de chaque extension
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))

    # Trie la liste alphabétiquement pour un ordre d'exécution régulier
    image_paths = sorted(image_paths)
    # Compte le nombre total d'images trouvées
    total = len(image_paths)
    # Affiche le total dans le terminal
    print(f"Nombre total de documents trouvés dans '{data_dir}' : {total}")

    # Liste pour mémoriser les chemins des images nettoyées sauvegardées
    saved_paths = []
    # Parcourt chaque image avec son numéro d'ordre (commençant à 1)
    for index, img_path in enumerate(image_paths, start=1):
        # Récupère le chemin relatif par rapport à data/ (ex: IDENTITY CARDS\carte1.jpg)
        rel_path = os.path.relpath(img_path, data_dir)
        # Remplace les slashs/antislashs par des underscores pour éviter les doublons de noms
        safe_name = rel_path.replace("\\", "_").replace("/", "_")
        # Construit le chemin complet du fichier de sortie nettoyé
        out_path = os.path.join(output_dir, f"clean_{safe_name}")

        try:
            # Charge l'image originale
            img_bgr = load_image_from_path(img_path)
            # Applique le prétraitement (gris + débruitage + CLAHE + rotation)
            cleaned_img = preprocess_image(img_bgr)
            # Enregistre l'image nettoyée sur le disque
            cv2.imwrite(out_path, cleaned_img)
            # Ajoute le chemin à la liste des fichiers enregistrés
            saved_paths.append(out_path)
            # Affiche la progression réussie dans le terminal
            print(f"[{index}/{total}] OK -> {rel_path}  ==>  {out_path}")
        except Exception as e:
            # Affiche l'erreur pour cette image sans arrêter le traitement des autres
            print(f"[{index}/{total}] ERREUR sur {rel_path} : {e}")

    # Affiche les messages de fin et le chemin absolu du dossier de résultats
    print("=== FIN DU PRÉTRAITEMENT ===")
    print(f"Images nettoyées enregistrées dans : {os.path.abspath(output_dir)}")
    # Retourne la liste des chemins générés
    return saved_paths


# Exécute le traitement complet uniquement si ce fichier est lancé directement
if __name__ == "__main__":
    run_batch_preprocessing(data_dir="data", output_dir="outputs/cleaned_images")
    