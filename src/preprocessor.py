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


def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """Convertit l'image en niveaux de gris et améliore le contraste avec CLAHE."""
    # Convertit l'image couleur (3 canaux : Bleu, Vert, Rouge) en niveaux de gris (1 seul canal)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Crée l'algorithme CLAHE (contraste adaptatif par petites zones) pour corriger ombres et reflets
    # clipLimit=2.0 évite d'amplifier trop le bruit ; tileGridSize=(8, 8) définit la taille des zones
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Applique l'amélioration de contraste sur l'image en niveaux de gris
    enhanced = clahe.apply(gray)
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
            # Applique le prétraitement (gris + CLAHE)
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
