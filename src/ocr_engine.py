import os  # Pour manipuler les dossiers et chemins de fichiers
import glob  # Pour trouver automatiquement tous les fichiers images d'un dossier
import cv2  # Pour charger une image et dessiner les cadres détectés par l'OCR
import easyocr  # Bibliothèque d'OCR (détection et lecture du texte sur une image)
import numpy as np  # Pour manipuler les tableaux de pixels d'une image

# Variable globale pour garder le lecteur EasyOCR en mémoire une seule fois
_READER_CACHE: dict[tuple[str, ...], easyocr.Reader] = {}


def get_reader(langs: tuple[str, ...] = ("fr", "en")) -> easyocr.Reader:
    """Charge le modèle EasyOCR une seule fois et le garde en mémoire."""
    # Vérifie si le modèle pour ces langues a déjà été chargé
    if langs not in _READER_CACHE:
        print(f"Chargement du modèle EasyOCR pour les langues {langs} (sur CPU)...")
        # gpu=False car on exécute sur le processeur classique du PC
        _READER_CACHE[langs] = easyocr.Reader(list(langs), gpu=False)
    # Retourne le lecteur déjà prêt
    return _READER_CACHE[langs]


def extract_ocr_text(img: np.ndarray, langs: tuple[str, ...] = ("fr", "en")) -> str:
    """Extrait et retourne tout le texte lisible d'une image."""
    # Récupère le lecteur EasyOCR
    reader = get_reader(langs)
    # detail=0 -> retourne uniquement les chaînes de texte
    # paragraph=True -> regroupe les mots proches en blocs de texte cohérents
    blocks = reader.readtext(img, detail=0, paragraph=True)
    # Joint tous les blocs avec un retour à la ligne et nettoie les espaces superflus
    return "\n".join(blocks).strip()


def save_ocr_image_boxes(img: np.ndarray, output_path: str, langs: tuple[str, ...] = ("fr", "en")) -> str:
    """Dessine des cadres verts autour des zones de texte détectées et sauvegarde l'image."""
    # Récupère le lecteur EasyOCR
    reader = get_reader(langs)
    # detail=1 -> retourne (coordonnees_cadre, texte, score_confiance) pour chaque zone
    detections = reader.readtext(img, detail=1)

    # Convertit l'image en couleur (BGR) pour pouvoir dessiner des rectangles verts visibles
    if len(img.shape) == 2:
        vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = img.copy()

    # Parcourt chaque zone de texte détectée
    for bbox, _text, _conf in detections:
        # Coin supérieur gauche (x1, y1) et coin inférieur droit (x2, y2)
        pt1 = (int(bbox[0][0]), int(bbox[0][1]))
        pt2 = (int(bbox[2][0]), int(bbox[2][1]))
        # Dessine un rectangle vert d'épaisseur 2 pixels autour du texte
        cv2.rectangle(vis_img, pt1, pt2, (0, 255, 0), 2)

    # Sauvegarde l'image annotée sur le disque
    cv2.imwrite(output_path, vis_img)
    return output_path


def run_batch_ocr(
    input_dir: str = "outputs/cleaned_images",
    vis_dir: str = "outputs/ocr_visualisations",
    report_path: str = "outputs/ocr_resultats.txt",
    langs: tuple[str, ...] = ("fr", "en"),
    max_images: int | None = None,
) -> None:
    """Lance l'OCR sur les images et affiche tout le texte OCR dans le terminal."""
    print("=== DÉMARRAGE DU MOTEUR OCR (EASYOCR) ===")
    # Crée les dossiers de sortie s'ils n'existent pas encore
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # Cherche toutes les images (.jpg, .jpeg, .png) dans le dossier d'entrée
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    image_paths = sorted(image_paths)

    # Si l'utilisateur veut limiter à un petit nombre d'images pour tester vite
    if max_images is not None:
        image_paths = image_paths[:max_images]

    total = len(image_paths)
    print(f"Nombre d'images à analyser dans '{input_dir}' : {total}")

    rapport_lignes: list[str] = []

    # Traite chaque image une par une
    for idx, img_path in enumerate(image_paths, start=1):
        nom_fichier = os.path.basename(img_path)
        print("\n" + "-" * 60)
        print(f"[{idx}/{total}] FICHIER : {nom_fichier}")
        print("-" * 60)

        # Charge l'image depuis le disque
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            msg = f"ERREUR : impossible de lire {img_path}"
            print(msg)
            rapport_lignes.append(msg)
            continue

        # 1. Extraction de tout le texte OCR
        texte_ocr = extract_ocr_text(img, langs=langs)
        print("--- TEXTE OCR ---")
        print(texte_ocr if texte_ocr else "(Aucun texte détecté)")

        # 2. Génération de l'image OCR avec les cadres verts
        vis_out_path = os.path.join(vis_dir, f"boxes_{nom_fichier}")
        save_ocr_image_boxes(img, vis_out_path, langs=langs)
        print(f"--- IMAGE OCR (cadres) --- : {vis_out_path}")

        # Ajoute le résultat au rapport texte complet
        rapport_lignes.append(f"=== [{idx}/{total}] {nom_fichier} ===\n{texte_ocr}\n")

    # Enregistre tout le texte OCR dans un fichier .txt facile à relire
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rapport_lignes))

    print("\n=== FIN DE L'EXTRACTION OCR ===")
    print(f"Tout le texte OCR a été enregistré dans : {os.path.abspath(report_path)}")
    print(f"Les images avec cadres OCR sont dans : {os.path.abspath(vis_dir)}")


# S'exécute uniquement quand on lance directement : python src/ocr_engine.py
if __name__ == "__main__":
    # Astuce : mets max_images=3 si tu veux tester d'abord sur 3 images, ou None pour toutes
    run_batch_ocr(
        input_dir="outputs/cleaned_images",
        vis_dir="outputs/ocr_visualisations",
        report_path="outputs/ocr_resultats.txt",
        langs=("fr", "en"),
        max_images=None,
    )