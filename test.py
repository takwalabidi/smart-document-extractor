import json  # Module standard pour lire et décoder les fichiers au format JSON
import unicodedata  # Module standard pour normaliser les caractères Unicode et supprimer les accents

# --- CHOIX DU MODE ---
# Mets None pour calculer le taux sur TOUS les documents (ex: 10 docs = 50 champs)
# OU mets le nom d'un seul fichier pour tester sur 1 image (ex: 1 doc = 5 champs)
IMAGE_CIBLE = None
# Exemple pour tester sur 1 seule image :
# IMAGE_CIBLE = "clean_IDENTITY CARDS_FRENCH_IDCARD_ZH_2025041100008.jpg"


def normaliser(val) -> str:
    """Nettoie le texte (majuscules, sans accents, espaces normalisés) pour comparer proprement."""
    # Vérifie si la valeur est absente (None) ou correspond à une chaîne vide/nulle
    if val is None or str(val).strip().lower() in ("", "none", "null"):
        # Retourne une chaîne vide pour uniformiser les valeurs manquantes
        return ""
    # Supprime les espaces en début/fin et convertit tout le texte en majuscules
    texte = str(val).strip().upper()
    # Décompose les caractères accentués (NFD) et filtre les marques diacritiques (accents)
    texte = "".join(
        c for c in unicodedata.normalize("NFD", texte) if unicodedata.category(c) != "Mn"
    )
    # Remplace les espaces multiples ou retours à la ligne par un seul espace standard
    return " ".join(texte.split())


def charger_documents_json(chemin: str) -> dict:
    """Charge un JSON (dict ou liste) et retourne un dictionnaire {nom_fichier: {5 champs}}."""
    # Ouvre le fichier JSON en mode lecture avec encodage UTF-8 pour gérer les accents
    with open(chemin, "r", encoding="utf-8") as f:
        # Charge le contenu du fichier JSON dans une variable Python (dict ou list)
        data = json.load(f)

    # Initialise le dictionnaire final qui associera chaque nom de fichier à ses champs
    resultat = {}
    # Cas 1 : le JSON est structuré comme un dictionnaire { "nom_doc": { ... } }
    if isinstance(data, dict):
        # Parcourt chaque paire clé (nom du document) et valeur (contenu associé)
        for nom_doc, contenu in data.items():
            # Vérifie que le contenu associé au document est bien un dictionnaire
            if isinstance(contenu, dict):
                # Récupère le sous-dictionnaire des champs s'il est imbriqué, sinon prend contenu directement
                champs = contenu.get("champs_extraits") or contenu.get("champs") or contenu
                # Enregistre les champs sous la clé du nom de document
                resultat[nom_doc] = champs
    # Cas 2 : le JSON est structuré comme une liste d'objets [ { "fichier": "...", ... }, ... ]
    elif isinstance(data, list):
        # Parcourt chaque élément (dictionnaire représentant un document) dans la liste
        for item in data:
            # Vérifie que l'élément courant est bien un dictionnaire
            if isinstance(item, dict):
                # Cherche le nom du fichier parmi les différentes clés possibles
                nom_doc = (
                    item.get("fichier")
                    or item.get("filename")
                    or item.get("document")
                    or item.get("nom_fichier")
                )
                # Récupère le dictionnaire des champs extraits ou l'objet complet par défaut
                champs = item.get("champs_extraits") or item.get("champs") or item
                # Si un nom de fichier valide a été trouvé, on l'ajoute au dictionnaire résultat
                if nom_doc:
                    resultat[nom_doc] = champs
    # Retourne le dictionnaire uniformisé {nom_fichier: champs}
    return resultat


# 1. Charger les résultats produits par llm_extractor.py
# Initialise un dictionnaire vide pour regrouper toutes les prédictions du modèle IA
extractions = {}
# Ajoute les résultats extraits pour les cartes d'identité (CIN)
extractions.update(charger_documents_json("outputs/extracteur/cin/resultats_cin.json"))
# Ajoute les résultats extraits pour les badges
extractions.update(charger_documents_json("outputs/extracteur/badges/resultats_badges.json"))

# 2. Charger la vérité terrain (fonctionne que ground_truth.json soit une liste ou un dict)
verite_terrain = charger_documents_json("data/ground_truth.json")

# Filtrer sur une seule image si IMAGE_CIBLE est renseigné
if IMAGE_CIBLE:
    # Vérifie que le fichier ciblé existe bien dans le fichier de référence ground_truth.json
    if IMAGE_CIBLE not in verite_terrain:
        # Affiche un message d'erreur et la liste des fichiers valides si le nom est introuvable
        print(f"Erreur : '{IMAGE_CIBLE}' n'est pas trouvé dans data/ground_truth.json")
        print("Documents disponibles :", list(verite_terrain.keys()))
        # Arrête immédiatement l'exécution du script
        exit()
    # Restreint l'évaluation uniquement au document spécifié dans IMAGE_CIBLE
    docs_a_evaluer = {IMAGE_CIBLE: verite_terrain[IMAGE_CIBLE]}
else:
    # Évalue l'ensemble des documents présents dans la vérité terrain
    docs_a_evaluer = verite_terrain

# 3. Calculer le taux
# Compteur du nombre total de champs correctement extraits par l'IA
champs_corrects = 0
# Compteur du nombre total de champs évalués sur l'ensemble des documents
total_champs = 0
# Liste fixe des 5 champs à comparer pour chaque document
CHAMPS_CIBLES = ["nom", "prenom", "date", "numero_document", "organisme_emetteur"]

# Parcourt chaque document de référence et ses valeurs attendues
for doc, vrais_champs in docs_a_evaluer.items():
    # Récupère les champs extraits par l'IA pour ce document (dictionnaire vide si absent)
    champs_ia = extractions.get(doc, {})
    # Compteur de champs corrects uniquement pour le document en cours
    doc_ok = 0
    # Affiche le nom du document évalué dans le terminal
    print(f"\n📄 Document : {doc}")

    # Compare chacun des 5 champs cibles un par un
    for champ in CHAMPS_CIBLES:
        # Récupère la valeur de référence (attendue) pour ce champ
        vraie_val = vrais_champs.get(champ)
        # Récupère la valeur prédite par l'IA pour ce champ
        val_ia = champs_ia.get(champ)
        # Incrémente le compteur global de champs évalués
        total_champs += 1

        # Compare les deux valeurs après normalisation (casse, accents, espaces)
        est_exact = normaliser(vraie_val) == normaliser(val_ia)
        # Si les deux valeurs normalisées sont identiques
        if est_exact:
            # Incrémente le score global et le score du document
            champs_corrects += 1
            doc_ok += 1
            statut = "✅ OK"
        else:
            # Marque le champ comme erroné en cas de différence
            statut = "❌ FAUX"

        # Affiche le détail aligné : nom du champ, valeur attendue, valeur IA et statut
        print(f"   - {champ:20s} | Attendu: {str(vraie_val):25s} | IA: {str(val_ia):25s} | {statut}")

    # Calcule le pourcentage de réussite pour ce document spécifique
    taux_doc = (doc_ok / len(CHAMPS_CIBLES)) * 100
    # Affiche le score individuel du document
    print(f"   --> Score de ce document : {doc_ok}/{len(CHAMPS_CIBLES)} ({taux_doc:.1f} %)")

# Calcule le taux global d'extraction en évitant toute division par zéro
taux_global = (champs_corrects / total_champs) * 100 if total_champs > 0 else 0
# Affiche le bilan final formaté dans le terminal
print("\n" + "=" * 55)
print(f"RÉSULTAT FINAL : {champs_corrects} / {total_champs} champs corrects")
print(f"TAUX D'EXTRACTION : {taux_global:.2f} %")
print("=" * 55)