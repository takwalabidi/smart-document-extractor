import json
import unicodedata

# --- CHOIX DU MODE ---
# Mets None pour calculer le taux sur TOUS les documents (ex: 10 docs = 50 champs)
# OU mets le nom d'un seul fichier pour tester sur 1 image (ex: 1 doc = 5 champs)
IMAGE_CIBLE = None
# Exemple pour tester sur 1 seule image :
# IMAGE_CIBLE = "clean_IDENTITY CARDS_FRENCH_IDCARD_ZH_2025041100008.jpg"


def normaliser(val) -> str:
    """Nettoie le texte (majuscules, sans accents, espaces normalisés) pour comparer proprement."""
    if val is None or str(val).strip().lower() in ("", "none", "null"):
        return ""
    texte = str(val).strip().upper()
    texte = "".join(
        c for c in unicodedata.normalize("NFD", texte) if unicodedata.category(c) != "Mn"
    )
    return " ".join(texte.split())


def charger_documents_json(chemin: str) -> dict:
    """Charge un JSON (dict ou liste) et retourne un dictionnaire {nom_fichier: {5 champs}}."""
    with open(chemin, "r", encoding="utf-8") as f:
        data = json.load(f)

    resultat = {}
    if isinstance(data, dict):
        for nom_doc, contenu in data.items():
            if isinstance(contenu, dict):
                champs = contenu.get("champs_extraits") or contenu.get("champs") or contenu
                resultat[nom_doc] = champs
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                nom_doc = (
                    item.get("fichier")
                    or item.get("filename")
                    or item.get("document")
                    or item.get("nom_fichier")
                )
                champs = item.get("champs_extraits") or item.get("champs") or item
                if nom_doc:
                    resultat[nom_doc] = champs
    return resultat


# 1. Charger les résultats produits par llm_extractor.py
extractions = {}
extractions.update(charger_documents_json("outputs/extracteur/cin/resultats_cin.json"))
extractions.update(charger_documents_json("outputs/extracteur/badges/resultats_badges.json"))

# 2. Charger la vérité terrain (fonctionne que ground_truth.json soit une liste ou un dict)
verite_terrain = charger_documents_json("data/ground_truth.json")

# Filtrer sur une seule image si IMAGE_CIBLE est renseigné
if IMAGE_CIBLE:
    if IMAGE_CIBLE not in verite_terrain:
        print(f"Erreur : '{IMAGE_CIBLE}' n'est pas trouvé dans data/ground_truth.json")
        print("Documents disponibles :", list(verite_terrain.keys()))
        exit()
    docs_a_evaluer = {IMAGE_CIBLE: verite_terrain[IMAGE_CIBLE]}
else:
    docs_a_evaluer = verite_terrain

# 3. Calculer le taux
champs_corrects = 0
total_champs = 0
CHAMPS_CIBLES = ["nom", "prenom", "date", "numero_document", "organisme_emetteur"]

for doc, vrais_champs in docs_a_evaluer.items():
    champs_ia = extractions.get(doc, {})
    doc_ok = 0
    print(f"\n📄 Document : {doc}")

    for champ in CHAMPS_CIBLES:
        vraie_val = vrais_champs.get(champ)
        val_ia = champs_ia.get(champ)
        total_champs += 1

        est_exact = normaliser(vraie_val) == normaliser(val_ia)
        if est_exact:
            champs_corrects += 1
            doc_ok += 1
            statut = "✅ OK"
        else:
            statut = "❌ FAUX"

        print(f"   - {champ:20s} | Attendu: {str(vraie_val):25s} | IA: {str(val_ia):25s} | {statut}")

    taux_doc = (doc_ok / len(CHAMPS_CIBLES)) * 100
    print(f"   --> Score de ce document : {doc_ok}/{len(CHAMPS_CIBLES)} ({taux_doc:.1f} %)")

taux_global = (champs_corrects / total_champs) * 100 if total_champs > 0 else 0
print("\n" + "=" * 55)
print(f"RÉSULTAT FINAL : {champs_corrects} / {total_champs} champs corrects")
print(f"TAUX D'EXTRACTION : {taux_global:.2f} %")
print("=" * 55)