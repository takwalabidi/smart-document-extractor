import os  # Module standard pour manipuler les dossiers et chemins de fichiers
import json  # Module standard pour lire et écrire des fichiers au format JSON
import ollama  # Bibliothèque officielle pour communiquer avec le serveur IA local Ollama
from pydantic import BaseModel, Field  # Pour imposer une structure JSON stricte aux 5 champs demandés


# Définit le schéma exact des 5 informations demandées par l'enseignante
class ExtractedDocument(BaseModel):
    # Champ 1 : Nom de famille de la personne
    nom: str | None = Field(default=None, description="Nom de famille de la personne principale")
    # Champ 2 : Prénom de la personne
    prenom: str | None = Field(default=None, description="Prénom de la personne principale")
    # Champ 3 : Date (naissance, délivrance, validité ou date du document)
    date: str | None = Field(default=None, description="Date pertinente trouvée sur le document")
    # Champ 4 : Numéro de document (numéro de CIN, matricule, référence ou identifiant)
    numero_document: str | None = Field(default=None, description="Numéro de pièce d'identité, badge ou document")
    # Champ 5 : Organisme émetteur (autorité, préfecture, ministère, école ou entreprise)
    organisme_emetteur: str | None = Field(default=None, description="Organisme ou entité ayant émis le document")


def extract_fields_with_ollama(ocr_text: str, model_name: str = "llama3.2") -> dict:
    """Envoie le texte OCR à Ollama et retourne les 5 champs demandés."""
    # Prépare le prompt (consignes) pour guider le modèle local sans invention
    prompt = f"""
Tu es un système d'extraction d'informations administratives.
À partir du texte issu d'un OCR ci-dessous (qui peut contenir de petites fautes d'orthographe), extrais uniquement ces 5 informations :
- nom
- prenom
- date
- numero_document
- organisme_emetteur

Règles strictes :
- Si une information n'est pas présente ou n'est pas lisible, utilise null.
- Ne devine jamais et n'invente rien.

Texte OCR :
\"\"\"
{ocr_text}
\"\"\"
"""

    # Appelle le modèle local via Ollama en forçant une sortie JSON conforme à ExtractedDocument
    response = ollama.chat(
        model=model_name,  # Nom du modèle installé localement (ex: "llama3.2")
        messages=[{"role": "user", "content": prompt}],  # Message envoyé au LLM
        format=ExtractedDocument.model_json_schema(),  # Force la structure JSON des 5 champs
        options={"temperature": 0.0},  # Température à 0.0 pour des réponses stables et factuelles
    )

    # Récupère la chaîne JSON générée par Ollama
    raw_json = response["message"]["content"]
    # Valide que le JSON correspond bien au schéma Pydantic
    validated = ExtractedDocument.model_validate_json(raw_json)
    # Retourne un dictionnaire Python propre avec les 5 clés
    return validated.model_dump()


def parse_ocr_report(report_path: str = "outputs/ocr_resultats.txt") -> list[tuple[str, str]]:
    """Lit le fichier outputs/ocr_resultats.txt et sépare (nom_fichier, texte_ocr) pour chaque document."""
    # Vérifie que le fichier de résultats OCR existe bien
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Fichier OCR introuvable : {report_path}. Lance d'abord 'python src/ocr_engine.py'.")

    # Lit le contenu complet du rapport OCR en UTF-8
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Découpe le rapport selon l'en-tête généré par ocr_engine.py : "=== [1/10] nom_fichier ==="
    entries: list[tuple[str, str]] = []
    blocks = content.split("=== [")

    for block in blocks[1:]:
        # Sépare la première ligne (qui contient l'index et le nom du fichier) du reste du texte OCR
        header_end = block.find("] ")
        line_end = block.find(" ===")
        if header_end != -1 and line_end != -1:
            filename = block[line_end + 5 : block.find(" ===\n")].strip() if " ===\n" in block else ""
            # Solution plus simple et sûre pour récupérer le nom de fichier après "] "
            after_bracket = block[header_end + 2 :]
            filename = after_bracket.split(" ===")[0].strip()
            ocr_text = after_bracket.split(" ===\n", 1)[1].strip() if " ===\n" in after_bracket else ""
            entries.append((filename, ocr_text))

    return entries


def detect_category(filename: str) -> str:
    """Détermine si le document appartient à la catégorie 'cin' ou 'badges' selon son nom."""
    # Convertit le nom en majuscules pour éviter les problèmes de casse
    upper_name = filename.upper()
    # Dans preprocessor.py, le nom contient le dossier d'origine (ex: clean_IDENTITY CARDS_... ou clean_BADGES AND PASSES_...)
    if "IDENTITY" in upper_name or "CIN" in upper_name or "IDCARD" in upper_name or "ID_CARD" in upper_name:
        return "cin"
    if "BADGE" in upper_name or "PASS" in upper_name:
        return "badges"
    # Par défaut si ni CIN ni badge n'est détecté dans le nom
    return "autres"


def run_batch_extraction(
    report_path: str = "outputs/ocr_resultats.txt",
    output_base_dir: str = "outputs/extracteur",
    model_name: str = "llama3.2",
    max_docs: int | None = None,
) -> None:
    """Exécute l'extraction Ollama sur les textes OCR et classe les résultats CIN et Badges séparément."""
    # Affiche l'en-tête de démarrage dans le terminal
    print("=== DÉMARRAGE DE L'EXTRACTEUR (OLLAMA) ===")
    # Crée les sous-dossiers séparés pour les CIN et les Badges
    cin_dir = os.path.join(output_base_dir, "cin")
    badges_dir = os.path.join(output_base_dir, "badges")
    autres_dir = os.path.join(output_base_dir, "autres")
    os.makedirs(cin_dir, exist_ok=True)
    os.makedirs(badges_dir, exist_ok=True)
    os.makedirs(autres_dir, exist_ok=True)

    # Récupère la liste des (nom_fichier, texte_ocr) depuis le rapport OCR
    documents = parse_ocr_report(report_path)

    # Limite le nombre de documents si max_docs est renseigné (utile pour un test rapide)
    if max_docs is not None:
        documents = documents[:max_docs]

    total = len(documents)
    print(f"Nombre de documents à extraire avec '{model_name}' : {total}")

    # Listes pour regrouper les résultats par catégorie
    resultats_cin: list[dict] = []
    resultats_badges: list[dict] = []
    resultats_autres: list[dict] = []

    # Parcourt chaque document un par un
    for idx, (filename, ocr_text) in enumerate(documents, start=1):
        # Détermine la catégorie ('cin', 'badges' ou 'autres')
        category = detect_category(filename)

        print("\n" + "=" * 60)
        print(f"[{idx}/{total}] DOCUMENT : {filename}  |  CATÉGORIE : {category.upper()}")
        print("=" * 60)

        # Si aucun texte OCR n'a été trouvé pour ce document, on met des valeurs vides sans appeler Ollama
        if not ocr_text or ocr_text == "(Aucun texte détecté)":
            fields = {
                "nom": None,
                "prenom": None,
                "date": None,
                "numero_document": None,
                "organisme_emetteur": None,
            }
            print("Texte OCR vide -> champs mis à None.")
        else:
            # Appelle Ollama pour extraire les 5 champs demandés
            fields = extract_fields_with_ollama(ocr_text, model_name=model_name)

        # Affiche clairement les 5 champs demandés dans le terminal
        print("--- CHAMPS EXTRAITS ---")
        print(f"nom                = {fields.get('nom')}")
        print(f"prenom             = {fields.get('prenom')}")
        print(f"date               = {fields.get('date')}")
        print(f"numero_document    = {fields.get('numero_document')}")
        print(f"organisme_emetteur = {fields.get('organisme_emetteur')}")

        # Prépare l'objet complet à sauvegarder en JSON pour ce document
        record = {
            "fichier": filename,
            "categorie": category,
            "champs_extraits": fields,
        }

        # Choisit le bon dossier de sortie et ajoute au bon groupe
        if category == "cin":
            target_dir = cin_dir
            resultats_cin.append(record)
        elif category == "badges":
            target_dir = badges_dir
            resultats_badges.append(record)
        else:
            target_dir = autres_dir
            resultats_autres.append(record)

        # Sauvegarde un fichier JSON individuel par document dans son dossier (cin ou badges)
        base_name = os.path.splitext(filename)[0]
        individual_json_path = os.path.join(target_dir, f"{base_name}_extrait.json")
        with open(individual_json_path, "w", encoding="utf-8") as jf:
            json.dump(record, jf, indent=4, ensure_ascii=False)
        print(f"Sauvegardé dans : {individual_json_path}")

    # Sauvegarde un fichier global dédié aux résultats CIN
    with open(os.path.join(cin_dir, "resultats_cin.json"), "w", encoding="utf-8") as f_cin:
        json.dump(resultats_cin, f_cin, indent=4, ensure_ascii=False)

    # Sauvegarde un fichier global dédié aux résultats Badges
    with open(os.path.join(badges_dir, "resultats_badges.json"), "w", encoding="utf-8") as f_badges:
        json.dump(resultats_badges, f_badges, indent=4, ensure_ascii=False)

    # Sauvegarde aussi un fichier global si d'autres documents existent
    if resultats_autres:
        with open(os.path.join(autres_dir, "resultats_autres.json"), "w", encoding="utf-8") as f_autres:
            json.dump(resultats_autres, f_autres, indent=4, ensure_ascii=False)

    print("\n=== FIN DE L'EXTRACTION ===")
    print(f"Résultats CIN ({len(resultats_cin)})    -> {os.path.abspath(cin_dir)}")
    print(f"Résultats Badges ({len(resultats_badges)}) -> {os.path.abspath(badges_dir)}")


# Exécute l'extraction uniquement si ce fichier est lancé directement
if __name__ == "__main__":
    # Astuce : mets max_docs=3 pour tester vite sur 3 documents, ou None pour tout traiter
    run_batch_extraction(
        report_path="outputs/ocr_resultats.txt",
        output_base_dir="outputs/extracteur",
        model_name="llama3.2",
        max_docs=None,
    )