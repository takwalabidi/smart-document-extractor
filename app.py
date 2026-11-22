import json  # Pour formater et exporter les résultats en JSON
import pandas as pd  # Pour afficher et exporter l'historique sous forme de tableau / CSV
import streamlit as st  # Framework pour créer l'interface web interactive

# Importe les fonctions de tes modules indépendants dans src/
from src.preprocessor import load_image_from_upload, preprocess_image
from src.ocr_engine import extract_ocr_text, save_ocr_image_boxes
from src.llm_extractor import detect_category, extract_fields_with_ollama

# 1. Configuration générale de la page web
st.set_page_config(
    page_title="Smart Document Extractor — CIN & Badges",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Style CSS léger pour donner un rendu plus propre et moderne aux cartes
st.markdown(
    """
    <style>
        .main-header {
            padding: 1.2rem 1.5rem;
            border-radius: 12px;
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
            color: white;
            margin-bottom: 1.2rem;
        }
        .main-header h1 {
            margin: 0;
            font-size: 1.7rem;
            font-weight: 700;
        }
        .main-header p {
            margin: 0.35rem 0 0 0;
            color: #cbd5e1;
            font-size: 0.95rem;
        }
        .stButton > button[kind="primary"] {
            border-radius: 8px;
            font-weight: 600;
            padding: 0.55rem 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# 3. Initialise l'historique des analyses dans la session Streamlit
if "historique" not in st.session_state:
    st.session_state.historique = []

# 4. Bandeau supérieur de l'application
st.markdown(
    """
    <div class="main-header">
        <h1>🪪 Smart Document Extractor</h1>
        <p>Pipeline local d'extraction d'informations sur pièces justificatives (CIN & Badges) — OpenCV • EasyOCR • Ollama</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 5. Barre latérale (paramètres et statistiques de session)
with st.sidebar:
    st.header("⚙️ Configuration")

    # Choix du modèle local Ollama
    model_name = st.selectbox(
        "Modèle LLM (Ollama)",
        options=["llama3.2", "llama3.1", "mistral"],
        index=0,
        help="Assurez-vous que le modèle est téléchargé via 'ollama pull <modele>'.",
    )

    # Choix des langues pour EasyOCR
    langs = st.selectbox(
        "Langues EasyOCR",
        options=[("fr", "en"), ("ar", "en"), ("en",)],
        format_func=lambda pair: " + ".join(pair).upper(),
        index=0,
    )

    # Option pour afficher ou non les cadres verts détectés par EasyOCR
    show_boxes = st.toggle("Afficher les cadres de détection OCR", value=True)

    st.divider()
    st.subheader("📊 Session en cours")
    st.metric("Documents analysés", len(st.session_state.historique))

    # Bouton pour vider l'historique si besoin
    if st.session_state.historique and st.button("🗑️ Effacer l'historique", use_container_width=True):
        st.session_state.historique = []
        st.rerun()

# 6. Zone d'importation du document (image CIN, badge, etc.)
uploaded_file = st.file_uploader(
    "📥 Importez une image de document (CIN, badge...)",
    type=["jpg", "jpeg", "png"],
    help="Glissez-déposez une image depuis votre dossier data/ ou votre ordinateur.",
)

if uploaded_file is not None:
    # Charge l'image originale en mémoire
    img_bgr = load_image_from_upload(uploaded_file)
    # Convertit BGR (OpenCV) -> RGB pour l'affichage fidèle dans Streamlit
    img_rgb = img_bgr[:, :, ::-1]
    # Applique le prétraitement OpenCV (niveaux de gris + filtre CLAHE)
    cleaned_img = preprocess_image(img_bgr)
    # Détecte automatiquement la catégorie selon le nom du fichier ('cin', 'badges' ou 'autres')
    categorie_auto = detect_category(uploaded_file.name)

    # Barre d'action principale au-dessus des onglets
    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.info(f"**Fichier :** `{uploaded_file.name}`  |  **Catégorie détectée :** `{categorie_auto.upper()}`")
    with col_btn:
        launch_btn = st.button("🚀 Lancer l'extraction", type="primary", use_container_width=True)

    # 7. Exécution du pipeline OCR + LLM lorsque l'utilisateur clique sur le bouton
    if launch_btn:
        try:
            with st.spinner("Étape 1/2 — Lecture du texte avec EasyOCR..."):
                ocr_text = extract_ocr_text(cleaned_img, langs=langs)

            with st.spinner(f"Étape 2/2 — Extraction structurée avec Ollama ({model_name})..."):
                if ocr_text:
                    champs = extract_fields_with_ollama(ocr_text, model_name=model_name)
                else:
                    champs = {
                        "nom": None,
                        "prenom": None,
                        "date": None,
                        "numero_document": None,
                        "organisme_emetteur": None,
                    }

            # Sauvegarde aussi l'image annotée avec les cadres verts si activé
            box_image_path = None
            if show_boxes:
                import os

                os.makedirs("outputs/ocr_visualisations", exist_ok=True)
                box_image_path = os.path.join("outputs/ocr_visualisations", f"preview_{uploaded_file.name}")
                save_ocr_image_boxes(cleaned_img, box_image_path, langs=langs)

            # Enregistre le résultat dans l'historique de la session
            resultat_courant = {
                "fichier": uploaded_file.name,
                "categorie": categorie_auto,
                "modele": model_name,
                "champs": champs,
                "ocr_text": ocr_text,
                "box_image_path": box_image_path,
            }
            st.session_state.last_result = resultat_courant
            st.session_state.historique.append(resultat_courant)
            st.success("Extraction terminée avec succès !")

        except Exception as err:
            st.error(f"Erreur pendant l'extraction : {err}")
            st.caption("Vérifiez que l'application Ollama est bien lancée et que le modèle est téléchargé (`ollama pull llama3.2`).")

    # 8. Affichage dynamique des résultats s'il y a un résultat disponible
    if "last_result" in st.session_state:
        res = st.session_state.last_result
        champs = res["champs"]

        tab1, tab2, tab3 = st.tabs(["📋 Champs Extraits", "🖼️ Prévisualisation Visuelle", "📝 Texte OCR Brut"])

        # Onglet 1 : Les 5 champs demandés + export JSON
        with tab1:
            st.markdown("### Informations structurées")
            c1, c2 = st.columns(2)
            with c1:
                nom_val = st.text_input("Nom", value=champs.get("nom") or "")
                date_val = st.text_input("Date", value=champs.get("date") or "")
                org_val = st.text_input("Organisme émetteur", value=champs.get("organisme_emetteur") or "")
            with c2:
                prenom_val = st.text_input("Prénom", value=champs.get("prenom") or "")
                num_val = st.text_input("Numéro de document", value=champs.get("numero_document") or "")
                cat_val = st.text_input("Catégorie", value=res["categorie"])

            # JSON final (reflète aussi d'éventuelles corrections manuelles dans les champs)
            final_payload = {
                "fichier": res["fichier"],
                "categorie": cat_val,
                "champs_extraits": {
                    "nom": nom_val or None,
                    "prenom": prenom_val or None,
                    "date": date_val or None,
                    "numero_document": num_val or None,
                    "organisme_emetteur": org_val or None,
                },
            }

            st.json(final_payload)

            st.download_button(
                label="📥 Télécharger le JSON",
                data=json.dumps(final_payload, indent=4, ensure_ascii=False),
                file_name=f"{res['fichier'].rsplit('.', 1)[0]}_extrait.json",
                mime="application/json",
            )

        # Onglet 2 : Comparaison visuelle (Original / Nettoyé OpenCV / Cadres OCR)
        with tab2:
            v1, v2, v3 = st.columns(3)
            with v1:
                st.caption("Image originale")
                st.image(img_rgb, use_container_width=True)
            with v2:
                st.caption("Image nettoyée (Gris + CLAHE)")
                st.image(cleaned_img, clamp=True, use_container_width=True)
            with v3:
                st.caption("Zones détectées par EasyOCR")
                if res.get("box_image_path"):
                    st.image(res["box_image_path"], use_container_width=True)
                else:
                    st.info("Activez 'Afficher les cadres de détection OCR' dans la barre latérale.")

        # Onglet 3 : Texte OCR complet
        with tab3:
            st.text_area("Texte brut extrait par EasyOCR", value=res["ocr_text"], height=260)

# 9. Section Historique global de la session (tableau + export CSV/JSON)
if st.session_state.historique:
    st.divider()
    st.subheader("🗂️ Historique des documents traités dans cette session")

    rows = []
    for item in st.session_state.historique:
        c = item["champs"]
        rows.append(
            {
                "Fichier": item["fichier"],
                "Catégorie": item["categorie"],
                "Nom": c.get("nom"),
                "Prénom": c.get("prenom"),
                "Date": c.get("date"),
                "Numéro document": c.get("numero_document"),
                "Organisme émetteur": c.get("organisme_emetteur"),
            }
        )

    df_hist = pd.DataFrame(rows)
    st.dataframe(df_hist, use_container_width=True, hide_index=True)

    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button(
            label="📥 Exporter l'historique (CSV)",
            data=df_hist.to_csv(index=False).encode("utf-8-sig"),
            file_name="historique_extractions.csv",
            mime="text/csv",
        )
    with exp2:
        st.download_button(
            label="📥 Exporter l'historique (JSON)",
            data=json.dumps(rows, indent=4, ensure_ascii=False),
            file_name="historique_extractions.json",
            mime="application/json",
        )