import json  # Pour formater et exporter les résultats en JSON
import re  # Pour nettoyer proprement les chaînes HTML multilignes
import streamlit as st  # Framework web interactif

# Importe tes modules existants dans src/
from src.preprocessor import load_image_from_upload, preprocess_image
from src.ocr_engine import extract_ocr_text
from src.llm_extractor import extract_fields_with_ollama

# 1. Configuration générale de la page
st.set_page_config(
    page_title="Smart Extractor Document",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def compact_html(html_str: str) -> str:
    """Supprime les retours à la ligne et espaces superflus pour éviter que Streamlit affiche du HTML brut."""
    return re.sub(r">\s+<", "><", html_str).strip()


# 2. CSS personnalisé (fond clair, sidebar bleu nuit, file_uploader clair sans bande noire)
st.markdown(
    compact_html(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #f3f5ff 0%, #f8faff 45%, #ffffff 100%);
                color: #0f172a;
            }

            /* Barre latérale bleu nuit profonde */
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0b132b 0%, #111d42 65%, #191847 100%);
                color: #ffffff;
            }
            [data-testid="stSidebar"] * {
                color: #e2e8f0;
            }

            /* Correction de la bande noire du file_uploader en mode clair */
            [data-testid="stFileUploader"] {
                background-color: #ffffff;
                border-radius: 14px;
                padding: 0.4rem;
            }
            [data-testid="stFileUploader"] section,
            [data-testid="stFileUploaderDropzone"] {
                background-color: #ffffff !important;
                border: 1px solid #cbd5e1 !important;
                border-radius: 12px !important;
                color: #0f172a !important;
            }
            [data-testid="stFileUploader"] section *,
            [data-testid="stFileUploaderDropzone"] * {
                color: #334155 !important;
            }
            [data-testid="stFileUploader"] button,
            [data-testid="stFileUploaderDropzone"] button {
                background-color: #eef2ff !important;
                color: #4338ca !important;
                border: 1px solid #c7d2fe !important;
                border-radius: 8px !important;
                font-weight: 600 !important;
            }

            /* Badge "En ligne" en haut à droite */
            .status-online {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background-color: #dcfce7;
                color: #15803d;
                border: 1px solid #bbf7d0;
                padding: 6px 14px;
                border-radius: 999px;
                font-weight: 600;
                font-size: 0.88rem;
                float: right;
            }
            .status-dot {
                width: 9px;
                height: 9px;
                background-color: #16a34a;
                border-radius: 50%;
                display: inline-block;
            }

            /* Titre principal et sous-titre */
            .hero-title {
                text-align: center;
                font-size: 2.35rem;
                font-weight: 800;
                color: #0f172a;
                margin-top: 0.2rem;
                margin-bottom: 0.3rem;
                letter-spacing: -0.5px;
            }
            .hero-title span {
                color: #4f46e5;
            }
            .hero-subtitle {
                text-align: center;
                color: #475569;
                font-size: 1rem;
                margin-bottom: 1.5rem;
            }

            /* Carte d'importation en pointillés */
            .upload-banner {
                background: #ffffff;
                border: 2px dashed #c7d2fe;
                border-radius: 16px;
                padding: 1rem 1.4rem;
                margin-bottom: 0.6rem;
                box-shadow: 0 6px 20px rgba(79, 70, 229, 0.05);
            }
            .fmt-badge {
                display: inline-block;
                background: #eef2ff;
                color: #4338ca;
                font-weight: 700;
                font-size: 0.72rem;
                padding: 2px 8px;
                border-radius: 6px;
                margin-right: 6px;
                border: 1px solid #e0e7ff;
            }

            /* Cartes blanches */
            .info-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 16px;
                padding: 1.15rem 1.35rem;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
                margin-bottom: 1rem;
            }
            .card-header-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.85rem;
                padding-bottom: 0.6rem;
                border-bottom: 1px solid #f1f5f9;
            }
            .card-title {
                font-size: 1.05rem;
                font-weight: 700;
                color: #0f172a;
                margin: 0;
            }
            .success-pill {
                background: #dcfce7;
                color: #15803d;
                font-size: 0.78rem;
                font-weight: 600;
                padding: 4px 10px;
                border-radius: 999px;
            }

            /* Lignes des 5 champs extraits */
            .field-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0.62rem 0.15rem;
                border-bottom: 1px solid #f1f5f9;
            }
            .field-row:last-child {
                border-bottom: none;
            }
            .field-left {
                display: flex;
                align-items: center;
                gap: 10px;
                color: #475569;
                font-size: 0.92rem;
                font-weight: 500;
            }
            .field-icon {
                width: 30px;
                height: 30px;
                border-radius: 9px;
                background: #eef2ff;
                color: #4f46e5;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 0.9rem;
            }
            .field-val {
                font-weight: 700;
                color: #0f172a;
                font-size: 0.94rem;
                text-align: right;
            }
            .field-empty {
                color: #94a3b8;
                font-style: italic;
                font-weight: 500;
            }

            /* Étapes dans la barre latérale */
            .step-box {
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 0.65rem 0.75rem;
                border-radius: 12px;
                margin-bottom: 0.55rem;
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
            .step-box.active {
                background: rgba(99, 102, 241, 0.22);
                border: 1px solid rgba(129, 140, 248, 0.55);
            }
            .step-circle {
                width: 28px;
                height: 28px;
                border-radius: 50%;
                background: #4f46e5;
                color: #ffffff;
                font-weight: 700;
                font-size: 0.85rem;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
            }
            .step-circle.done {
                background: #16a34a;
            }
            .step-title {
                font-size: 0.9rem;
                font-weight: 700;
                color: #ffffff;
                line-height: 1.2;
            }
            .step-sub {
                font-size: 0.76rem;
                color: #94a3b8;
            }

            /* Encadré bas de sidebar */
            .sidebar-footer-card {
                margin-top: 1.2rem;
                padding: 1rem;
                border-radius: 14px;
                background: linear-gradient(145deg, rgba(79, 70, 229, 0.28), rgba(30, 27, 75, 0.65));
                border: 1px solid rgba(129, 140, 248, 0.35);
                text-align: center;
            }
            .sidebar-pill {
                display: inline-block;
                margin-top: 0.5rem;
                background: #4f46e5;
                color: #ffffff !important;
                font-size: 0.75rem;
                font-weight: 700;
                padding: 5px 12px;
                border-radius: 999px;
            }

            /* Bouton principal violet dégradé */
            div.stButton > button[kind="primary"] {
                background: linear-gradient(90deg, #4f46e5 0%, #6366f1 100%);
                color: #ffffff;
                border: none;
                border-radius: 999px;
                padding: 0.65rem 2.2rem;
                font-weight: 700;
                font-size: 1rem;
                box-shadow: 0 10px 22px rgba(79, 70, 229, 0.28);
            }
            div.stButton > button[kind="primary"]:hover {
                background: linear-gradient(90deg, #4338ca 0%, #4f46e5 100%);
            }
        </style>
        """
    ),
    unsafe_allow_html=True,
)

# 3. Initialisation de l'état de session
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []
if "current_step" not in st.session_state:
    st.session_state.current_step = 1


def render_fields_card(doc_index: int, filename: str, data: dict) -> None:
    """Affiche proprement une carte avec les 5 champs extraits (sans risque de HTML brut)."""
    def fmt_val(val):
        if val:
            return f'<span class="field-val">{val}</span>'
        return '<span class="field-val field-empty">Non détecté</span>'

    html_card = compact_html(
        f"""
        <div class="info-card">
            <div class="card-header-row">
                <span class="card-title">✨ Doc {doc_index} — {filename}</span>
                <span class="success-pill">✓ Extraction réussie !</span>
            </div>
            <div class="field-row">
                <div class="field-left"><span class="field-icon">👤</span>Nom</div>
                {fmt_val(data.get("nom"))}
            </div>
            <div class="field-row">
                <div class="field-left"><span class="field-icon">👤</span>Prénom</div>
                {fmt_val(data.get("prenom"))}
            </div>
            <div class="field-row">
                <div class="field-left"><span class="field-icon">📅</span>Date</div>
                {fmt_val(data.get("date"))}
            </div>
            <div class="field-row">
                <div class="field-left"><span class="field-icon">🪪</span>Numéro</div>
                {fmt_val(data.get("numero_document"))}
            </div>
            <div class="field-row">
                <div class="field-left"><span class="field-icon">🏛️</span>Organisme émetteur</div>
                {fmt_val(data.get("organisme_emetteur"))}
            </div>
        </div>
        """
    )
    st.markdown(html_card, unsafe_allow_html=True)


# 4. Barre latérale : Étapes du traitement + paramètres
with st.sidebar:
    st.markdown(
        compact_html(
            "<h3>📄 <b>Smart Extractor</b> <span style='color:#818cf8;font-weight:700;'>Document</span></h3>"
        ),
        unsafe_allow_html=True,
    )
    st.caption("Étapes du traitement")

    step = st.session_state.current_step
    steps_info = [
        (1, "Importation", "Chargez vos documents"),
        (2, "OCR", "Lecture du texte"),
        (3, "LLM / NLP", "Extraction intelligente"),
        (4, "Résultat", "Informations extraites"),
    ]
    for num, titre, sous_titre in steps_info:
        is_done = step > num
        is_active = step == num
        box_cls = "step-box active" if is_active else "step-box"
        circle_cls = "step-circle done" if is_done else "step-circle"
        badge_txt = "✓" if is_done else str(num)
        st.markdown(
            compact_html(
                f"""
                <div class="{box_cls}">
                    <div class="{circle_cls}">{badge_txt}</div>
                    <div>
                        <div class="step-title">{titre}</div>
                        <div class="step-sub">{sous_titre}</div>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    with st.expander("⚙️ Paramètres (Ollama & OCR)", expanded=False):
        model_name = st.text_input("Modèle Ollama", value="llama3.2")
        langs_choice = st.selectbox("Langues OCR", options=[("fr", "en"), ("ar", "en"), ("en",)], index=0)
        cols_per_row = st.selectbox("Colonnes par ligne", options=[1, 2, 3], index=1)

    st.markdown(
        compact_html(
            """
            <div class="sidebar-footer-card">
                <div style="font-size:1.4rem;">🚀</div>
                <div style="font-weight:700;font-size:0.95rem;color:#fff;margin-top:0.2rem;">Une IA puissante<br>pour vos documents</div>
                <div class="sidebar-pill">Rapide • Précis • Fiable</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

# 5. En-tête principal + badge "En ligne"
st.markdown(
    compact_html('<div class="status-online"><span class="status-dot"></span>En ligne</div><div style="clear:both;"></div>'),
    unsafe_allow_html=True,
)
st.markdown(compact_html('<div class="hero-title">Smart Extractor <span>Document</span></div>'), unsafe_allow_html=True)
st.markdown(
    compact_html('<div class="hero-subtitle">Extrayez automatiquement les informations clés de vos documents grâce à l\'IA</div>'),
    unsafe_allow_html=True,
)

# 6. Zone d'importation multi-documents
st.markdown(
    compact_html(
        """
        <div class="upload-banner">
            <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
                <div style="display:flex;align-items:center;gap:14px;">
                    <div style="font-size:2rem;color:#4f46e5;">☁️</div>
                    <div>
                        <div style="font-weight:700;font-size:1.05rem;color:#0f172a;">Importez vos documents</div>
                        <div style="color:#64748b;font-size:0.88rem;margin-bottom:4px;">Glissez-déposez une ou plusieurs images (JPG, PNG)</div>
                        <span class="fmt-badge">JPG</span><span class="fmt-badge">PNG</span>
                    </div>
                </div>
                <div style="font-size:1.8rem;opacity:0.75;">🪪</div>
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Choisir des fichiers",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files and st.session_state.current_step == 1:
    st.session_state.current_step = 2

# Bouton centré "Start Exécution"
_, btn_col, _ = st.columns([2, 1.4, 2])
with btn_col:
    start_clicked = st.button("▶  Start Exécution", type="primary", use_container_width=True)

# 7. Traitement de tous les documents sélectionnés
if start_clicked:
    if not uploaded_files:
        st.warning("Veuillez d'abord importer au moins une image (JPG ou PNG).")
    else:
        try:
            st.session_state.batch_results = []
            progress_bar = st.progress(0)
            total_docs = len(uploaded_files)

            for idx, file in enumerate(uploaded_files, start=1):
                st.session_state.current_step = 2
                with st.spinner(f"[{idx}/{total_docs}] OCR sur {file.name}..."):
                    file.seek(0)
                    img_bgr = load_image_from_upload(file)
                    cleaned_img = preprocess_image(img_bgr)
                    ocr_txt = extract_ocr_text(cleaned_img, langs=langs_choice)

                st.session_state.current_step = 3
                with st.spinner(f"[{idx}/{total_docs}] Extraction Ollama ({model_name}) sur {file.name}..."):
                    fields = (
                        extract_fields_with_ollama(ocr_txt, model_name=model_name)
                        if ocr_txt
                        else {
                            "nom": None,
                            "prenom": None,
                            "date": None,
                            "numero_document": None,
                            "organisme_emetteur": None,
                        }
                    )

                file.seek(0)
                st.session_state.batch_results.append(
                    {
                        "index": idx,
                        "filename": file.name,
                        "image_bytes": file.getvalue(),
                        "fields": fields,
                    }
                )
                progress_bar.progress(idx / total_docs)

            st.session_state.current_step = 4
            st.rerun()
        except Exception as e:
            st.error(f"Erreur lors de l'exécution : {e}")

# 8. Affichage : pour chaque document, INFORMATIONS AU-DESSUS puis IMAGE EN DESSOUS
if st.session_state.batch_results:
    st.divider()
    results = st.session_state.batch_results

    for start in range(0, len(results), cols_per_row):
        row_items = results[start : start + cols_per_row]
        cols = st.columns(cols_per_row, gap="large")

        for col, item in zip(cols, row_items):
            with col:
                # 1) Informations extraites AU-DESSUS de l'image
                render_fields_card(item["index"], item["filename"], item["fields"])

                # 2) Aperçu du document EN DESSOUS
                st.image(item["image_bytes"], caption=f"Aperçu — {item['filename']}", use_container_width=True)

                # 3) Boutons Télécharger JSON & Réessayer sous le document
                b1, b2 = st.columns(2)
                with b1:
                    st.download_button(
                        label="⬇  Télécharger JSON",
                        data=json.dumps(item["fields"], indent=4, ensure_ascii=False),
                        file_name=f"{item['filename'].rsplit('.', 1)[0]}_extrait.json",
                        mime="application/json",
                        key=f"dl_{item['index']}_{item['filename']}",
                        use_container_width=True,
                    )
                with b2:
                    if st.button(
                        "↻  Réessayer",
                        key=f"retry_{item['index']}_{item['filename']}",
                        use_container_width=True,
                    ):
                        st.session_state.batch_results = []
                        st.session_state.current_step = 1
                        st.rerun()