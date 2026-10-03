# Fichier : analyser/advanced_engine.py
import os
import time
import base64
import colorsys
import io
import json
import traceback
import pandas as pd
import numpy as np
import networkx as nx
import cv2
import markdown
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from google import genai
from google.genai import types
from django.shortcuts import render
from groq import Groq
from google import genai

import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import segmentation_models_pytorch as smp

# Chargement rasterio pour SIG
try:
    import rasterio
    from rasterio.plot import reshape_as_image
except ImportError:
    rasterio = None

# =====================================================
# CONFIGURATION CLÉS API & ENVIRONNEMENT
# =====================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Chargement global unique du modèle pour optimiser la mémoire RAM/GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS = DeepLabV3_ResNet50_Weights.DEFAULT
MODEL_IA = deeplabv3_resnet50(weights=WEIGHTS).to(DEVICE).eval()
PREPROCESS = WEIGHTS.transforms()


def generer_ai(prompt, retries=3):
    """Interroge le modèle Llama via Groq avec gestion des tentatives."""
    if not client:
        return "Erreur : La clé GROQ_API_KEY n'est pas configurée."

    for tentative in range(retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="qwen/qwen3.8-27b",
                max_tokens=800
            )
            return chat_completion.choices[0].message.content

        except Exception as e:
            print(f"Tentative Groq {tentative + 1}/{retries} : {e}")
            if "429" in str(e):
                time.sleep(2 ** (tentative + 1))  # Backoff exponentiel
                continue
            break

    return "Service IA temporairement indisponible."


# =====================================================
# 1. MARKETING & FINANCE & INDUSTRIE & SANTE
# =====================================================

def analyser_marketing_sentiment(texte, marque_produit="Produit/Marque"):
    """
    Analyse le sentiment, extrait les entités et identifie les tendances marketing
    à partir d'un texte ou commentaire client.
    """
    prompt = f"""
    Effectue une analyse marketing et de sentiment approfondie sur ce texte concernant '{marque_produit}' :
    
    Texte :
    \"\"\"{texte}\"\"\"
    
    Fournis le résultat structuré sous les axes suivants :
    1. Sentiment Général : (Positif, Neutre, Négatif avec un score estimé sur 100).
    2. Points Forts / Compliments détectés.
    3. Points de Douleur / Critiques / Axes d'amélioration.
    4. Mots-clés et Intention d'achat (Élevée, Moyenne, Faible).
    5. Recommandation d'action rapide pour le service client ou marketing.
    """
    return generer_ai(prompt)


def analyse_finance_reseau(df_transactions):
    """
    Analyse un réseau de transactions financières à partir d'un DataFrame.
    Colonnes requises : 'source', 'cible', 'montant'
    Calcul de la centralité d'intermédiation (betweenness centrality) avec NetworkX.
    """
    G = nx.from_pandas_edgelist(df_transactions, source='source', target='cible', edge_attr='montant', create_using=nx.DiGraph())
    centralite = nx.betweenness_centrality(G)
    compte_cles = sorted(centralite.items(), key=lambda x: x[1], reverse=True)[:5]
    
    prompt = f"""
    Voici les comptes les plus centraux (fort niveau d'intermédiation) d'un réseau financier : {compte_cles}
    Fournis une analyse financière détaillée :
    1. Analyse du risque de fraude ou de blanchiment potentiel.
    2. Identification des nœuds critiques (intermédiaires majeurs).
    3. Recommandations d'audit et de contrôle d'exposition.
    """
    analyse = generer_ai(prompt)
    return {
        "nombre_noeuds": G.number_of_nodes(),
        "nombre_liens": G.number_of_edges(),
        "analyse_risque": analyse
    }

# Nom de fonction harmonisé pour l'import dans views.py
def analyser_vibrations_industrie(frequences, amplitudes):
    """
    Analyse les signaux vibratoires issus de capteurs industriels (moteurs, pompes, turbines).
    """
    frequences_np = np.array(frequences)
    amplitudes_np = np.array(amplitudes)

    rms = np.sqrt(np.mean(amplitudes_np**2))
    max_amp = np.max(amplitudes_np)
    freq_pic = frequences_np[np.argmax(amplitudes_np)]
    
    prompt = f"""
        Analyse vibratoire industrielle d'un équipement rotatif :
        - Valeur efficace (RMS) : {rms:.2f} mm/s
        - Amplitude maximale : {max_amp:.2f} mm/s
        - Fréquence critique dominante : {freq_pic:.1f} Hz
        
        Fournis un rapport technique complet comprenant :
        1. Diagnostic probable du dysfonctionnement (ex: balourd, alignement, usure de roulement, jeu mécanique).
        2. Niveau de sévérité du risque (Faible, Modéré, Élevé, Critique).
        3. Recommandations d'intervention de maintenance préventive/corrective.
    """
    diagnostic = generer_ai(prompt)
    return {
        "rms": float(rms),
        "amplitude_max": float(max_amp),
        "frequence_critique_hz": float(freq_pic),
        "prescription_maintenance": diagnostic
    }

# Alias pour compatibilité descendante
analyse_vibrations_industrie = analyser_vibrations_industrie


# ======================================================
# MODULE 3 : SANTE & MEDECINE
# =====================================================

def analyse_scanner_medical_vision(image_pil, type_examen="Scanner / Radiographie"):
    """
    Analyse visuelle multimodale d'un scanner, d'une IRM ou d'une radiographie via Gemini.
    """
    if not gemini_client:
        return "Erreur : La configuration Gemini API n'est pas exacte."
    try:
        if hasattr(image_pil, 'convert') and image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        prompt = f"""
        Tu es un assistant virtuel spécialisé en imagerie médicale.
        Analyse ce cliché médical ({type_examen}) :
        1. Description des structures anatomiques visibles et anomalies éventuelles.
        2. Observations sur la densité, le contraste ou les lésions identifiables.
        3. Hypothèses d'orientation diagnostique à valider par un radiologue.

        Avertissement obligatoire : Ce compte-rendu est une aide à l'analyse et doit être validé par un médecin.
        """
        response = gemini_client.models.generate_content(
            model='gemini-3.8-flash',
            contents=[prompt, image_pil]
        )
        return response.text
    except Exception as e:
        return f"Erreur lors de l'analyse de l'image médicale : {str(e)}"


def segmenter_zones_scanner(image_pil):
    """
    Segmentation et délimitation des zones d'intérêt anatomiques/tissulaires.
    """
    if image_pil.mode in ("RGBA", "P"):
        image_pil = image_pil.convert("RGB")
    largeur_orig, hauteur_orig = image_pil.size
    surface_totale = largeur_orig * hauteur_orig
    img_np = np.array(image_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    zones_identifiees = []
    
    try:
        input_tensor = PREPROCESS(image_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = MODEL_IA(input_tensor)['out'][0]
        prediction = output.argmax(0).byte().cpu().numpy()
        prediction = cv2.resize(prediction, (largeur_orig, hauteur_orig), interpolation=cv2.INTER_NEAREST)

        classes_uniques = np.unique(prediction)
        for class_id in classes_uniques:
            if class_id == 0:
                continue
            mask_classe = (prediction == class_id).astype(np.uint8)
            pixels_count = np.sum(mask_classe)
            pourcentage = round((pixels_count / surface_totale) * 100, 1)
            if pourcentage > 0.3:
                zones_identifiees.append({
                    "id": int(class_id),
                    "nom": f"Zone tissulaire {class_id}",
                    "pct": pourcentage,
                    "mask": mask_classe
                })
    except Exception as e:
        print(f"Erreur lors de la segmentation IA du scanner : {e}")

    carte_annotee = (img_bgr * 0.6).astype(np.uint8)
    palette = generer_palette_couleurs(len(zones_identifiees))

    for idx, z in enumerate(zones_identifiees):
        z["couleur"] = palette[idx]
        masque_bool = (z["mask"] == 1)
        couleur_bgr = np.array(z["couleur"], dtype=np.uint8)

        zone_cible = carte_annotee[masque_bool]
        zone_couleur = np.tile(couleur_bgr, (np.sum(masque_bool), 1))
        
        carte_annotee[masque_bool] = cv2.addWeighted(
            zone_cible, 0.3, zone_couleur, 0.7, 0
        )

    image_rgb = cv2.cvtColor(carte_annotee, cv2.COLOR_BGR2RGB)
    pil_resultat = Image.fromarray(image_rgb)
    rapport = f"Segmentation médicale terminée : {len(zones_identifiees)} zones de densité/tissus isolées."
    return pil_resultat, rapport


def analyser_statistiques_sante(df_patients):
    """ 
    Calcule des indicateurs statistiques clés sur un jeu de données médicales.
    """
    stats_descriptives = df_patients.describe(include='all').to_dict()
    valeurs_manquantes = df_patients.isnull().sum().to_dict()
    
    prompt = f"""
    Voici le résumé statistique d'une cohorte de patients / données cliniques :
    {stats_descriptives}

    Nombre de valeurs manquantes par variable :
    {valeurs_manquantes}

    Fournis une analyse statistique médicale :
    1. Résumé des moyennes, écarts-types et distributions des constantes vitales/biologiques.
    2. Identification des valeurs aberrantes ou des facteurs de risque prédominants.
    3. Recommandations pour la stratification des patients ou le suivi clinique.
    """
    analyse_llm = generer_ai(prompt)
    return {
        "nb_patients": len(df_patients),
        "nb_variables": len(df_patients.columns),
        "statistiques": stats_descriptives,
        "rapport_statistique_medical": analyse_llm,
    }


def diagnostic_medical_assistance(symptomes_ou_rapport):
    """
    Synthèse clinique et aide au diagnostic textuel à partir d'un rapport ou de symptômes.
    """
    prompt = f"""
    Analyse le texte ou le dossier médical suivant :
    "{symptomes_ou_rapport}"

    Fournis un document synthétique structuré :
    1. Résumé clinique et antécédents marquants.
    2. Hypothèses diagnostiques principales et différentielles.
    3. Examens complémentaires conseillés (imagerie, biologie, explorations fonctionnelles).

    Avertissement : Outil d'aide à la décision clinique - ne remplace pas l'avis d'un médecin.
    """
    return generer_ai(prompt)


# ======================================================
# MODULE 4 : GEOGRAPHIE & TELEDETECTION
# =====================================================   

def generer_palette_couleurs(nb_classes):
    """
    Génère une palette de couleurs BGR distinctes basées sur l'espace HSV.
    """
    couleur_bgr = []
    for i in range(nb_classes):
        teinte = i / float(max(nb_classes, 1))
        r, g, b = colorsys.hsv_to_rgb(teinte, 0.85, 0.95)
        couleur_bgr.append((int(b * 255), int(g * 255), int(r * 255)))
    return couleur_bgr


def analyse_image_geographique(image_data, sujet="Analyse de carte et télédétection"):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Modèles multimodaux généraux inclus dans le Plan Gratuit (Free Tier)
    modeles_gemini = [
        'gemini-2.5-flash',
        'gemini-2.5-pro',
        'gemini-1.5-flash-latest'
    ]
    
    dernier_erreur = None

    for nom_modele in modeles_gemini:
        try:
            prompt_systeme = (
                f"Vous êtes un expert en télédétection et SIG. "
                f"Analysez cette image concernant : {sujet}."
            )
            
            response = client.models.generate_content(
                model=nom_modele,
                contents=[image_data, prompt_systeme]
            )
            return response.text

        except errors.APIError as e:
            dernier_erreur = e
            print(f"[Gemini Fallback] Modèle {nom_modele} indisponible ({e.code}). Tentative suivante...")
            # On continue sur le modèle suivant pour TOUTES les erreurs API (404, 429, 503)
            continue
        except Exception as e:
            dernier_erreur = e
            if any(code in str(e) for code in ["404", "429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"]):
                continue
            raise e

    return f"Avertissement Quota : Les modèles Gemini ont atteint leur limite journalière (429). {dernier_erreur}"

def delimiter_zones_teledection(image_pil):
    """
    Délimite les zones sur une image satellite ou coupe pédologique.
    """
    if image_pil.mode in ("RGBA", "P"):
        image_pil = image_pil.convert("RGB")

    largeur_orig, hauteur_orig = image_pil.size
    surface_totale = largeur_orig * hauteur_orig
    img_np = np.array(image_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elements_detectes = []

    try:
        input_tensor = PREPROCESS(image_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = MODEL_IA(input_tensor)['out'][0]
        prediction = output.argmax(0).byte().cpu().numpy()
        prediction = cv2.resize(prediction, (largeur_orig, hauteur_orig), interpolation=cv2.INTER_NEAREST)
        classes_uniques = np.unique(prediction)

        for class_id in classes_uniques:
            if class_id == 0:
                continue
            mask_classe = (prediction == class_id).astype(np.uint8)
            pixels_count = np.sum(mask_classe)
            pourcentage = round((pixels_count / surface_totale) * 100, 1)

            if pourcentage > 0.5:
                nom_trouve = WEIGHTS.meta["categories"][class_id]
                elements_detectes.append({
                    "id": class_id,
                    "nom": nom_trouve.capitalize(),
                    "pct": pourcentage,
                    "mask": mask_classe
                })
    except Exception as e:
        print(f"Erreur segmentation DeepLab: {e}")

    # Fallback K-Means si DeepLab ne renvoie rien
    if not elements_detectes:
        try:
            k = 3
            data_pixels = img_np.reshape((-1, 3)).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, _ = cv2.kmeans(data_pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            labels = labels.reshape((hauteur_orig, largeur_orig))

            for idx in range(k):
                mask_k = (labels == idx).astype(np.uint8)
                pct = round((np.sum(mask_k) / surface_totale) * 100, 1)
                elements_detectes.append({
                    "id": idx + 1,
                    "nom": f"Zone homogène {idx + 1}",
                    "pct": pct,
                    "mask": mask_k
                })
        except Exception as e:
            print(f"Erreur segmentation K-Means: {e}")

    # Tracé des surbrillances et contours
    nb_elements = len(elements_detectes)
    palette = generer_palette_couleurs(nb_elements)
    carte_sig = (img_bgr * 0.55).astype(np.uint8)

    for idx, el in enumerate(elements_detectes):
        el["couleur"] = palette[idx]
        masque_bool = (el["mask"] == 1)
        couleur_bgr = np.array(el["couleur"], dtype=np.uint8)

        carte_sig[masque_bool] = cv2.addWeighted(
            carte_sig[masque_bool], 0.3,
            np.tile(couleur_bgr, (np.sum(masque_bool), 1)), 0.7, 0
        )

        contours, _ = cv2.findContours(el["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > (surface_totale * 0.001):
                epsilon = 0.003 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                cv2.drawContours(carte_sig, [approx], -1, (255, 255, 255), 1)

    # Bande pour la légende
    hauteur_legende = 60
    image_finale = cv2.copyMakeBorder(carte_sig, 0, hauteur_legende, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))

    if nb_elements > 0:
        largeur_colonne = int(largeur_orig / nb_elements)
        y_pos = hauteur_orig + 35

        for idx, el in enumerate(elements_detectes):
            x_pos = 15 + (idx * largeur_colonne)
            cv2.rectangle(image_finale, (x_pos, y_pos - 12), (x_pos + 18, y_pos + 8), el["couleur"], -1)
            cv2.rectangle(image_finale, (x_pos, y_pos - 12), (x_pos + 18, y_pos + 8), (255, 255, 255), 1)
            texte = f"{el['nom']}: {el['pct']}%"
            cv2.putText(image_finale, texte, (x_pos + 24, y_pos + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    image_rgb = cv2.cvtColor(image_finale, cv2.COLOR_BGR2RGB)
    pil_result = Image.fromarray(image_rgb)
    description_sig = f"Délimitation effectuée : {nb_elements} zones spatiales identifiées."

    return pil_result, description_sig


def lire_geotiff(chemin_ou_fichier):
    """
    Lecture et conversion d'un fichier GeoTIFF en image PIL et métadonnées SIG.
    """
    if rasterio is None:
        raise ImportError("La bibliothèque 'rasterio' n'est pas installée.")

    with rasterio.open(chemin_ou_fichier) as src:
        if src.count >= 3:
            data = src.read([1, 2, 3])
            img_array = reshape_as_image(data)
        else:
            data = src.read(1)
            img_array = np.stack([data] * 3, axis=-1)

        if img_array.dtype != np.uint8:
            img_min, img_max = img_array.min(), img_array.max()
            if img_max > img_min:
                img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_array = np.zeros_like(img_array, dtype=np.uint8)

        meta = {
            "driver": src.driver,
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "crs": str(src.crs),
            "bounds": src.bounds
        }

        image_pil = Image.fromarray(img_array)
        return image_pil, meta


# =====================================================
# MODULE 5 : STATISTIQUES GÉNÉRALES & POWER BI
# =====================================================

def analyse_dataframe(df, sujet=None):
    """
    Réalise une analyse statistique descriptive d'un DataFrame.
    """
    dimensions = f"Dimensions du jeu de données : {len(df)} lignes et {len(df.columns)} colonnes.\n"
    colonnes_info = df.dtypes.astype(str).to_dict()
    valeurs_manquantes = df.isnull().sum().to_dict()
    
    stats_numeriques = df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else "Aucune variable numérique"
    
    prompt = f"""
    Effectue une analyse décisionnelle approfondie de ce jeu de données :
    
    1. Structure : {dimensions}
    2. Colonnes et Types : {colonnes_info}
    3. Valeurs manquantes : {valeurs_manquantes}
    4. Résumé statistique numérique : {stats_numeriques}
    
    {f"Sujet d'étude spécifique : {sujet}" if sujet else ""}
    
    Livre un rapport d'analyse structuré :
    - Synthèse globale et qualité des données.
    - Principales tendances, corrélations et constats clés.
    - Recommandations d'actions opérationnelles / business.
    """
    return generer_ai(prompt)


def generer_table_powerbi(df, n_rows=10):
    """
    Génère un composant HTML style Power BI.
    """
    table_html = df.head(n_rows).to_html(
        classes="table table-striped table-hover table-bordered table-stats",
        index=False,
        border=0
    )
    return table_html


def get_plot_data_json(df):
    """
    Exporte la structure des données et statistiques au format JSON.
    """
    df_num = df.select_dtypes(include=[np.number]).iloc[:, :5]
    
    if df_num.empty:
        return "{}"

    data_export = {
        "colonnes": list(df_num.columns),
        "moyennes": df_num.mean().round(2).to_dict(),
        "medianes": df_num.median().round(2).to_dict(),
        "ecart_types": df_num.std().round(2).to_dict(),
        "min": df_num.min().to_dict(),
        "max": df_num.max().to_dict()
    }
    
    return json.dumps(data_export, ensure_ascii=False)


def format_stats_to_html(stats_dict):
    """
    Mise en forme des dictionnaires sous forme de blocs HTML.
    """
    html_output = "<div class='stats-container'>"
    for key, val in stats_dict.items():
        html_output += f"<div class='stat-card'><strong>{key}</strong> : {val}</div>"
    html_output += "</div>"
    return html_output