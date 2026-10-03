import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib import colors

pdf_filename = "Guide_Construction_Projet_Django.pdf"
doc = SimpleDocTemplate(
    pdf_filename,
    pagesize=letter,
    rightMargin=40,
    leftMargin=40,
    topMargin=40,
    bottomMargin=40
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    'DocTitle',
    parent=styles['Heading1'],
    fontSize=20,
    textColor=colors.HexColor('#003366'),
    spaceAfter=15,
    alignment=1
)

h1_style = ParagraphStyle(
    'Heading1_Custom',
    parent=styles['Heading2'],
    fontSize=13,
    textColor=colors.HexColor('#0056b3'),
    spaceBefore=12,
    spaceAfter=6
)

body_style = ParagraphStyle(
    'Body_Custom',
    parent=styles['Normal'],
    fontSize=9.5,
    leading=13.5,
    textColor=colors.HexColor('#222222'),
    spaceAfter=8
)

code_style = ParagraphStyle(
    'Code_Custom',
    parent=styles['Code'],
    fontName='Courier',
    fontSize=7.5,
    leading=10,
    backColor=colors.HexColor('#f8f9fa'),
    borderColor=colors.HexColor('#d0d7de'),
    borderWidth=1,
    borderPadding=6,
    spaceBefore=4,
    spaceAfter=8
)

story = []

# En-tête du document
story.append(Paragraph("<b>GUIDE D'ARCHITECTURE ET D'ÉVOLUTION IA</b>", title_style))
story.append(Paragraph("Plateforme d'Analyse de Données & Réponses Automatiques aux Documents", ParagraphStyle('Sub', alignment=1, fontSize=10, textColor=colors.gray)))
story.append(Spacer(1, 10))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#003366'), spaceAfter=12))

# Sections de base de l'architecture
sections_base = [
    ("1. Origine & Correction de l'Architecture", 
     "Le projet a débuté avec un conflit entre deux moteurs Web (FastAPI et Django), provoquant des erreurs au démarrage.<br/>"
     "<b>Solution :</b> Le projet a été unifié sous l'architecture Django REST standard avec <code>JsonResponse</code> et <code>@csrf_exempt</code>."),

    ("2. Traitement Backend & Graphiques",
     "• <b>Importation en mémoire :</b> Les fichiers (CSV, Excel, PDF) sont lus à la volée via <code>BytesIO</code>.<br/>"
     "• <b>Rendu graphique :</b> <code>Matplotlib</code> et <code>Seaborn</code> génèrent la courbe, convertie en image <b>Base64</b> intégrée au JSON de réponse."),

    ("3. Interface Frontend & Exportation",
     "• <b>Flux :</b> <code>index.html</code> envoie le fichier, enregistre les résultats dans le <code>localStorage</code>, puis redirige vers <code>resultats.html</code>.<br/>"
     "• <b>Mise en page :</b> Grille CSS à 2 colonnes (Résultats textuels à gauche, Visualisation à droite).<br/>"
     "• <b>Export PDF :</b> Intégration de la librairie <code>html2pdf.js</code> pour sauvegarder le rapport au format paysage."),
]

for titre, texte in sections_base:
    story.append(Paragraph(titre, h1_style))
    story.append(Paragraph(texte, body_style))

# Section 4 : Intégration du Moteur PNL / IA
story.append(Paragraph("4. Extension IA : Résolution des questions du document", h1_style))
story.append(Paragraph(
    "Pour que le système réponde automatiquement aux questions posées dans les documents soumis (PDF/Texte), le module <code>nlp_engine.py</code> exploite le modèle <b>GPT-4o-mini</b> via l'API OpenAI :",
    body_style
))

code_ia = """# Fichier : analyser/nlp_engine.py
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def analyse_text(text):
    prompt = f\"\"\"
    Tu es un assistant d'entreprise. Lis le document ci-dessous.
    Identifie toutes les questions posées et réponds-y de manière claire et détaillée en t'appuyant sur les données fournies.

    Document :
    {text}
    \"\"\"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Assistant d'analyse documentaire."},
            {"role": "user", "content": prompt}
        ]
    )

    return {
        "nb_caracteres": len(text),
        "reponses_questions": response.choices[0].message.content
    }"""

# Conversion des sauts de ligne et espaces pour le rendu HTML dans ReportLab
formatted_code_ia = code_ia.replace('\n', '<br/>').replace(' ', '&nbsp;')
story.append(Paragraph(formatted_code_ia, code_style))

# Section 5 : Dépendances
story.append(Paragraph("5. Dépendances système complètes", h1_style))
code_deps = "pip install django pandas pypdf openpyxl matplotlib seaborn django-cors-headers reportlab openai"
story.append(Paragraph(code_deps, code_style))

# Génération finale
doc.build(story)
print(f"Document généré avec succès : {pdf_filename}")

# Ajout dans generer_doc.py
story.append(Paragraph("6. Architecture Multi-Domaines (Marketing, Finance, Industrie, Santé, Géographie)", h1_style))
story.append(Paragraph(
    "La plateforme s'enrichit de modules spécialisés exploitant l'IA analytique et les algorithmes métiers :<br/>"
    "• <b>Marketing :</b> Analyse de sentiments et prédiction de churn.<br/>"
    "• <b>Finance :</b> Analyse de graphes de transactions (NetworkX) et séries temporelles.<br/>"
    "• <b>Industrie :</b> Analyse spectrale vibratoire (FFT) et maintenance prescriptive.<br/>"
    "• <b>Santé :</b> Synthèse médicale et aide au diagnostic.<br/>"
    "• <b>Écologie/Géo :</b> Télédétection multimodale sur images satellites.",
    body_style
))

code_multi = """pip install networkx scipy numpy matplotlib google-genai"""
story.append(Paragraph(code_multi, code_style))