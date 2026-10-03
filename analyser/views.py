import base64
import io
import json
import markdown
import traceback

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.safestring import mark_safe
from .advanced_engine import analyse_image_geographique
# Imports des modules locaux
from .data_loader import load_file
from .analyse import analyse_dataframe, generer_table_powerbi
from .nlp_engine import analyse_text
from .visualise import (
    generate_interactive_plot,
    generate_summary_plots,
    get_plot_data_json,
)
from .advanced_engine import (
    analyser_marketing_sentiment,
    analyse_finance_reseau,
    analyser_vibrations_industrie,
    analyse_scanner_medical_vision,
    segmenter_zones_scanner,
    analyser_statistiques_sante,
    diagnostic_medical_assistance,
    analyse_image_geographique,
    delimiter_zones_teledection,
    lire_geotiff,
    analyse_dataframe,
    generer_table_powerbi,
    get_plot_data_json,
    format_stats_to_html,
)


def format_stats_to_html(stats_dict):
    """
    Génère une structure HTML propre et moderne pour les statistiques.
    Gère aussi bien si stats_numeriques est un dictionnaire qu’une liste.
    """
    if not stats_dict or not isinstance(stats_dict, dict):
        return "<p>Aucune statistique disponible.</p>"

    html = []

    # 1. Section KPI (Cartes d’informations globales)
    html.append('<div class="kpi-conteneur">')
    
    if 'total_lignes' in stats_dict:
        html.append(f'''
            <div class="kpi-card">
                <div class="kpi-title">Total Lignes</div>
                <div class="kpi-value">{stats_dict["total_lignes"]}</div>
            </div>
        ''')
    if 'total_colonnes' in stats_dict:
        html.append(f'''
            <div class="kpi-card">
                <div class="kpi-title">Total Colonnes</div>
                <div class="kpi-value">{stats_dict["total_colonnes"]}</div>
            </div>
        ''')
    if 'valeurs_manquantes' in stats_dict:
        html.append(f'''
            <div class="kpi-card">
                <div class="kpi-title">Valeurs Manquantes</div>
                <div class="kpi-value">{stats_dict["valeurs_manquantes"]}</div>
            </div>
        ''')
    
    html.append('</div>')

    # 2. Section Tableau des statistiques numériques
    stats_num = stats_dict.get('stats_numeriques', {})
    if stats_num:
        html.append('<h3>Statistiques Descriptives</h3>')
        html.append('<table class="table-stats">')
        html.append('''
            <thead>
                <tr>
                    <th>Colonne</th>
                    <th class="text-end">Moyenne</th>
                    <th class="text-end">Min</th>
                    <th class="text-end">Max</th>
                    <th class="text-end">Somme</th>
                    <th class="text-end">Écart-type</th>
                    <th class="text-end">Médiane</th>
                    <th class="text-end">CV (%)</th>
                    <th class="text-end">Quartiles</th>
                </tr>
            </thead>
            <tbody>
        ''')

        items_to_iterate = []
        if isinstance(stats_num, dict):
            items_to_iterate = stats_num.items()
        elif isinstance(stats_num, list):
            items_to_iterate = [
                (item.get('colonne', f'Colonne {i+1}'), item) if isinstance(item, dict) else (f'Colonne {i+1}', {})
                for i, item in enumerate(stats_num)
            ]

        for col, values in items_to_iterate:
            if not isinstance(values, dict):
                continue

            moyenne = f"{values.get('moyenne', 0):,.2f}" if isinstance(values.get('moyenne'), (int, float)) else values.get('moyenne', '-')
            minimum = f"{values.get('min', 0):,.2f}" if isinstance(values.get('min'), (int, float)) else values.get('min', '-')
            maximum = f"{values.get('max', 0):,.2f}" if isinstance(values.get('max'), (int, float)) else values.get('max', '-')
            somme = f"{values.get('somme', 0):,.2f}" if isinstance(values.get('somme'), (int, float)) else values.get('somme', '-')
            ecart_type = f"{values.get('ecart_type', 0):,.2f}" if isinstance(values.get('ecart_type'), (int, float)) else values.get('ecart_type', '-')
            mediane = f"{values.get('mediane', 0):,.2f}" if isinstance(values.get('mediane'), (int, float)) else values.get('mediane', '-')
            cv = f"{values.get('Coefficient_de_Variation', 0):,.2f}" if isinstance(values.get('Coefficient_de_Variation'), (int, float)) else values.get('Coefficient_de_Variation', '-')
            quartiles = f"{values.get('quartiles', 0):,.2f}" if isinstance(values.get('quartiles'), (int, float)) else values.get('quartiles', '-')

            html.append(f'''
                <tr>
                    <td><strong>{col}</strong></td>
                    <td class="text-end">{moyenne}</td>
                    <td class="text-end">{minimum}</td>
                    <td class="text-end">{maximum}</td>
                    <td class="text-end">{somme}</td>
                    <td class="text-end">{ecart_type}</td>
                    <td class="text-end">{mediane}</td>
                    <td class="text-end">{cv}</td>
                    <td class="text-end">{quartiles}</td>
                </tr>
            ''')

        html.append('</tbody></table>')

    return "".join(html)


def index(request):
    """Affiche la page d'accueil avec le formulaire (GET)."""
    return render(request, 'index.html')


@csrf_exempt
def api_analyse_multidomaine(request):
    """API pour l'analyse spécifique par domaine."""
    if request.method == 'POST':
        domaine = request.POST.get('domaine')
        
        if domaine == 'marketing':
            texte = request.POST.get('texte', '')
            res = analyser_marketing_sentiment(texte)
            return JsonResponse({"status": "success", "domaine": "marketing", "resultat": res})
            
        elif domaine == 'sante':
            rapport = request.POST.get('texte', '')
            res = diagnostic_medical_assistance(rapport)
            return JsonResponse({"status": "success", "domaine": "sante", "resultat": res})
            
        elif domaine == 'industrie':
            freq = json.loads(request.POST.get('frequences', '[]'))
            amp = json.loads(request.POST.get('amplitudes', '[]'))
            res = analyser_vibrations_industrie(freq, amp)
            return JsonResponse({"status": "success", "domaine": "industrie", "resultat": res})

        return JsonResponse({"status": "error", "message": "Domaine non reconnu"}, status=400)


def analyser_view(request):
    """Vue pour traiter un formulaire de texte simple."""
    if request.method == "POST":
        sujet = request.POST.get("sujet", "").strip()
        if not sujet:
            return render(request, "index.html", {"error": "Veuillez saisir un sujet."})
        
        resultat = analyser_marketing_sentiment(sujet)
        return render(request, "index.html", {"resultat": mark_safe(markdown.markdown(resultat))})

    return render(request, "index.html")


def analyse_fichier(request):
    """Traite le fichier (données, texte ou image) et le sujet envoyés (POST)."""
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return render(request, 'index.html', {'erreur': 'Aucun fichier fourni.'})

        uploaded_file = request.FILES['file']
        sujet = request.POST.get('sujet', '').strip()

        try:
            # Chargement du fichier via le data_loader
            data_type, data = load_file(uploaded_file)

            resultats = None
            graphique_base64 = None
            graphique_interactif = None
            table_html = None
            plot_json = None
            is_image = False
            image_delimitee_data = None
            rapport_teledection = None

            # Traitement selon le type de fichier
            if data_type == "dataframe":
                resultats = analyse_dataframe(data, sujet=sujet) if sujet else analyse_dataframe(data)
                graphique_base64 = generate_summary_plots(data)
                graphique_interactif = generate_interactive_plot(data)

                table_html = generer_table_powerbi(data)
                if table_html and isinstance(table_html, str):
                    table_html = table_html.replace('class="dataframe"', 'class="table-stats"')
                    if 'class="table-stats"' not in table_html:
                        table_html = table_html.replace('<table>', '<table class="table-stats">')

                plot_json = get_plot_data_json(data)

            elif data_type == "texte":
                resultats = analyse_image_geographique(data, sujet=prompt_final) 

            elif data_type == "image":
                consigne_systeme = (
                    "Analyse cette image d'un point de vue géographique, géologique ou pédologique. "
                    "S'il s'agit d'une coupe de sol ou d'un profil dans une fosse, identifie précisément les différents horizons pédologiques. "
                    "S'il s'agit d'une vue satellite ou d'un paysage, identifie l'occupation du sol et le relief."
                )
                prompt_final = f"{consigne_systeme}\n\nSujet spécifique : {sujet}" if sujet else consigne_systeme

                resultats = analyse_image_geographique(data, sujet=prompt_final)
                is_image = True

                # Délimitation des zones
                try:
                    res_teledection = delimiter_zones_teledection(data)

                    if isinstance(res_teledection, (tuple, list)):
                        image_annotee = res_teledection[0]
                        if len(res_teledection) > 1:
                            rapport_teledection = res_teledection[1]
                    else:
                        image_annotee = res_teledection

                    if image_annotee.mode != 'RGB':
                        image_annotee = image_annotee.convert('RGB')

                    buffer = io.BytesIO()
                    image_annotee.save(buffer, format="JPEG")
                    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    image_delimitee_data = f"data:image/jpeg;base64,{img_str}"

                except Exception as img_err:
                    print(f"Erreur lors de la délimitation de l'image : {img_err}")

            # Conversion HTML sécurisée pour Django
            if resultats:
                if isinstance(resultats, dict):
                    if "stats_numeriques" in resultats or "total_lignes" in resultats:
                        resultats_html = format_stats_to_html(resultats)
                    else:
                        texte_markdown = (
                            resultats.get('analyse') or 
                            resultats.get('texte') or 
                            resultats.get('resultat') or 
                            str(resultats)
                        )
                        resultats_html = markdown.markdown(
                            texte_markdown, 
                            extensions=['tables', 'fenced_code', 'nl2br']
                        )
                else:
                    resultats_html = markdown.markdown(
                        str(resultats), 
                        extensions=['tables', 'fenced_code', 'nl2br']
                    )
            else:
                resultats_html = "<p>Aucun résultat n'a été généré.</p>"

            context = {
                "fichier": uploaded_file.name,
                "sujet": sujet,
                "resultats": mark_safe(resultats_html),
                "graphique": graphique_base64,
                "graphique_interactif": mark_safe(graphique_interactif) if graphique_interactif else None,
                "table_powerbi": mark_safe(table_html) if table_html else None,
                "plot_data_json": plot_json,
                "is_image": is_image,
                "image_delimitee": image_delimitee_data,
                "rapport_teledection": rapport_teledection,
            }
            return render(request, 'resultats.html', context)

        except Exception as e:
            traceback.print_exc()
            return render(request, 'index.html', {'erreur': f"Erreur lors de l'analyse : {str(e)}"})

    return render(request, 'index.html')