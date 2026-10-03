import ollama

def analyse_text(text):
    prompt = f"""
    Tu es un assistant d'entreprise expert. Lis attentivement le document ci-dessous.
    Identifie toutes les questions posées et réponds-y de manière claire et détaillée en t'appuyant sur les données du texte.

    Document :
    {text}
    """

    try:
        # Appel du modèle local Ollama (gratuit, aucune clé API)
        response = ollama.chat(
            model='llama3.2:1b',
            messages=[
                {'role': 'system', 'content': 'Tu es un assistant d\'analyse documentaire.'},
                {'role': 'user', 'content': prompt}
            ]
        )
        reponse_ia = response['message']['content']
    except Exception as e:
        reponse_ia = f"Erreur avec Ollama (assurez-vous que l'application Ollama tourne en arrière-plan) : {str(e)}"

    return {
        "nb_caracteres": len(text),
        "reponses_questions": reponse_ia
    }