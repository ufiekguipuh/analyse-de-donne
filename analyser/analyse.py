import pandas as pd 


def analyse_dataframe(df):
  """Calcule des statistiques descriptives et les renvoie sous forme de dictionnaire structuré."""
  if df is None or df.empty:
    return {'erreur': 'Le fichier ne contient pas de données.'}

  total_lignes = len(df)
  total_colonnes = len(df.columns)
  valeurs_manquantes = int(df.isnull().sum().sum())

  numeric_cols = df.select_dtypes(include=['number'])

  stats_numeriques = []
  for col in numeric_cols.columns:
    stats_numeriques.append({
        'colonne': col,
        'moyenne': round(df[col].mean(), 3),
        'min': round(df[col].min(), 3),
        'max': round(df[col].max(), 3),
        'somme': round(df[col].sum(), 3),
        'ecart_type': round(df[col].std(),3),
        'mediane': round(df[col].median(),3),
        'Coefficient_de_Variation': round(df[col].std() / df[col].mean(), 3),
        'quartiles': round(df[col].quantile(),3),
    })

  return {
      'total_lignes': total_lignes,  
      'total_colonnes': total_colonnes,
      'valeurs_manquantes': valeurs_manquantes,
      'stats_numeriques': stats_numeriques,
  }

def generer_table_powerbi(df):
    """
    Génère un tableau HTML propre sans colonnes 'Unnamed' et prêt pour DataTables.
    """
    if isinstance(df, pd.DataFrame) and not df.empty:
        # 1. Suppression des colonnes d'index fantômes (Unnamed: 0, etc.)
        df_clean = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # 2. Génération du tableau HTML sans l'index Pandas
        html_table = df_clean.head(100).to_html(
            classes='table-powerbi display style-powerbi', 
            index=False, 
            border=0
        )
        return html_table
    return None