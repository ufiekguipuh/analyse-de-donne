import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import plotly.express as px
import pandas as pd
import json


def generate_summary_plots(df):
    """Génère un graphique simple et renvoie l'image encodée en base64"""
    num_cols = df.select_dtypes(include=['number']).columns
    
    if len(num_cols) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    col = num_cols[0]
    sns.histplot(df[col], kde=True, ax=ax, color='skyblue')
    ax.set_title(f"Distribution de : {col}")
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

def generate_interactive_plot(df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

   
    df_clean = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    if df_clean.empty:
        return None

    
    numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()

    fig = None

    # Cas A : Texte + Numérique (ex: Ventes par Catégorie) -> Graphique en barres
    if categorical_cols and numeric_cols:
        cat = categorical_cols[0]
        num = numeric_cols[0]
        # Aggrégation Power BI (Somme par catégorie des 10 premiers)
        df_grouped = df_clean.groupby(cat, as_index=False)[num].sum().head(10)
        fig = px.bar(
            df_grouped, 
            x=cat, 
            y=num, 
            title=f"Total de {num} par {cat}",
            template="plotly_white",
            color=cat
        )

    # Cas B : Au moins 2 colonnes numériques -> Graphique de dispersion (Scatter)
    elif len(numeric_cols) >= 2:
        fig = px.scatter(
            df_clean, 
            x=numeric_cols[0], 
            y=numeric_cols[1], 
            title=f"Relation entre {numeric_cols[0]} et {numeric_cols[1]}",
            template="plotly_white"
        )

    # Cas C : Une seule colonne numérique valide -> Histogramme de distribution
    elif len(numeric_cols) == 1:
        fig = px.histogram(
            df_clean, 
            x=numeric_cols[0], 
            title=f"Distribution de {numeric_cols[0]}",
            template="plotly_white"
        )

    # 3. Exportation au format HTML pour Django
    if fig:
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    return None

def get_plot_data_json(df):
  """Exporte la structure complète du DataFrame et ses enregistrements en JSON."""
  if not isinstance(df, pd.DataFrame) or df.empty:
    return None

  df_clean = df.loc[:, ~df.columns.str.contains('^Unnamed')].copy()

  # Détection automatique des colonnes numériques et textuelles
  numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
  all_cols = df_clean.columns.tolist()

  # Conversion du dataframe en dictionnaires
  records = df_clean.to_dict(orient='records')

  data_payload = {
      'columns': all_cols,
      'numeric_cols': numeric_cols,
      'records': records,
  }

  return json.dumps(data_payload, default=str)