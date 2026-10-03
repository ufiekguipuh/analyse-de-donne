import pandas as pd

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how='all')
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    object_cols = df.select_dtypes(include=['object']).columns
    df[object_cols] = df[object_cols].fillna("Inconnu")
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    
    return df