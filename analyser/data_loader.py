import pandas as pd
from pypdf import PdfReader
from PIL import Image
import io

def load_file(file):
    filename = file.name
    content = file.read()
    
    if filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(content))
        return "dataframe", df

    elif filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
      
        image = Image.open(io.BytesIO(content))
        return "image", image

    elif filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(io.BytesIO(content))
        return "dataframe", df
    
    elif filename.endswith('.json'):
        df = pd.read_json(io.BytesIO(content))
        return "dataframe", df
    
    elif filename.endswith('.pdf'):
        pdf = PdfReader(io.BytesIO(content))
        text = "".join([page.extract_text() or "" for page in pdf.pages])
        return "text", text
    
    else:
        raise ValueError("Format de fichier non supporté.")