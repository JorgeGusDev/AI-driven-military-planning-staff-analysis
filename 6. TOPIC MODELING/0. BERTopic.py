import os
import glob
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction import text
import matplotlib.pyplot as plt

# Descargar stopwords de NLTK si no están
nltk.download("stopwords")

# Carpeta donde están los archivos
CARPETA = r"C:\Users\LENOVO\OneDrive - Grupo Educad\7. IDEA_LAB\7. SNA & SA\4. PAPER_SNA_SA\1. Data_Technics & Outputs\2. NLP_SENTIMENTAL ANALYSIS\NLP_v2\Resultados_NLP_v2"

# Obtener todos los archivos Excel
archivos = glob.glob(os.path.join(CARPETA, "*.xlsx"))

# Stopwords en español + inglés + personalizadas
stopwords_es = stopwords.words("spanish")
stopwords_en = list(text.ENGLISH_STOP_WORDS)

extra_stopwords = [
    "hola","buenos","buenas","tarde","noche","gracias","favor","ok","vale","aja","jaja","jeje","ajá","eh","mmm","xd","jajaja",
    "sr","sra","señor","señora","señores","jefe","compañeros","compañeras","grupo","equipo","gente","personas",
    "sí","no","ya","aquí","allí","ahí","entonces","también","además","bueno","pues","este","osea","ose","ehh","aja",
    "día","días","semana","mes","año","tiempo","momento","cosas","algo","nada","todo","todos","todas","algún","algunos","algunas",
    "cahue","video","omitido","eliminó","videos","rdo","mensaje","11","noches","haz click","unirte reunion","google","si","jerry",
    "jpg","haz","fotos","mas","dia","galindo","gandulias","peñaranda","crl","12 12","12","cta","llamada","celular 13","chavez",
    "aler","mayor","mayores","audios","moreno","audio","olivera solari","algonasi", "eliminaste", "solari", "capulian", "10", "chávez",
    "tarjeta contacto", "contacto omitida", "abrazo", "rodrígues","alex",
    "unirte", "clic vínculo", 
]

stopwords_custom = list(set(stopwords_es + stopwords_en + extra_stopwords))

# Vectorizador
vectorizer = CountVectorizer(
    stop_words=stopwords_custom,
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.95
)

# Iterar sobre cada archivo
for archivo in archivos:
    print(f"\nProcesando archivo: {archivo}")
    df = pd.read_excel(archivo, sheet_name=0)
    
    # Solo la columna 'phrase'
    docs = df["phrase"].dropna().astype(str).tolist()
    if len(docs) == 0:
        print("Archivo vacío, se omite.")
        continue

    # Crear modelo BERTopic
    topic_model = BERTopic(
        language="multilingual",
        vectorizer_model=vectorizer,
        min_topic_size=10,
        calculate_probabilities=True
    )
    
    topics, probs = topic_model.fit_transform(docs)
    
    # Reducir tópicos
    topic_model = topic_model.reduce_topics(docs, nr_topics=2)
    
    # Info de tópicos
    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info.Topic != -1]  # excluir -1
    print(topic_info.head(3))
    
    # Imprimir análisis de Beta (palabras y su peso por tópico)
    print("\nAnálisis de Beta por tópico:")
    for topic_num in topic_info.Topic.tolist():
        print(f"\nTópico {topic_num}:")
        beta_words = topic_model.get_topic(topic_num)
        for word, weight in beta_words:
            if weight > 0:
                print(f"{word}: {weight:.4f}")

    
    # Plot tipo LDA (barchart)
    fig = topic_model.visualize_barchart(top_n_topics=1, n_words=10)
    fig.update_layout(title_text=os.path.basename(archivo))  # título = nombre del archivo
    fig.show()
