import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderQuotaExceeded
import time
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv
import torch
from fuzzywuzzy import fuzz
from folium.plugins import HeatMap
import pydeck as pdk
from opencage.geocoder import OpenCageGeocode
from difflib import SequenceMatcher
from rapidfuzz import process, fuzz
import joblib
import joblib  # or pickle


# Fix for PyTorch and Streamlit conflict
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# Load environment variables
load_dotenv()

# Initialize models
# print("Loading models...")

# Load spaCy
nlp = spacy.load("pt_core_news_lg")

# Load NER model with explicit device mapping
ner_pipeline = pipeline(
    "ner",
    model="lfcc/bert-portuguese-ner",
    aggregation_strategy="simple",
    device=-1  # Force CPU usage
)

# Carregar modelo, tokenizer e LabelEncoder para classificação de tópicos
TOPIC_MODEL_PATH = "valterjpcaldeira/iVedrasQueixas"
TOPIC_ENCODER_PATH = "valterjpcaldeira/iVedrasQueixas"
topic_model = AutoModelForSequenceClassification.from_pretrained(TOPIC_MODEL_PATH)
topic_tokenizer = AutoTokenizer.from_pretrained(TOPIC_MODEL_PATH)

# Carregar modelo, tokenizer e LabelEncoder para classificação de urgência
URGENCY_MODEL_PATH = "valterjpcaldeira/iVedrasUrgencia"
URGENCY_ENCODER_PATH = "valterjpcaldeira/iVedrasUrgencia"
urgency_model = AutoModelForSequenceClassification.from_pretrained(URGENCY_MODEL_PATH)
urgency_tokenizer = AutoTokenizer.from_pretrained(URGENCY_MODEL_PATH)

def classificar_mensagem(texto):
    inputs = topic_tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = topic_model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
    predicted_class_id = logits.argmax().item()
    confidence = probs[0, predicted_class_id].item()
    label = topic_model.config.id2label[predicted_class_id]
    return label, confidence

def classificar_urgencia(texto):
    inputs = urgency_tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = urgency_model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
    predicted_class_id = logits.argmax().item()
    confidence = probs[0, predicted_class_id].item()
    label = urgency_model.config.id2label[predicted_class_id]
    probas = {urgency_model.config.id2label[i]: float(probs[0, i].item()) for i in range(probs.shape[1])}
    return label, probas

def normalize_address(addr):
    addr = addr.lower()
    addr = re.sub(r'[^\w\s]', '', addr)  # remove punctuation
    addr = re.sub(r'\s+', ' ', addr).strip()  # remove excess whitespace
    return addr

try:
    addresses_df = pd.read_csv("moradas_torres_vedras.csv")
    addresses_df = addresses_df.dropna(subset=['address', 'latitude', 'longitude'])
    addresses_df['normalized_address'] = addresses_df['address'].apply(normalize_address)
except Exception as e:
    addresses_df = None

# Predefined topics for classification
topicos_queixas = [
    "Limpeza e Resíduos",
    "Infraestruturas e Obras",
    "Trânsito e Mobilidade",
    "Áreas Verdes e Espaços Públicos",
    "Água e Saneamento",
    "Animais e Ambiente",
    "Serviços Sociais e Comunitários",
    "Segurança e Ordem Pública",
    "Comércio e Atividades Económicas",
    "Outros"
]

def calculate_similarity(str1, str2):
    str1 = str(str1) if pd.notna(str1) else ""
    str2 = str(str2) if pd.notna(str2) else ""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def find_similar_address(extracted_address, addresses_df, threshold=0.4):
    best_match = None
    best_score = 0
    best_lat = None
    best_lon = None
    for idx, row in addresses_df.iterrows():
        db_address = row['address']
        if pd.isna(db_address):
            continue
        similarity = calculate_similarity(extracted_address, db_address)
        if similarity > best_score:
            best_score = similarity
            best_match = db_address
            best_lat = row.get('latitude', None)
            best_lon = row.get('longitude', None)
    if best_score >= threshold:
        return best_match, best_score, best_lat, best_lon
    else:
        return None, best_score, None, None

def match_address(location, addresses_df, threshold=80):
    if addresses_df is None or addresses_df.empty:
        return None, None, None
    required_cols = {'address', 'latitude', 'longitude', 'normalized_address'}
    if not required_cols.issubset(addresses_df.columns):
        return None, None, None
    postal_code_match = re.fullmatch(r"\d{4}-\d{3}", str(location).strip())
    if postal_code_match:
        mask = addresses_df['address'].astype(str).str.contains(location)
        matches = addresses_df[mask]
        if not matches.empty:
            row = matches.iloc[0]
            return row['address'], row['latitude'], row['longitude']
        else:
            return None, None, None
    extracted_norm = normalize_address(location)
    address_list = addresses_df['normalized_address'].tolist()
    if not address_list:
        return None, None, None
    try:
        result = process.extractOne(
            extracted_norm,
            address_list,
            scorer=fuzz.token_sort_ratio
        )
        if result is None:
            return None, None, None
        match, score, index = result
    except Exception:
        return None, None, None
    if score >= threshold:
        try:
            row = addresses_df.iloc[index]
            return row['address'], row['latitude'], row['longitude']
        except Exception:
            return None, None, None
    else:
        return None, None, None

def get_coordinates_from_csv(location):
    if addresses_df is not None:
        best_match, lat, lon = match_address(location, addresses_df,55)
        if isinstance(lat, str):
            lat = lat.replace('GPS: ','')
        if best_match is not None and lat is not None and lon is not None:
            try:
                lat_f = float(lat)
                lon_f = float(lon)
            except Exception:
                lat_f, lon_f = lat, lon
            return lat_f, lon_f, best_match
    return None, None, None

def get_coordinates_from_opencage(location, city="Torres Vedras", country="Portugal"):
    key = os.getenv("OPENCAGE_API_KEY")
    if not key:
        raise ValueError("OpenCage API key not found. Please set OPENCAGE_API_KEY in your .env file.")
    geocoder = OpenCageGeocode(key)
    full_location = f"{location}, {city}, {country}"
    try:
        results = geocoder.geocode(full_location, no_annotations='1', limit=1, language='pt')
        if results and len(results):
            lat = results[0]['geometry']['lat']
            lon = results[0]['geometry']['lng']
            address_name = results[0]['formatted']
            return lat, lon, address_name
        else:
            return None, None, None
    except Exception:
        return None, None, None

def get_coordinates(location, city="Torres Vedras", country="Portugal"):
    lat, lon, best_match = get_coordinates_from_csv(location)
    if lat is not None and lon is not None:
        return lat, lon, best_match
    return get_coordinates_from_opencage(location, city, country)
import streamlit as st
def get_mongodb_client():
    try:
        uri = st.secrets["MONGODB_URI"]
        client = MongoClient(uri)
        return client
    except Exception as e:
        st.error(f"Erro ao ligar à base de dados MongoDB: {str(e)}")
        return None

def save_complaint_to_mongodb(complaint_data):
    client = get_mongodb_client()
    if client:
        try:
            db = client['complaints_db']
            collection = db['complaints']
            complaint_data['timestamp'] = datetime.utcnow()
            result = collection.insert_one(complaint_data)
            return result.inserted_id
        except Exception as e:
            st.error(f"Erro ao guardar na base de dados MongoDB: {str(e)}")
        finally:
            client.close()
    return None

def get_complaints_from_mongodb():
    client = get_mongodb_client()
    if client:
        try:
            db = client['complaints_db']
            collection = db['complaints']
            complaints = list(collection.find({}, {'_id': 0}))
            return pd.DataFrame(complaints)
        except Exception as e:
            st.error(f"Erro ao obter dados da base de dados MongoDB: {str(e)}")
        finally:
            client.close()
    return pd.DataFrame()

# Configuração da página
st.set_page_config(
    page_title="Sistema de Gestão de Queixas",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for light theme and consistent colors
st.markdown("""
    <style>
        /* Force light theme */
        .stApp {
            background-color: #ffffff;
            color: #00aae9;
        }
        .main {
            background-color: #ffffff;
        }
        .block-container {
            background-color: #ffffff;
            padding: 2rem 2rem;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #00aae9;
            color: #ffffff;
            border-radius: 10px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: bold;
            transition: background 0.2s;
        }
        .stButton>button:hover {
            background-color: #ec008c;
            color: #ffffff;
            border: none;
        }
        
        /* Input styling */
        .stTextInput>div>div>input {
            background-color: #ffffff;
            border: 2px solid #00aae9;
            border-radius: 5px;
            color: #00aae9;
        }
        
        /* Sidebar styling */
        div[data-testid="stSidebar"] {
            background-color: #fdee00;
        }
        div[data-testid="stSidebar"] .block-container {
            background-color: #fdee00;
        }
        
        /* Text colors */
        h1, h2, h3, h4, h5, h6 {
            color: #00aae9;
        }
        h1 {
            color: #ed1c24;
        }
        h4, h5, h6 {
            color: #ec008c;
        }
        
        /* Success/Error messages */
        .stSuccess {
            background-color: #fdee00;
            color: #00aae9;
        }
        .stError {
            background-color: #ed1c24;
            color: #ffffff;
        }
        .stWarning {
            background-color: #ec008c;
            color: #ffffff;
        }
        
        /* Map container */
        .stMap {
            border: 2px solid #00aae9;
            border-radius: 5px;
        }
        
        /* Dataframe styling */
        .stDataFrame {
            border: 2px solid #00aae9;
            border-radius: 5px;
        }
        
        /* Custom scrollbars */
        ::-webkit-scrollbar {
            width: 8px;
            background: #fdee00;
        }
        ::-webkit-scrollbar-thumb {
            background: #00aae9;
            border-radius: 4px;
        }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.title("iVedras")
st.markdown('<h4 style="margin-top: -1em; color: #666;">A tua voz na Iniciativa Vedras para uma cidade melhor.</h4>', unsafe_allow_html=True)

# Criar abas para separar as funcionalidades
tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 Analisador de Queixas"])

# Carregar o dataset do MongoDB
df = get_complaints_from_mongodb()

# Aba do Dashboard
with tab1:
    st.header("Visualização de Queixas de Cidadãos")
    
    # Select boxes para filtros dentro de um expander fechado por padrão
    with st.sidebar.expander("Filtros", expanded=False):
        if not df.empty and 'topic' in df.columns:
            topic_options = ['Todos'] + list(df['topic'].unique())
        else:
            topic_options = ['Todos']
        if not df.empty and 'urgency' in df.columns:
            urgency_options = ['Todos'] + list(df['urgency'].unique())
        else:
            urgency_options = ['Todos']
        selected_topic = st.selectbox("Selecione o Tópico", options=topic_options)
        selected_urgency = st.selectbox("Selecione o Nível de Urgência", options=urgency_options)

    # Filtrar o dataset
    filtered_df = df.copy()
    if selected_topic != 'Todos':
        filtered_df = filtered_df[filtered_df['topic'] == selected_topic]
    if selected_urgency != 'Todos':
        filtered_df = filtered_df[filtered_df['urgency'] == selected_urgency]

    # Gráfico de barras empilhadas
    st.subheader("Contagem de Queixas por Tópico e Urgência")
    pivot_df = filtered_df.groupby(['topic', 'urgency']).size().unstack(fill_value=0)

    # Portuguese labels and custom urgency order/colors
    urgency_order = ['Alta', 'Média', 'Baixa']
    urgency_colors = {
        'Alta': '#FF3B30',   # Vermelho
        'Média': '#FF9500',  # Laranja
        'Baixa': '#34C759'   # Verde
    }
    # Ensure all columns are present in the right order
    for urg in urgency_order:
        if urg not in pivot_df.columns:
            pivot_df[urg] = 0
    pivot_df = pivot_df[urgency_order]

    fig = px.bar(
        pivot_df,
        x=pivot_df.index,
        y=urgency_order,
        title="Queixas por Tópico e Nível de Urgência",
        labels={'x': 'Tópico', 'value': 'Número de Queixas', 'variable': 'Urgência'},
        barmode='stack',
        color_discrete_map=urgency_colors
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        legend_title_text='Urgência',
        legend=dict(
            itemsizing='constant',
            title_font=dict(size=14),
            font=dict(size=13)
        )
    )
    st.plotly_chart(fig)

    # Criar heatmap dinâmico com pydeck
    st.subheader("Mapa das Queixas")
    if not filtered_df.empty:
        # Remove rows with missing coordinates
        map_df = filtered_df.dropna(subset=['latitude', 'longitude'])
        view_state = pdk.ViewState(
            latitude=map_df['latitude'].mean(),
            longitude=map_df['longitude'].mean(),
            zoom=10,
            pitch=0
        )
        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            data=map_df,
            get_position='[longitude, latitude]',
            aggregation=pdk.types.String("MEAN"),
            get_weight=1,
            radiusPixels=60,
        )
        r = pdk.Deck(
            layers=[heatmap_layer],
            initial_view_state=view_state,
            tooltip={"text": "Latitude: {latitude}, Longitude: {longitude}"},
            map_style="light"
        )
        st.pydeck_chart(r, use_container_width=True)
    else:
        st.info("Não há dados de queixas para mostrar no mapa.")

    # Tabela com todas as mensagens e tópicos
    st.subheader("Lista de Queixas")
    st.dataframe(filtered_df[['problem', 'topic', 'urgency', 'timestamp']])

# Aba do Analisador de Queixas
with tab2:
    st.header("📍 Analisador Inteligente de Queixas Urbanas")
    st.write("Analisa queixas de cidadãos: deteta tópicos, urgência e extrai localizações.")

    def extract_addresses(text):
        # Extract entities from the NER pipeline

        entities = ner_pipeline(text)
        extracted_addresses = []

        # Extract postal codes in the format 2560-374
        import re
        postal_codes = re.findall(r'\b\d{4}-\d{3}\b', text)
        extracted_addresses.extend(postal_codes)

        for entity in entities:
            if entity['entity_group'] in ['Local', 'ORG', 'PER']:
                extracted_addresses.append(entity['word'])

        # Also extract with spaCy for LOC and GPE
        doc = nlp(text)
        spacy_addresses = [ent.text for ent in doc.ents if ent.label_ in ["LOC", "GPE"]]

        # Combine all extracted addresses, removing duplicates
        all_extracted = list(set(extracted_addresses + spacy_addresses))
        return all_extracted

    # Interface do analisador
    text_input = st.text_area("Cole aqui a queixa de um cidadão:", height=200, 
                             value="Há um buraco enorme na estrada da Rua Monte do Rossio, Vila Facaia, Ramalhal. Esta muito preigoso para os carros.")

    show_spinner = False
    analysis_success = False

    if st.button("Analisar Queixa"):
        # Clear previous messages
        st.session_state['error_message'] = ""
        st.session_state['warning_message'] = ""
        st.session_state['analysis_complete'] = False
        show_spinner = True
        error_flag = False
        warning_flag = False
        warning_message = ""
        error_message = ""

        if text_input:
            with st.spinner("A analisar..."):
                # Localização (NER with BERTimbau)
                address_data = extract_addresses(text_input)

                # Get coordinates and verify location
                found = False
                adresse_extracted = None
                lat = None
                lon = None

                if address_data:
                    # Try each extracted address in CSV first
                    for address in address_data:
                        lat, lon, adresse_extracted = get_coordinates_from_csv(address)
                        if lat is not None and lon is not None:
                            found = True
                            break
                    # If not found in CSV, try combined address in CSV
                    if not found:
                        combined_address = " ".join(address_data)
                        lat, lon, adresse_extracted = get_coordinates_from_csv(combined_address)
                        if lat is not None and lon is not None:
                            found = True

                    if not found or lat is None or lon is None:
                        error_message = "❌ Não foi possível obter morada na mensagem. Por favor, forneça um endereço válido em Torres Vedras, adicione o codigo postal."
                        error_flag = True
                else:
                    error_message = "❌ Não foi possível obter morada na mensagem. Por favor, forneça um endereço válido em Torres Vedras, adicione o codigo postal."
                    error_flag = True

                if found and not error_flag:
                    # For similarity, use the best available address string
                    if adresse_extracted is None:
                        combined_address = " ".join(address_data)
                        adresse_extracted = combined_address

                    similarity_ratio = fuzz.ratio(adresse_extracted.lower(), (" ".join(address_data)).lower())
                    if similarity_ratio < 40:
                        warning_message = "❌ A morada extraída é muito diferente da morada fornecida. Por favor, verifique se a morada está correta."
                        warning_flag = True

                    # Tópico (Classificação com modelo próprio)
                    topic, topic_score = classificar_mensagem(text_input)

                    # Urgência
                    urgencia, probas = classificar_urgencia(text_input)

                    # Resultados
                    st.subheader("📌 Resultados da Análise")
                    st.markdown(f"**Localização extraída:** {adresse_extracted}")
                    if isinstance(lat, (float, int)) and isinstance(lon, (float, int)):
                        st.markdown(f"**Coordenadas:** Latitude = {lat:.6f}, Longitude = {lon:.6f}")
                    else:
                        st.markdown(f"**Coordenadas:** Latitude = {lat}, Longitude = {lon}")
                    st.markdown(f"**Nível de urgência:** `{urgencia}`")
                    st.markdown(f"**Tópico Detetado:** `{topic}`")

                    # Exibir mapa
                    st.subheader("📍 Localização no Mapa")
                    map_data = pd.DataFrame({"lat": [lat], "lon": [lon]})
                    # Remove rows with missing or invalid coordinates
                    map_data = map_data.dropna(subset=["lat", "lon"])
                    map_data = map_data[(map_data["lat"].apply(lambda x: isinstance(x, (float, int)))) & \
                                        (map_data["lon"].apply(lambda x: isinstance(x, (float, int))))]
                    if not map_data.empty:
                        st.map(map_data, zoom=12)
                    else:
                        warning_message = "Não foi possível mostrar o mapa porque as coordenadas são inválidas ou estão em falta."
                        warning_flag = True

                    # Save to MongoDB
                    complaint_data = {
                        'problem': text_input,
                        'location': adresse_extracted,  # Use the extracted address directly
                        'latitude': lat,
                        'longitude': lon,
                        'topic': topic,
                        'topic_confidence': float(topic_score),
                        'urgency': urgencia,
                        'urgency_probabilities': probas,
                        'timestamp': datetime.utcnow()
                    }
                    if save_complaint_to_mongodb(complaint_data):
                        st.session_state['error_message'] = ""
                        st.session_state['warning_message'] = ""
                        st.session_state['analysis_complete'] = True
                        analysis_success = True
                    else:
                        error_message = "❌ Erro ao registar a queixa na base de dados."
                        error_flag = True
                elif not error_flag:
                    warning_message = "Não foi possível obter a morada, tente dar mais detalhe ou coloque outra."
                    warning_flag = True
        else:
            error_message = "Por favor cole uma queixa para analisar."
            st.session_state['analysis_complete'] = False
            error_flag = True

        # Set messages after spinner
        if error_flag:
            st.session_state['error_message'] = error_message
            st.session_state['warning_message'] = ""
            st.session_state['analysis_complete'] = False
        elif warning_flag:
            st.session_state['error_message'] = ""
            st.session_state['warning_message'] = warning_message
            st.session_state['analysis_complete'] = False
        elif analysis_success:
            st.session_state['error_message'] = ""
            st.session_state['warning_message'] = ""
            st.session_state['analysis_complete'] = True

    # Display messages in priority order
    if st.session_state.get('error_message'):
        st.error(st.session_state['error_message'])
    elif st.session_state.get('warning_message'):
        st.warning(st.session_state['warning_message'])
    elif st.session_state.get('analysis_complete'):
        st.success("✅ Queixa registada com sucesso na base de dados!")
