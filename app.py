import streamlit as st
import pandas as pd
import plotly.express as px
import re
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
import pydeck as pdk
from rapidfuzz import process, fuzz as rapidfuzz_fuzz, fuzz as rapidfuzz_fuzz_ratio
import requests
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch



# Fix for PyTorch and Streamlit conflict
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# Load environment variables
load_dotenv()

# Initialize models
# print("Loading models...")

# HuggingFace Inference API setup
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

FLY_API_URL = "https://ivedras-topic-api.fly.dev/predict"

TOPIC_LABELS = [
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

# Add this mapping at the top, after TOPIC_LABELS
URGENCY_LABELS = ["Alta", "Média", "Baixa"]
URGENCY_LABEL_MAP = {0: "Alta", 1: "Média", 2: "Baixa", "LABEL_0": "Alta", "LABEL_1": "Média", "LABEL_2": "Baixa", "alta": "Alta", "media": "Média", "baixa": "Baixa"}

# Update the urgency API URL
iVEDRAS_URGENCY_API_URL = "https://ivedras-urgency-api.fly.dev/predict"

# After TOPIC_LABELS
TOPIC_LABEL_MAP = {label: label for label in TOPIC_LABELS}

# Load topic and urgency models/tokenizers at startup
TOPIC_MODEL_REPO = "valterjpcaldeira/iVedrasQueixas"  # Change to your actual repo if different
URGENCY_MODEL_REPO = "valterjpcaldeira/iVedrasUrgencia"
HF_TOKEN = os.getenv("HF_API_TOKEN")  # Optional, for private models

topic_model = AutoModelForSequenceClassification.from_pretrained(TOPIC_MODEL_REPO, use_auth_token=HF_TOKEN)
topic_tokenizer = AutoTokenizer.from_pretrained(TOPIC_MODEL_REPO, use_auth_token=HF_TOKEN)

urgency_model = AutoModelForSequenceClassification.from_pretrained(URGENCY_MODEL_REPO, use_auth_token=HF_TOKEN)
urgency_tokenizer = AutoTokenizer.from_pretrained(URGENCY_MODEL_REPO, use_auth_token=HF_TOKEN)

# Topic classification local inference
def run_topic_classification(text):
    inputs = topic_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = topic_model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        score = probs[0, pred].item()
    return {"label_id": pred, "confidence": score}

# Urgency classification local inference
def run_urgency_classification(text):
    inputs = urgency_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = urgency_model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        score = probs[0, pred].item()
    return {"label_id": pred, "confidence": score}

# NER API call (Hugging Face Inference API)
def run_ner(text):
    model_id = "lfcc/bert-portuguese-ner"
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    response = requests.post(api_url, headers=HF_HEADERS, json={"inputs": text})
    response.raise_for_status()
    return response.json()

def extract_addresses(text):
    entities = run_ner(text)
    extracted_addresses = []
    # Extract postal codes in the format 2560-374
    import re
    postal_codes = re.findall(r'\b\d{4}-\d{3}\b', text)
    extracted_addresses.extend(postal_codes)
    for entity in entities:
        # HuggingFace NER returns a list of dicts with 'entity_group' and 'word'
        if isinstance(entity, dict) and entity.get('entity_group') in ['Local', 'ORG', 'PER', 'LOC', 'GPE']:
            extracted_addresses.append(entity['word'])
        # Some models return a list of lists
        elif isinstance(entity, list):
            for ent in entity:
                if ent.get('entity_group') in ['Local', 'ORG', 'PER', 'LOC', 'GPE']:
                    extracted_addresses.append(ent['word'])
    # Remove duplicates
    all_extracted = list(set(extracted_addresses))
    return all_extracted

def classificar_mensagem(texto):
    result = run_topic_classification(texto)
    label_id = result["label_id"]
    confidence = result["confidence"]
    label = TOPIC_LABELS[label_id] if 0 <= label_id < len(TOPIC_LABELS) else str(label_id)
    return label, confidence

def classificar_urgencia(texto):
    result = run_urgency_classification(texto)
    # New API returns a dict with 'label_id' and 'confidence'
    if isinstance(result, dict) and "label_id" in result:
        label_id = result["label_id"]
        confidence = result["confidence"]
        label = URGENCY_LABEL_MAP.get(label_id, str(label_id))
        probas = {label: confidence}
        return label, probas
    return 'LABEL_0', {}

def normalize_address(addr):
    addr = addr.lower()
    addr = re.sub(r'[^\w\s]', '', addr)  # remove punctuation
    addr = re.sub(r'\s+', ' ', addr).strip()  # remove excess whitespace
    return addr

def get_address_from_mongodb(normalized_address, threshold=40):
    from rapidfuzz import fuzz
    import re
    client = get_mongodb_client()
    if client:
        try:
            db = client['complaints_db']
            collection = db['addresses']
            address_docs = list(collection.find({}))
            # Check for format1 '2560046' (7 digits, no dash)
            match = re.search(r'\b\d{7}\b', normalized_address)
            if match:
                code = match.group(0)
                for doc in address_docs:
                    address_str = doc.get('address', '')
                    if code in address_str:
                        # Found exact match in address string
                        return doc['address'], doc['latitude'], doc['longitude'], 100
            # Fallback to fuzzy matching
            best_score = 0
            best_doc = None
            for doc in address_docs:
                db_norm = doc.get('normalized_address', '')
                score = fuzz.ratio(normalized_address, db_norm)
                if score > best_score:
                    best_score = score
                    best_doc = doc
            if best_doc and best_score >= threshold:
                return best_doc['address'], best_doc['latitude'], best_doc['longitude'], best_score
        except Exception as e:
            st.error(f"Erro ao obter morada da base de dados MongoDB: {str(e)}")
        finally:
            client.close()
    return None, None, None, 0

def get_coordinates(location, city="Torres Vedras", country="Portugal"):
    address, lat, lon, score = get_address_from_mongodb(normalize_address(location))
    if lat is not None and lon is not None:
        return lat, lon, address, score
    return None, None, None, 0

def get_mongodb_client():
    try:
        uri = os.getenv("MONGODB_URI")
        if not uri:
            st.error("MongoDB URI not found in environment variables")
            return None
        client = MongoClient(uri)
        # Test the connection
        client.admin.command('ping')
        return client
    except Exception as e:
        st.error(f"Error connecting to MongoDB Atlas: {str(e)}")
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
                print(address_data)

                # Get coordinates and verify location
                found = False
                best_score = 0
                best_lat, best_lon, best_address = None, None, None
                if address_data:
                    for address in address_data:
                        lat, lon, adresse_extracted, score = get_coordinates(address)
                        if score > best_score:
                            best_score = score
                            best_lat, best_lon, best_address = lat, lon, adresse_extracted
                    # Optionally, try the combined address as well
                    combined_address = " ".join(address_data)
                    lat, lon, adresse_extracted, score = get_coordinates(combined_address)
                    if score > best_score:
                        best_score = score
                        best_lat, best_lon, best_address = lat, lon, adresse_extracted
                    if best_score >= 30 and best_lat is not None and best_lon is not None:
                        found = True
                        lat, lon, adresse_extracted = best_lat, best_lon, best_address
                    else:
                        error_message = "❌ A morada tem de ser Torres Vedras, tente outra vez."
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
                    topic_display = TOPIC_LABEL_MAP.get(topic, topic)

                    # Urgência
                    urgencia, probas = classificar_urgencia(text_input)
                    urgencia_display = URGENCY_LABEL_MAP.get(urgencia, urgencia)

                    # Resultados
                    st.subheader("📌 Resultados da Análise")
                    st.markdown(f"**Localização extraída:** {adresse_extracted}")
                    if isinstance(lat, (float, int)) and isinstance(lon, (float, int)):
                        st.markdown(f"**Coordenadas:** Latitude = {lat:.6f}, Longitude = {lon:.6f}")
                    else:
                        st.markdown(f"**Coordenadas:** Latitude = {lat}, Longitude = {lon}")
                    st.markdown(f"**Nível de urgência:** `{urgencia_display}`")
                    st.markdown(f"**Tópico Detetado:** `{topic_display}`")

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
                        'topic': topic_display,
                        'topic_confidence': float(topic_score),
                        'urgency': urgencia_display,
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
