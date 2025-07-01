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
from streamlit_folium import st_folium
import folium
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import certifi
from streamlit_js_eval import streamlit_js_eval


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


TOPIC_LABELS = [
    "Animais e Ambiente",
    "Comércio e Atividades Económicas",
    "Infraestruturas e Obras",
    "Limpeza e Resíduos",
    "Outros",
    "Segurança e Ordem Pública",
    "Serviços Sociais e Comunitários",
    "Trânsito e Mobilidade",
    "Água e Saneamento",
    "Áreas Verdes e Espaços Públicos"
]

# Add this mapping at the top, after TOPIC_LABELS
URGENCY_LABEL_MAP = {0: "Não Urgente", 1: "Urgente"}


# After TOPIC_LABELS
TOPIC_LABEL_MAP = {label: label for label in TOPIC_LABELS}

# Load topic and urgency models/tokenizers at startup
TOPIC_MODEL_REPO = "valterjpcaldeira/iVedrasQueixas"  # Change to your actual repo if different
URGENCY_MODEL_REPO = "valterjpcaldeira/iVedrasUrgencia"
HF_TOKEN = os.getenv("HF_API_TOKEN")  # Optional, for private models

topic_model = AutoModelForSequenceClassification.from_pretrained(
    TOPIC_MODEL_REPO,
    use_auth_token=HF_TOKEN,
    device_map=None,  # Force CPU
    torch_dtype=torch.float32
)
topic_tokenizer = AutoTokenizer.from_pretrained(
    TOPIC_MODEL_REPO,
    use_auth_token=HF_TOKEN
)

urgency_model = AutoModelForSequenceClassification.from_pretrained(
    URGENCY_MODEL_REPO,
    use_auth_token=HF_TOKEN,
    device_map=None,  # Force CPU
    torch_dtype=torch.float32
)
urgency_tokenizer = AutoTokenizer.from_pretrained(
    URGENCY_MODEL_REPO,
    use_auth_token=HF_TOKEN
)

print(f"Topic model device: {next(topic_model.parameters()).device}")
print(f"Urgency model device: {next(urgency_model.parameters()).device}")

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

def classificar_mensagem(texto):
    result = run_topic_classification(texto)
    label_id = result["label_id"]
    confidence = result["confidence"]
    print(label_id)
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

def get_address_from_mongodb(normalized_address, threshold=30):
    from rapidfuzz import fuzz
    import re
    print(normalized_address)
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
                    print("---")
                    print(best_doc)
                    print(best_score)
                    print("---")
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
        client = MongoClient(uri, tlsCAFile=certifi.where())
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
    page_title="iVedras",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for app-like, mobile-friendly UI (keep colors, improve layout, touch, and hide Streamlit UI)
st.markdown('''
    <style>
        /* Hide Streamlit default UI */
        #MainMenu, footer, header {visibility: hidden;}
        /* App container */
        .stApp {
            padding: 0 !important;
            background: #fff;
        }
        /* Card-style containers */
        .app-card {
            background: #f9f9f9;
            border-radius: 18px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
            padding: 1.5rem 1.2rem;
            margin-bottom: 1.5rem;
        }
        /* Large, rounded buttons */
        .stButton>button {
            font-size: 1.2rem;
            border-radius: 12px;
            padding: 0.8rem 1.5rem;
            font-weight: 600;
            margin: 0.5rem 0;
        }
        /* Large text inputs */
        .stTextInput>div>div>input, .stTextArea textarea {
            font-size: 1.1rem;
            border-radius: 10px;
            padding: 0.7rem 1rem;
        }
        /* Larger selectboxes */
        .stSelectbox>div>div {
            font-size: 1.1rem;
        }
        /* Larger font for headers and body */
        h1, h2, h3, h4, h5, h6 {
            font-size: 2.1rem !important;
        }
        .stMarkdown, .stDataFrame, .stTable, .stText, .stAlert {
            font-size: 1.1rem;
        }
        /* Sticky submit button for mobile */
        @media (max-width: 600px) {
            .sticky-submit {
                position: fixed;
                left: 0; right: 0; bottom: 0;
                z-index: 1000;
                background: #fff;
                padding: 1rem 0.5rem 1.2rem 0.5rem;
                box-shadow: 0 -2px 12px rgba(0,0,0,0.08);
                text-align: center;
            }
            .stApp {padding-bottom: 90px !important;}
        }
        /* Card spacing for mobile */
        @media (max-width: 600px) {
            .app-card {padding: 1rem 0.5rem;}
        }
    </style>
''', unsafe_allow_html=True)

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
    
    # Move filters to main page (remove sidebar and expander)
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

    # --- Move Map to Top ---
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

    # --- Urgency Trends ---
    st.subheader("Tendência de Urgência das Queixas ao Longo do Tempo")
    if not filtered_df.empty and 'urgency' in filtered_df.columns and 'timestamp' in filtered_df.columns:
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(filtered_df['timestamp']):
            filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
        # Group by week and urgency
        time_group = filtered_df.groupby([pd.Grouper(key='timestamp', freq='W'), 'urgency']).size().unstack(fill_value=0)
        time_group = time_group[['Urgente', 'Não Urgente']] if 'Urgente' in time_group.columns and 'Não Urgente' in time_group.columns else time_group
        fig_trend = px.line(
            time_group,
            x=time_group.index,
            y=time_group.columns,
            labels={'value': 'Nº de Queixas', 'timestamp': 'Semana', 'urgency': 'Urgência'},
            title="Evolução Semanal das Queixas por Urgência"
        )
        fig_trend.update_layout(
            xaxis_title="Semana",
            yaxis_title="Nº de Queixas",
            legend_title_text='Urgência',
            legend=dict(itemsizing='constant', title_font=dict(size=14), font=dict(size=13)),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Não há dados suficientes para mostrar a tendência de urgência.")

    # --- Bar Chart to Bottom ---
    st.subheader("Contagem de Queixas por Tópico e Urgência")
    pivot_df = filtered_df.groupby(['topic', 'urgency']).size().unstack(fill_value=0)
    # Portuguese labels and custom urgency order/colors
    urgency_order = ['Urgente', 'Não Urgente']
    urgency_colors = {
        'Urgente': '#FF3B30',   # Vermelho
        'Não Urgente': '#34C759'   # Verde
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

    # Tabela com todas as mensagens e tópicos
    st.subheader("Lista de Queixas")
    st.dataframe(filtered_df[['problem', 'topic', 'urgency', 'timestamp']])

# Aba do Analisador de Queixas
with tab2:
    st.header("📍 Analisador Inteligente de Queixas Urbanas")
    st.write("Analisa queixas de cidadãos: deteta tópicos, urgência e extrai localizações.")

    # Initialize all required session state variables for the wizard
    if 'show_map_modal' not in st.session_state:
        st.session_state['show_map_modal'] = False
    if 'manual_lat' not in st.session_state:
        st.session_state['manual_lat'] = None
    if 'manual_lon' not in st.session_state:
        st.session_state['manual_lon'] = None
    if 'location_permission_checked' not in st.session_state:
        st.session_state['location_permission_checked'] = False
    if 'step' not in st.session_state:
        st.session_state['step'] = 1
    if 'complaint_input' not in st.session_state:
        st.session_state['complaint_input'] = ""
    if 'last_suggestion_length' not in st.session_state:
        st.session_state['last_suggestion_length'] = 0
    if 'last_suggestion_topic' not in st.session_state:
        st.session_state['last_suggestion_topic'] = ''
    if 'last_suggestion_urgency' not in st.session_state:
        st.session_state['last_suggestion_urgency'] = ''
    if 'submit_queixa' not in st.session_state:
        st.session_state['submit_queixa'] = False

    # Multi-step wizard for complaint analyzer
    if 'step' not in st.session_state:
        st.session_state['step'] = 1

    # Step 1: Select Location
    if st.session_state['step'] == 1:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.header("Passo 1: Selecione a Localização")
        st.markdown("<div style='margin-bottom:1em;'><b>Selecione a localização do problema no mapa (clique para marcar).</b></div>", unsafe_allow_html=True)
        # Use folium map with click to select location
        default_location = [39.0917, -9.2589]
        if st.session_state['manual_lat'] is not None and st.session_state['manual_lon'] is not None:
            default_location = [st.session_state['manual_lat'], st.session_state['manual_lon']]
        m = folium.Map(location=default_location, zoom_start=16)
        # Add marker if location is set
        if st.session_state['manual_lat'] is not None and st.session_state['manual_lon'] is not None:
            folium.Marker(location=default_location).add_to(m)
        output = st_folium(m, width=500, height=400)
        # Update location on map click
        if output and output.get('last_clicked'):
            lat, lon = output['last_clicked']['lat'], output['last_clicked']['lng']
            st.session_state['manual_lat'] = lat
            st.session_state['manual_lon'] = lon
            st.success(f"Localização selecionada: Latitude = {lat:.6f}, Longitude = {lon:.6f}")
        elif st.session_state['manual_lat'] is not None and st.session_state['manual_lon'] is not None:
            st.success(f"Localização selecionada: Latitude = {st.session_state['manual_lat']:.6f}, Longitude = {st.session_state['manual_lon']:.6f}")
        # Next button right below the map
        if st.session_state['manual_lat'] is not None and st.session_state['manual_lon'] is not None:
            if st.button('Próximo', key='next_to_text'):
                st.session_state['step'] = 2
        st.markdown('</div>', unsafe_allow_html=True)

    # Step 2: Write Complaint
    elif st.session_state['step'] == 2:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.header("Passo 2: Escreva a Queixa")
        def force_rerun():
            pass
        st.text_area(
            "Cole aqui a queixa de um cidadão:",
            height=200,
            value=st.session_state.get('complaint_input', ""),
            key="complaint_input",
            on_change=force_rerun
        )
        text_input = st.session_state['complaint_input']
        # Smart Suggestions for Category and Urgency (update only every 5 or 6 chars)
        if 'last_suggestion_length' not in st.session_state:
            st.session_state['last_suggestion_length'] = 0
        if 'last_suggestion_topic' not in st.session_state:
            st.session_state['last_suggestion_topic'] = ''
        if 'last_suggestion_urgency' not in st.session_state:
            st.session_state['last_suggestion_urgency'] = ''
        current_length = len(text_input.strip())
        length_diff = abs(current_length - st.session_state['last_suggestion_length'])
        should_update = (
            (current_length > 0 and (length_diff in [5, 6])) or
            (st.session_state['last_suggestion_topic'] == '' and current_length > 0)
        ) and not st.session_state.get('analysis_complete', False)
        if should_update:
            topic_suggestion, topic_score = classificar_mensagem(text_input)
            urgencia_suggestion, probas = classificar_urgencia(text_input)
            urgencia_display = URGENCY_LABEL_MAP.get(urgencia_suggestion, urgencia_suggestion)
            st.session_state['last_suggestion_topic'] = topic_suggestion
            st.session_state['last_suggestion_urgency'] = urgencia_display
            st.session_state['last_suggestion_length'] = current_length
        # Show the last suggestion if available
        if st.session_state['last_suggestion_topic'] and not st.session_state.get('analysis_complete', False) and current_length > 0:
            st.markdown(f"<div style='margin-top:0.5em; margin-bottom:0.5em; padding:0.7em 1em; background:#f1f8ff; border-radius:10px; font-size:1.1em;'><b>Sugestão de categoria:</b> <span style='color:#00aae9'>{st.session_state['last_suggestion_topic']}</span> <br><b>Sugestão de urgência:</b> <span style='color:#FF3B30'>{st.session_state['last_suggestion_urgency']}</span></div>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button('Voltar', key='back_to_location'):
                st.session_state['step'] = 1
        with col2:
            if text_input.strip():
                if st.button('Próximo', key='next_to_review'):
                    st.session_state['step'] = 3
        st.markdown('</div>', unsafe_allow_html=True)

    # Step 3: Review & Submit
    elif st.session_state['step'] == 3:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.header("Passo 3: Rever e Submeter")
        # Show summary
        st.markdown(f"<b>Localização:</b> Latitude = {st.session_state['manual_lat']:.6f}, Longitude = {st.session_state['manual_lon']:.6f}", unsafe_allow_html=True)
        st.markdown(f"<b>Queixa:</b> {st.session_state['complaint_input']}", unsafe_allow_html=True)
        st.markdown(f"<b>Categoria sugerida:</b> {st.session_state['last_suggestion_topic']}", unsafe_allow_html=True)
        st.markdown(f"<b>Urgência sugerida:</b> {st.session_state['last_suggestion_urgency']}", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button('Voltar', key='back_to_text'):
                st.session_state['step'] = 2
        with col2:
            if st.button('Submeter Queixa', key='submit_queixa'):
                st.session_state['submit_queixa'] = True
        st.markdown('</div>', unsafe_allow_html=True)

    # Submission and results
    if st.session_state.get('submit_queixa', False) and not st.session_state.get('analysis_complete', False):
        # Only run submission logic once
        st.session_state['submit_queixa'] = False
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
                # Only use manual location
                lat, lon = st.session_state['manual_lat'], st.session_state['manual_lon']
                adresse_extracted = "Selecionado manualmente"
                found = True
                best_score = 100

                # Tópico (Classificação com modelo próprio)
                topic, topic_score = classificar_mensagem(text_input)
                topic_display = topic

                # Urgência
                urgencia, probas = classificar_urgencia(text_input)
                urgencia_display = URGENCY_LABEL_MAP.get(urgencia, urgencia)

                # Resultados
                st.markdown('<a name="resultados"></a>', unsafe_allow_html=True)
                st.subheader("📌 Resultados da Análise")
                st.markdown(f"**Localização extraída:** {adresse_extracted}")
                if isinstance(lat, (float, int)) and isinstance(lon, (float, int)):
                    st.markdown(f"**Coordenadas:** Latitude = {lat:.6f}, Longitude = {lon:.6f}")
                else:
                    st.markdown(f"**Coordenadas:** Latitude = {lat}, Longitude = {lon}")
                st.markdown(f"**Nível de urgência:** `{urgencia_display} ({probas.get(urgencia_display, 0):.2f})`")
                st.markdown(f"**Tópico Detetado:** `{topic_display} ({topic_score:.2f})`")

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
                    'location': adresse_extracted,  # Use the extracted address directly or manual
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
                else:
                    error_message = "❌ Erro ao registar a queixa na base de dados."
                    error_flag = True
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
        elif st.session_state.get('analysis_complete'):
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
        # Scroll to the results section after processing
        st.markdown(
            '''
            <script>
            var anchor = document.getElementsByName('resultados')[0];
            if(anchor){
                anchor.scrollIntoView({behavior: 'smooth'});
            }
            </script>
            ''',
            unsafe_allow_html=True
        )

    # Only show warning if user tries to submit without a location
    if not st.session_state.get('analysis_complete', False) and st.session_state['step'] == 3 and (st.session_state['manual_lat'] is None or st.session_state['manual_lon'] is None):
        st.warning("É obrigatório selecionar a localização no mapa antes de submeter a queixa.")
