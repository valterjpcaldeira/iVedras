import streamlit as st
import pandas as pd
import plotly.express as px
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
from datetime import datetime
import torch
import pydeck as pdk
from difflib import SequenceMatcher
from rapidfuzz import process, fuzz as rapidfuzz_fuzz
import json

# Fix for PyTorch and Streamlit conflict
import os
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# Initialize models
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

# Map model labels to human-readable urgency names
URGENCY_LABEL_MAP = {
    "LABEL_0": "Baixa",
    "LABEL_1": "Alta",
    "LABEL_2": "Média",
    "LABEL_3": "Baixa"
}

# Map model labels to human-readable topic names
TOPIC_LABEL_MAP = {
    "LABEL_9": "Limpeza e Resíduos",
    "LABEL_8": "Infraestruturas e Obras",
    "LABEL_7": "Trânsito e Mobilidade",
    "LABEL_6": "Áreas Verdes e Espaços Públicos",
    "LABEL_5": "Água e Saneamento",
    "LABEL_4": "Animais e Ambiente",
    "LABEL_3": "Serviços Sociais e Comunitários",
    "LABEL_2": "Segurança e Ordem Pública",
    "LABEL_1": "Comércio e Atividades Económicas",
    "LABEL_0": "Outros"
}

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

def get_snowflake_session():
    """Get Snowflake session using Streamlit connection"""
    return st.connection('snowflake').session()

def get_addresses_from_snowflake():
    """Get addresses from Snowflake table"""
    try:
        session = get_snowflake_session()
        result = session.sql("SELECT address, latitude, longitude, normalized_address FROM addresses").collect()
        if result:
            df = pd.DataFrame(result, columns=['address', 'latitude', 'longitude', 'normalized_address'])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error getting addresses from Snowflake: {e}")
        return pd.DataFrame()

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
            scorer=rapidfuzz_fuzz.token_sort_ratio
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

def get_coordinates_from_snowflake(location):
    addresses_df = get_addresses_from_snowflake()
    if not addresses_df.empty:
        best_match, lat, lon = match_address(location, addresses_df, 55)
        if best_match:
            return best_match, lat, lon
    return None, None, None

def save_complaint_to_snowflake(complaint_data):
    """Save complaint to Snowflake table"""
    try:
        session = get_snowflake_session()
        
        # Convert urgency probabilities to JSON string
        urgency_probs_json = json.dumps(complaint_data['urgency_probabilities'])
        
        sql = f"""
        INSERT INTO complaints (
            problem, location, latitude, longitude, topic, 
            topic_confidence, urgency, urgency_probabilities, timestamp
        ) VALUES (
            '{complaint_data['problem'].replace("'", "''")}',
            '{complaint_data['location'].replace("'", "''")}',
            {complaint_data['latitude'] if complaint_data['latitude'] else 'NULL'},
            {complaint_data['longitude'] if complaint_data['longitude'] else 'NULL'},
            '{complaint_data['topic'].replace("'", "''")}',
            {complaint_data['topic_confidence']},
            '{complaint_data['urgency'].replace("'", "''")}',
            '{urgency_probs_json}',
            CURRENT_TIMESTAMP()
        )
        """
        session.sql(sql).collect()
        return True
    except Exception as e:
        st.error(f"Error saving complaint to Snowflake: {e}")
        return False

def get_complaints_from_snowflake():
    """Get complaints from Snowflake table"""
    try:
        session = get_snowflake_session()
        result = session.sql("""
            SELECT 
                complaint_id,
                problem,
                location,
                latitude,
                longitude,
                topic,
                topic_confidence,
                urgency,
                urgency_probabilities,
                timestamp,
                created_at
            FROM complaints 
            ORDER BY created_at DESC
        """).collect()
        
        if result:
            df = pd.DataFrame(result, columns=[
                'complaint_id', 'problem', 'location', 'latitude', 'longitude',
                'topic', 'topic_confidence', 'urgency', 'urgency_probabilities',
                'timestamp', 'created_at'
            ])
            # Parse urgency probabilities from JSON
            df['urgency_probabilities'] = df['urgency_probabilities'].apply(
                lambda x: json.loads(x) if x else {}
            )
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error getting complaints from Snowflake: {e}")
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

# Carregar o dataset do Snowflake
df = get_complaints_from_snowflake()

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
    if not filtered_df.empty:
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
        # Remove rows with missing coordinates
        map_df = filtered_df.dropna(subset=['latitude', 'longitude'])
        if not map_df.empty:
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
                tooltip={"text": "{location}\n{topic}\n{urgency}"}
            )
            st.pydeck_chart(r)
        else:
            st.info("Não há dados de localização disponíveis para exibir no mapa.")
    else:
        st.info("Não há queixas registadas para exibir.")

# Aba do Analisador
with tab2:
    st.header("Analisador de Queixas")
    st.markdown("Cole aqui a sua queixa e o sistema irá analisá-la automaticamente.")

    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state['analysis_complete'] = False
    if 'error_message' not in st.session_state:
        st.session_state['error_message'] = ""
    if 'warning_message' not in st.session_state:
        st.session_state['warning_message'] = ""

    # Text input for complaint
    text_input = st.text_area(
        "Queixa:",
        placeholder="Descreva aqui a sua queixa...",
        height=150
    )

    # Analyze button
    if st.button("🔍 Analisar Queixa", type="primary"):
        error_flag = False
        warning_flag = False
        analysis_success = False
        error_message = ""
        warning_message = ""

        if text_input.strip():
            with st.spinner("A analisar a queixa..."):
                # Classify topic
                topic_label, topic_score = classificar_mensagem(text_input)
                topic_display = TOPIC_LABEL_MAP.get(topic_label, topic_label)

                # Classify urgency
                urgency_label, probas = classificar_urgencia(text_input)
                urgencia_display = URGENCY_LABEL_MAP.get(urgency_label, urgency_label)

                # Extract address using NER
                def extract_addresses(text):
                    # Extract entities from the NER pipeline
                    entities = ner_pipeline(text)
                    addresses = []
                    for entity in entities:
                        if entity['entity_group'] in ['LOC', 'GPE', 'FAC']:
                            addresses.append(entity['word'])
                    return addresses

                extracted_addresses = extract_addresses(text_input)
                adresse_extracted = ", ".join(extracted_addresses) if extracted_addresses else ""

                # Get coordinates from Snowflake addresses table
                lat, lon = None, None
                if adresse_extracted:
                    best_match, lat, lon = get_coordinates_from_snowflake(adresse_extracted)
                    if best_match:
                        adresse_extracted = best_match
                    else:
                        warning_message = "Não foi possível obter a morada, tente dar mais detalhe ou coloque outra."
                        warning_flag = True

                # Save to Snowflake
                complaint_data = {
                    'problem': text_input,
                    'location': adresse_extracted,
                    'latitude': lat,
                    'longitude': lon,
                    'topic': topic_display,
                    'topic_confidence': float(topic_score),
                    'urgency': urgencia_display,
                    'urgency_probabilities': probas,
                    'timestamp': datetime.utcnow()
                }
                if save_complaint_to_snowflake(complaint_data):
                    st.session_state['error_message'] = ""
                    st.session_state['warning_message'] = ""
                    st.session_state['analysis_complete'] = True
                    analysis_success = True
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

    # Display recent complaints
    if not df.empty:
        st.subheader("Queixas Recentes")
        recent_df = df.head(10)[['problem', 'topic', 'urgency', 'created_at']]
        recent_df['created_at'] = pd.to_datetime(recent_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(recent_df, use_container_width=True) 