import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
from datetime import datetime
import re
from rapidfuzz import process, fuzz as rapidfuzz_fuzz
import json

# Fix for Streamlit
import os
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

def get_snowflake_session():
    """Get Snowflake session using Streamlit connection"""
    return st.connection('snowflake').session()

def classificar_mensagem(texto):
    """Classify topic using Snowflake UDF"""
    try:
        session = get_snowflake_session()
        # Escape single quotes in the text
        escaped_text = texto.replace("'", "''")
        result = session.sql(f"SELECT predict_topic('{escaped_text}') AS topic").collect()
        return result[0]['TOPIC']
    except Exception as e:
        st.error(f"Error classifying topic: {e}")
        return "Outros"

def classificar_urgencia(texto):
    """Classify urgency using Snowflake UDF"""
    try:
        session = get_snowflake_session()
        # Escape single quotes in the text
        escaped_text = texto.replace("'", "''")
        result = session.sql(f"SELECT predict_urgency('{escaped_text}') AS urgency").collect()
        return result[0]['URGENCY']
    except Exception as e:
        st.error(f"Error classifying urgency: {e}")
        return "Média"

def get_urgency_probabilities(texto):
    """Get urgency probabilities using Snowflake UDF"""
    try:
        session = get_snowflake_session()
        # Escape single quotes in the text
        escaped_text = texto.replace("'", "''")
        result = session.sql(f"SELECT predict_urgency_probabilities('{escaped_text}') AS probabilities").collect()
        probas_str = result[0]['PROBABILITIES']
        if isinstance(probas_str, str):
            return json.loads(probas_str)
        else:
            return probas_str
    except Exception as e:
        st.error(f"Error getting urgency probabilities: {e}")
        return {"Baixa": 0.33, "Média": 0.34, "Alta": 0.33}

def normalize_address(addr):
    addr = addr.lower()
    addr = re.sub(r'[^\w\s]', '', addr)  # remove punctuation
    addr = re.sub(r'\s+', ' ', addr).strip()  # remove excess whitespace
    return addr

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

def extract_addresses_from_text(text):
    """Extract addresses from text using regex patterns and keywords"""
    extracted_addresses = []
    
    # Extract postal codes in the format 2560-374
    postal_codes = re.findall(r'\b\d{4}-\d{3}\b', text)
    extracted_addresses.extend(postal_codes)
    
    # Extract location keywords and following words
    location_keywords = [
        'rua', 'avenida', 'estrada', 'largo', 'praça', 'bairro', 'zona', 'área',
        'travessa', 'calçada', 'alameda', 'passeio', 'caminho', 'monte', 'vila',
        'freguesia', 'concelho', 'distrito'
    ]
    
    words = text.split()
    for i, word in enumerate(words):
        word_lower = word.lower()
        for keyword in location_keywords:
            if keyword in word_lower:
                # Extract the location name (current word + next few words)
                location_parts = [word]
                # Add next 2-3 words if they exist
                for j in range(1, 4):
                    if i + j < len(words):
                        next_word = words[i + j]
                        # Stop if we hit punctuation or common words
                        if next_word.lower() in ['em', 'na', 'no', 'da', 'de', 'do', 'para', 'com', 'que', 'está', 'muito', 'muito']:
                            break
                        location_parts.append(next_word)
                    else:
                        break
                extracted_address = ' '.join(location_parts)
                if extracted_address not in extracted_addresses:
                    extracted_addresses.append(extracted_address)
                break
    
    return extracted_addresses

def get_coordinates_from_snowflake(location):
    """Get coordinates from Snowflake addresses table with improved matching"""
    addresses_df = get_addresses_from_snowflake()
    if not addresses_df.empty:
        # Try exact match first
        exact_match = addresses_df[addresses_df['address'].str.contains(location, case=False, na=False)]
        if not exact_match.empty:
            row = exact_match.iloc[0]
            return row['address'], row['latitude'], row['longitude']
        
        # Try fuzzy matching
        best_match, lat, lon = match_address(location, addresses_df, 55)
        if best_match:
            return best_match, lat, lon
        
        # Try with normalized address
        normalized_location = normalize_address(location)
        for idx, row in addresses_df.iterrows():
            if normalized_location in normalize_address(row['address']):
                return row['address'], row['latitude'], row['longitude']
    
    return None, None, None

def save_complaint_to_snowflake(complaint_data):
    """Save complaint to Snowflake table"""
    try:
        session = get_snowflake_session()
        
        # Get urgency probabilities and ensure it's a dictionary
        urgency_probs = complaint_data['urgency_probabilities']
        if isinstance(urgency_probs, str):
            urgency_probs = json.loads(urgency_probs)
        
        # Convert to JSON string and escape properly for SQL
        urgency_probs_json = json.dumps(urgency_probs, ensure_ascii=False)
        # Escape single quotes for SQL
        urgency_probs_json = urgency_probs_json.replace("'", "''")
        
        sql = f"""
        INSERT INTO complaints (
            problem, location, latitude, longitude, topic, 
            topic_confidence, urgency, urgency_probabilities, timestamp
        )
        SELECT 
            '{complaint_data['problem'].replace("'", "''")}',
            '{complaint_data['location'].replace("'", "''")}',
            {complaint_data['latitude'] if complaint_data['latitude'] else 'NULL'},
            {complaint_data['longitude'] if complaint_data['longitude'] else 'NULL'},
            '{complaint_data['topic'].replace("'", "''")}',
            {complaint_data['topic_confidence']},
            '{complaint_data['urgency'].replace("'", "''")}',
            TRY_PARSE_JSON('{urgency_probs_json}'),
            CURRENT_TIMESTAMP()
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
                complaint_id, problem, location, latitude, longitude,
                topic, topic_confidence, urgency, urgency_probabilities,
                timestamp, created_at
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

# Page configuration
st.set_page_config(
    page_title="Sistema de Gestão de Queixas",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': 'iVedras - Sistema de Gestão de Queixas de Cidadãos'
    }
)

# Set light theme with blue accents
st.markdown("""
    <style>
        .stApp {
            background-color: #ffffff !important;
            color: #00aae9 !important;
        }
        .main {
            background-color: #ffffff !important;
            padding: 1rem !important;
        }
        .block-container {
            background-color: #ffffff !important;
            padding: 1rem !important;
            max-width: 1200px !important;
        }
        /* Blue minimalist buttons */
        .stButton > button {
            background-color: #00aae9 !important;
            color: #ffffff !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            font-weight: 400 !important;
            font-size: 0.9rem !important;
            border-radius: 6px !important;
            transition: background-color 0.2s ease !important;
        }
        .stButton > button:hover {
            background-color: #ec008c !important;
        }
        /* Blue sidebar */
        div[data-testid="stSidebar"] {
            background-color: #fdee00 !important;
            border-right: 1px solid #00aae9 !important;
        }
        div[data-testid="stSidebar"] .block-container {
            background-color: #fdee00 !important;
        }
        /* Blue headers */
        h1, h2, h3, h4, h5, h6 {
            color: #00aae9 !important;
            font-weight: 400 !important;
        }
        h1 {
            font-weight: 700 !important;
            font-size: 2.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        /* Blue input borders */
        .stTextInput>div>div>input, .stTextArea textarea, .stSelectbox > div > div {
            border: 2px solid #00aae9 !important;
            border-radius: 5px !important;
            color: #00aae9 !important;
        }
        .stTextArea textarea:focus {
            border-color: #00aae9 !important;
            box-shadow: none !important;
        }
        /* Blue metric cards */
        div[data-testid="metric-container"] {
            background-color: #ffffff !important;
            border: 1.5px solid #00aae9 !important;
            border-radius: 6px !important;
            padding: 1rem !important;
        }
        /* Blue dataframes */
        .dataframe {
            background-color: #ffffff !important;
            border: 1.5px solid #00aae9 !important;
            border-radius: 6px !important;
        }
        /* Blue map borders */
        .stMap {
            border: 2px solid #00aae9 !important;
            border-radius: 6px !important;
        }
        /* Blue alerts */
        .stSuccess {
            background-color: #fdee00 !important;
            color: #00aae9 !important;
        }
        .stError {
            background-color: #ed1c24 !important;
            color: #ffffff !important;
        }
        .stWarning {
            background-color: #ec008c !important;
            color: #ffffff !important;
        }
        /* Blue scrollbars */
        ::-webkit-scrollbar {
            width: 8px;
            background: #fdee00;
        }
        ::-webkit-scrollbar-thumb {
            background: #00aae9;
            border-radius: 4px;
        }
        /* Remove excessive shadows and gradients */
        * {
            box-shadow: none !important;
        }
        /* Clean charts */
        .js-plotly-plot {
            background-color: #ffffff !important;
            border-radius: 6px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown("""
    <div style="text-align: center; padding: 2rem 0; margin-bottom: 2rem;">
        <h1 style="margin: 0; font-weight: 300; color: #2c3e50;">iVedras</h1>
        <p style="margin: 0.5rem 0 0 0; color: #6c757d; font-weight: 300; font-size: 1.1rem;">A tua voz na Iniciativa Vedras para uma cidade melhor</p>
    </div>
""", unsafe_allow_html=True)

# Tab navigation (like original)
tab1, tab2 = st.tabs(["📊 Dashboard", "➕ Adicionar Queixa"])

df = get_complaints_from_snowflake()

with tab1:
    st.header("Visualização de Queixas de Cidadãos")
    # Filters
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

    filtered_df = df.copy()
    if selected_topic != 'Todos':
        filtered_df = filtered_df[filtered_df['topic'] == selected_topic]
    if selected_urgency != 'Todos':
        filtered_df = filtered_df[filtered_df['urgency'] == selected_urgency]

    if not filtered_df.empty:
        st.subheader("Contagem de Queixas por Tópico e Urgência")
        pivot_df = filtered_df.groupby(['topic', 'urgency']).size().unstack(fill_value=0)
        urgency_order = ['Alta', 'Média', 'Baixa']
        urgency_colors = {'Alta': '#FF3B30', 'Média': '#FF9500', 'Baixa': '#34C759'}
        for urg in urgency_order:
            if urg not in pivot_df.columns:
                pivot_df[urg] = 0
        pivot_df = pivot_df[urgency_order]
        fig = px.bar(
            pivot_df, x=pivot_df.index, y=urgency_order,
            title="Queixas por Tópico e Nível de Urgência",
            labels={'x': 'Tópico', 'value': 'Número de Queixas', 'variable': 'Urgência'},
            barmode='stack', color_discrete_map=urgency_colors
        )
        fig.update_layout(xaxis_tickangle=-45, legend_title_text='Urgência')
        st.plotly_chart(fig)
        st.subheader("Mapa das Queixas")
        map_df = filtered_df.dropna(subset=['latitude', 'longitude'])
        if not map_df.empty:
            view_state = pdk.ViewState(
                latitude=map_df['latitude'].mean(),
                longitude=map_df['longitude'].mean(),
                zoom=10, pitch=0
            )
            heatmap_layer = pdk.Layer(
                "HeatmapLayer", data=map_df,
                get_position='[longitude, latitude]',
                aggregation=pdk.types.String("MEAN"),
                get_weight=1, radiusPixels=60,
            )
            r = pdk.Deck(layers=[heatmap_layer], initial_view_state=view_state,
                        tooltip={"text": "{location}\n{topic}\n{urgency}"})
            st.pydeck_chart(r)
        else:
            st.info("Não há dados de localização disponíveis para exibir no mapa.")
    else:
        st.info("Não há queixas registadas para exibir.")

with tab2:
    st.header("Analisador de Queixas")
    st.markdown("Cole aqui a sua queixa e o sistema irá analisá-la automaticamente.")
    if 'analysis_complete' not in st.session_state:
        st.session_state['analysis_complete'] = False
    if 'error_message' not in st.session_state:
        st.session_state['error_message'] = ""
    if 'warning_message' not in st.session_state:
        st.session_state['warning_message'] = ""
    text_input = st.text_area(
        "Queixa:",
        placeholder="Descreva aqui a sua queixa...",
        height=150
    )
    if st.button("🔍 Analisar Queixa", type="primary"):
        error_flag = False
        warning_flag = False
        analysis_success = False
        error_message = ""
        warning_message = ""
        
        if text_input.strip():
            with st.spinner("A analisar a queixa com IA..."):
                # Classify topic and urgency using UDFs
                topic_display = classificar_mensagem(text_input)
                urgencia_display = classificar_urgencia(text_input)
                probas = get_urgency_probabilities(text_input)

                # Extract addresses from text using improved logic
                address_data = extract_addresses_from_text(text_input)
                
                # Get coordinates and verify location
                found = False
                extracted_location = "Torres Vedras"  # Default
                lat, lon = 39.0917, -9.2583  # Default coordinates
                
                if address_data:
                    # Try each extracted address in Snowflake first
                    best_score = 0
                    best_match = None
                    best_lat = None
                    best_lon = None
                    
                    for address in address_data:
                        match, extracted_lat, extracted_lon = get_coordinates_from_snowflake(address)
                        if match and extracted_lat is not None and extracted_lon is not None:
                            # Calculate similarity score for this match
                            score = rapidfuzz_fuzz.token_sort_ratio(
                                normalize_address(address), 
                                normalize_address(match)
                            )
                            if score > best_score:
                                best_score = score
                                best_match = match
                                best_lat = extracted_lat
                                best_lon = extracted_lon
                    
                    if best_match:
                        extracted_location = best_match
                        lat, lon = best_lat, best_lon
                        found = True
                    
                    # If not found, try combined address
                    if not found:
                        combined_address = " ".join(address_data)
                        best_match, extracted_lat, extracted_lon = get_coordinates_from_snowflake(combined_address)
                        if best_match and extracted_lat is not None and extracted_lon is not None:
                            extracted_location = best_match
                            lat, lon = extracted_lat, extracted_lon
                            found = True
                
                # If still not found, use the first extracted address as fallback
                if not found and address_data:
                    extracted_location = address_data[0]
                
                # Store results in session state for display
                st.session_state['analysis_results'] = {
                    'topic': topic_display,
                    'urgency': urgencia_display,
                    'probabilities': probas,
                    'location': extracted_location,
                    'latitude': lat,
                    'longitude': lon,
                    'addresses_found': address_data
                }

                # Save to Snowflake
                complaint_data = {
                    'problem': text_input,
                    'location': extracted_location,
                    'latitude': lat,
                    'longitude': lon,
                    'topic': topic_display,
                    'topic_confidence': 0.85,  # Default confidence
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
                    st.session_state['error_message'] = "❌ Erro ao registar a queixa na base de dados."
                    st.session_state['analysis_complete'] = False
                    error_flag = True
        else:
            error_message = "Por favor cole uma queixa para analisar."
            st.session_state['analysis_complete'] = False
            error_flag = True
            
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

    # Display messages
    if st.session_state.get('error_message'):
        st.error(st.session_state['error_message'])
    elif st.session_state.get('warning_message'):
        st.warning(st.session_state['warning_message'])
    elif st.session_state.get('analysis_complete'):
        st.success("✅ Queixa registada com sucesso na base de dados!")

    # Display analysis results
    if st.session_state.get('analysis_results'):
        st.subheader("📊 Resultados da Análise IA")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("🎯 Tópico", st.session_state['analysis_results']['topic'])
        
        with col2:
            st.metric("⚡ Urgência", st.session_state['analysis_results']['urgency'])
        
        # Display location and map
        st.subheader("📍 Localização Extraída")
        location_col1, location_col2 = st.columns([1, 2])
        
        with location_col1:
            st.metric("📍 Localização", st.session_state['analysis_results']['location'])
            st.metric("🌍 Coordenadas", f"{st.session_state['analysis_results']['latitude']:.4f}, {st.session_state['analysis_results']['longitude']:.4f}")
            
            # Show extracted addresses if available
            if st.session_state['analysis_results'].get('addresses_found'):
                st.write("**Endereços extraídos:**")
                for addr in st.session_state['analysis_results']['addresses_found']:
                    st.write(f"• {addr}")
        
        with location_col2:
            # Create map
            lat = st.session_state['analysis_results']['latitude']
            lon = st.session_state['analysis_results']['longitude']
            location_name = st.session_state['analysis_results']['location']
            
            # Create a simple map with the location
            map_data = pd.DataFrame({
                'lat': [lat],
                'lon': [lon],
                'location': [location_name]
            })
            
            st.map(map_data, use_container_width=True)
        
        # Display urgency probabilities
        if st.session_state['analysis_results']['probabilities']:
            st.subheader("📈 Probabilidades de Urgência")
            probs = st.session_state['analysis_results']['probabilities']
            if isinstance(probs, str):
                probs = json.loads(probs)
            
            # Create a bar chart for probabilities
            prob_df = pd.DataFrame(list(probs.items()), columns=['Urgência', 'Probabilidade'])
            fig = px.bar(prob_df, x='Urgência', y='Probabilidade', 
                        title="Probabilidades de Classificação de Urgência",
                        color='Urgência',
                        color_discrete_map={'Baixa': '#34C759', 'Média': '#FF9500', 'Alta': '#FF3B30'})
            fig.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)
    if not df.empty:
        st.subheader("Queixas Recentes")
        recent_df = df.head(10)[['problem', 'topic', 'urgency', 'created_at']]
        recent_df['created_at'] = pd.to_datetime(recent_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(recent_df, use_container_width=True) 