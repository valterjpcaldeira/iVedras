"""
Snowpark Python UDF Registration Script for iVedras ML Models
This script registers the topic and urgency classification UDFs in Snowflake.
"""

import os
import sys
from snowflake.snowpark import Session
from snowflake.snowpark.types import StringType, FloatType, VariantType
from snowflake.snowpark.functions import udf
import json

# Snowflake connection parameters
SNOWFLAKE_CONFIG = {
    "account": "RIMGCPU-GQ46314",  # Your account
    "user": "valterjpcaldeira",
    "password": "azrSU8KB4EidTdU",  # Replace with your password
    "warehouse": "SNOWFLAKE_LEARNING_WH",
    "database": "IVEDRAS",
    "schema": "PUBLIC",
    "role": "ACCOUNTADMIN"
}

def create_session():
    """Create Snowflake session"""
    config = SNOWFLAKE_CONFIG.copy()
    config['custom_package_usage_config'] = {'enabled': True}
    return Session.builder.configs(config).create()

def register_topic_classifier_udf(session):
    """Register the topic classification UDF"""
    
    @udf(name="predict_topic", 
         return_type=StringType(),
         input_types=[StringType()],
         is_permanent=True,
         stage_location="@IVEDRAS.PUBLIC.ML_MODELS_STAGE",
         replace=True)
    def predict_topic(text: str) -> str:
        """Classify the topic of a complaint text"""
        try:
            # For now, return a simple classification based on keywords
            # This is a placeholder until we can properly load the models
            text_lower = text.lower()
            
            if any(word in text_lower for word in ['lixo', 'lixeira', 'resíduos', 'limpeza']):
                return "Limpeza e Resíduos"
            elif any(word in text_lower for word in ['estrada', 'rua', 'pavimento', 'buraco']):
                return "Infraestruturas e Obras"
            elif any(word in text_lower for word in ['trânsito', 'carro', 'estacionamento', 'semáforo']):
                return "Trânsito e Mobilidade"
            elif any(word in text_lower for word in ['parque', 'jardim', 'árvore', 'verde']):
                return "Áreas Verdes e Espaços Públicos"
            elif any(word in text_lower for word in ['água', 'esgoto', 'saneamento']):
                return "Água e Saneamento"
            elif any(word in text_lower for word in ['cão', 'gato', 'animal']):
                return "Animais e Ambiente"
            else:
                return "Outros"
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    return predict_topic

def register_urgency_classifier_udf(session):
    """Register the urgency classification UDF"""
    
    @udf(name="predict_urgency", 
         return_type=StringType(),
         input_types=[StringType()],
         is_permanent=True,
         stage_location="@IVEDRAS.PUBLIC.ML_MODELS_STAGE",
         replace=True)
    def predict_urgency(text: str) -> str:
        """Classify the urgency level of a complaint text"""
        try:
            # Simple keyword-based urgency classification
            text_lower = text.lower()
            
            # High urgency keywords
            high_urgency_words = ['emergência', 'urgente', 'perigo', 'acidente', 'incêndio', 'inundação']
            # Medium urgency keywords  
            medium_urgency_words = ['problema', 'avaria', 'defeito', 'quebrado', 'danificado']
            
            if any(word in text_lower for word in high_urgency_words):
                return "Alta"
            elif any(word in text_lower for word in medium_urgency_words):
                return "Média"
            else:
                return "Baixa"
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    return predict_urgency

def register_urgency_probabilities_udf(session):
    """Register the urgency probabilities UDF (returns JSON)"""
    
    @udf(name="predict_urgency_probabilities", 
         return_type=VariantType(),
         input_types=[StringType()],
         is_permanent=True,
         stage_location="@IVEDRAS.PUBLIC.ML_MODELS_STAGE",
         replace=True)
    def predict_urgency_probabilities(text: str) -> str:
        """Get urgency classification probabilities as JSON"""
        try:
            # Simple keyword-based urgency classification with probabilities
            text_lower = text.lower()
            
            # High urgency keywords
            high_urgency_words = ['emergência', 'urgente', 'perigo', 'acidente', 'incêndio', 'inundação']
            # Medium urgency keywords  
            medium_urgency_words = ['problema', 'avaria', 'defeito', 'quebrado', 'danificado']
            
            if any(word in text_lower for word in high_urgency_words):
                probas = {"Baixa": 0.1, "Média": 0.2, "Alta": 0.7}
            elif any(word in text_lower for word in medium_urgency_words):
                probas = {"Baixa": 0.2, "Média": 0.6, "Alta": 0.2}
            else:
                probas = {"Baixa": 0.7, "Média": 0.2, "Alta": 0.1}
            
            return json.dumps(probas)
            
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    return predict_urgency_probabilities

def create_ml_stage(session):
    """Create the ML models stage if it doesn't exist"""
    try:
        session.sql("CREATE STAGE IF NOT EXISTS ML_MODELS_STAGE").collect()
        print("✅ ML models stage created successfully")
    except Exception as e:
        print(f"⚠️ Stage creation warning: {e}")

def main():
    """Main function to register all UDFs"""
    print("🚀 Starting UDF registration for iVedras ML models...")
    
    try:
        # Create session
        session = create_session()
        print("✅ Snowflake session created")
        
        # Create stage
        create_ml_stage(session)
        
        # Register UDFs
        print("📝 Registering topic classifier UDF...")
        topic_udf = register_topic_classifier_udf(session)
        
        print("📝 Registering urgency classifier UDF...")
        urgency_udf = register_urgency_classifier_udf(session)
        
        print("📝 Registering urgency probabilities UDF...")
        urgency_probs_udf = register_urgency_probabilities_udf(session)
        
        # Test the UDFs
        print("🧪 Testing UDFs...")
        test_text = "Há lixo acumulado na rua da Liberdade"
        
        topic_result = session.sql(f"SELECT predict_topic('{test_text}') AS topic").collect()
        urgency_result = session.sql(f"SELECT predict_urgency('{test_text}') AS urgency").collect()
        probs_result = session.sql(f"SELECT predict_urgency_probabilities('{test_text}') AS probabilities").collect()
        
        print(f"✅ Test results:")
        print(f"   Topic: {topic_result[0]['TOPIC']}")
        print(f"   Urgency: {urgency_result[0]['URGENCY']}")
        print(f"   Probabilities: {probs_result[0]['PROBABILITIES']}")
        
        print("🎉 All UDFs registered successfully!")
        
        # Close session
        session.close()
        
    except Exception as e:
        print(f"❌ Error during UDF registration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 