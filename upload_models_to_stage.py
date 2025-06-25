"""
Script to upload model files to Snowflake stage
Use this if you have local model files that need to be uploaded.
"""

from snowflake.snowpark import Session
import os

# Snowflake connection parameters
SNOWFLAKE_CONFIG = {
    "account": "RIMGCPU-GQ46314",
    "user": "valterjpcaldeira", 
    "password": "your_password_here",  # Replace with your password
    "warehouse": "SNOWFLAKE_LEARNING_WH",
    "database": "IVEDRAS",
    "schema": "PUBLIC",
    "role": "ACCOUNTADMIN"
}

def upload_models_to_stage():
    """Upload model files to Snowflake stage"""
    
    session = Session.builder.configs(SNOWFLAKE_CONFIG).create()
    
    try:
        # Create stage if it doesn't exist
        session.sql("CREATE STAGE IF NOT EXISTS ML_MODELS_STAGE").collect()
        print("✅ ML models stage created")
        
        # List of model files to upload (if you have local files)
        model_files = [
            # Add your local model files here if needed
            # "path/to/your/model.pkl",
            # "path/to/your/tokenizer.json",
        ]
        
        if model_files:
            for file_path in model_files:
                if os.path.exists(file_path):
                    # Upload file to stage
                    session.sql(f"PUT file://{file_path} @ML_MODELS_STAGE OVERWRITE=TRUE").collect()
                    print(f"✅ Uploaded {file_path} to stage")
                else:
                    print(f"⚠️ File not found: {file_path}")
        else:
            print("ℹ️ No local model files to upload (using HuggingFace models)")
        
        # List files in stage
        files = session.sql("LIST @ML_MODELS_STAGE").collect()
        print(f"📁 Files in stage: {len(files)}")
        for file in files:
            print(f"   - {file['name']}")
            
    except Exception as e:
        print(f"❌ Error uploading models: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    upload_models_to_stage() 