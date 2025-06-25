# iVedras ML UDF Setup Guide

This guide explains how to set up Machine Learning UDFs in Snowflake and deploy the updated Streamlit app.

## Overview

The ML models (topic and urgency classification) are now deployed as Snowpark Python UDFs in Snowflake, while the Streamlit app calls these UDFs via SQL queries. This approach allows us to use PyTorch and Transformers in Snowflake while keeping the Streamlit app lightweight.

## Prerequisites

1. **Snowflake Account** with appropriate privileges
2. **Python Environment** with Snowpark installed
3. **Your ML Models** (HuggingFace models: `valterjpcaldeira/iVedrasQueixas` and `valterjpcaldeira/iVedrasUrgencia`)

## Step-by-Step Setup

### 1. Install Snowpark Python

```bash
pip install snowflake-snowpark-python>=1.15.0
```

### 2. Update Connection Configuration

Edit `register_ml_udfs.py` and update the `SNOWFLAKE_CONFIG` with your credentials:

```python
SNOWFLAKE_CONFIG = {
    "account": "RIMGCPU-GQ46314",  # Your account
    "user": "valterjpcaldeira",
    "password": "your_actual_password",  # Replace with your password
    "warehouse": "SNOWFLAKE_LEARNING_WH",
    "database": "IVEDRAS",
    "schema": "PUBLIC",
    "role": "ACCOUNTADMIN"
}
```

### 3. Register the ML UDFs

Run the UDF registration script:

```bash
python register_ml_udfs.py
```

This will:
- Create a stage for ML models
- Register 3 UDFs:
  - `predict_topic(text)` - Classifies complaint topics
  - `predict_urgency(text)` - Classifies urgency levels
  - `predict_urgency_probabilities(text)` - Returns probability distributions

### 4. Test the UDFs

The registration script includes a test. You should see output like:

```
✅ Test results:
   Topic: Limpeza e Resíduos
   Urgency: Média
   Probabilities: {"LABEL_0": 0.1, "LABEL_1": 0.2, "LABEL_2": 0.7}
```

### 5. Deploy the Updated Streamlit App

Update your `snowflake.yml` to use the new app:

```yaml
definition_version: 2
entities:
  ivedras_complaint_system:
    type: streamlit
    identifier: ivedras_complaint_system
    stage: streamlit_stage
    query_warehouse: SNOWFLAKE_LEARNING_WH
    main_file: app_snowflake_udf.py
    artifacts:
      - environment.yml
```

Deploy the app:

```bash
snow streamlit deploy --replace
```

## File Structure

```
iVedras/
├── register_ml_udfs.py          # UDF registration script
├── upload_models_to_stage.py    # Model upload script (if needed)
├── app_snowflake_udf.py         # Updated Streamlit app (uses UDFs)
├── environment.yml              # Updated dependencies (no torch/transformers)
├── requirements_udf.txt         # Requirements for UDF registration
└── README_ML_UDF_SETUP.md       # This file
```

## How It Works

### 1. **UDF Registration**
- ML models are registered as Snowpark Python UDFs
- UDFs can use PyTorch, Transformers, and other ML libraries
- Models are loaded from HuggingFace Hub

### 2. **Streamlit App**
- Calls UDFs via SQL queries
- No ML libraries needed in Streamlit environment
- Lightweight and fast

### 3. **Data Flow**
```
User Input → Streamlit → SQL Query → UDF → ML Model → Result → Streamlit → Display
```

## Troubleshooting

### Common Issues

1. **UDF Registration Fails**
   - Check your Snowflake credentials
   - Ensure you have CREATE UDF privileges
   - Verify the warehouse is running

2. **Model Loading Errors**
   - Check internet connectivity (for HuggingFace models)
   - Verify model names are correct
   - Check package versions compatibility

3. **Streamlit App Errors**
   - Ensure UDFs are registered before deploying Streamlit
   - Check SQL syntax in UDF calls
   - Verify table structure matches expectations

### Error Messages

- `"UDF not found"` - Run the UDF registration script
- `"Package not available"` - Check package compatibility
- `"Model loading failed"` - Verify HuggingFace model names

## Performance Considerations

1. **UDF Cold Start**: First call to UDF may be slower (model loading)
2. **Caching**: Subsequent calls are faster
3. **Concurrency**: Multiple users can use UDFs simultaneously
4. **Resource Usage**: UDFs use Snowflake compute resources

## Security

- UDFs run in Snowflake's secure environment
- No external network access (unless configured)
- Data stays within Snowflake
- Access controlled by Snowflake privileges

## Next Steps

1. **Test the complete workflow**
2. **Monitor UDF performance**
3. **Add more sophisticated address extraction**
4. **Enhance error handling**
5. **Add model versioning**

## Support

If you encounter issues:
1. Check the error messages in Snowflake logs
2. Verify UDF registration was successful
3. Test UDFs individually using SQL
4. Check Streamlit app logs

## Example Usage

### Testing UDFs in Snowflake

```sql
-- Test topic classification
SELECT predict_topic('Há lixo acumulado na rua da Liberdade') AS topic;

-- Test urgency classification  
SELECT predict_urgency('Há lixo acumulado na rua da Liberdade') AS urgency;

-- Test probabilities
SELECT predict_urgency_probabilities('Há lixo acumulado na rua da Liberdade') AS probabilities;
```

### Using in Streamlit

The updated Streamlit app automatically calls these UDFs when analyzing complaints. 