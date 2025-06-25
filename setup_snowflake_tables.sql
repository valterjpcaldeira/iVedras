-- Create database and schema if they don't exist
CREATE DATABASE IF NOT EXISTS IVEDRAS;
USE DATABASE IVEDRAS;
CREATE SCHEMA IF NOT EXISTS PUBLIC;
USE SCHEMA PUBLIC;

-- Create complaints table
CREATE OR REPLACE TABLE complaints (
    complaint_id NUMBER AUTOINCREMENT,
    problem TEXT NOT NULL,
    location TEXT,
    latitude FLOAT,
    longitude FLOAT,
    topic VARCHAR(100),
    topic_confidence FLOAT,
    urgency VARCHAR(50),
    urgency_probabilities VARIANT, -- JSON object for probabilities
    timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Create addresses table for geocoding data
CREATE OR REPLACE TABLE addresses (
    address_id NUMBER AUTOINCREMENT,
    address TEXT NOT NULL,
    latitude FLOAT,
    longitude FLOAT,
    normalized_address TEXT,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Insert sample addresses from your CSV file
-- You can load this data using the PUT command or Snowsight file upload
INSERT INTO addresses (address, latitude, longitude, normalized_address) VALUES
('Rua da Liberdade, Torres Vedras', 39.0917, -9.2583, 'rua da liberdade torres vedras'),
('Avenida 5 de Outubro, Torres Vedras', 39.0925, -9.2591, 'avenida 5 de outubro torres vedras'),
('Praça 25 de Abril, Torres Vedras', 39.0933, -9.2579, 'praca 25 de abril torres vedras');

-- Create warehouse if it doesn't exist
CREATE WAREHOUSE IF NOT EXISTS SNOWFLAKE_LEARNING_WH
    WAREHOUSE_SIZE = 'X-SMALL'
    AUTO_SUSPEND = 300
    AUTO_RESUME = TRUE;

-- Grant necessary permissions
GRANT USAGE ON DATABASE IVEDRAS TO ROLE ACCOUNTADMIN;
GRANT USAGE ON SCHEMA IVEDRAS.PUBLIC TO ROLE ACCOUNTADMIN;
GRANT ALL ON TABLE IVEDRAS.PUBLIC.complaints TO ROLE ACCOUNTADMIN;
GRANT ALL ON TABLE IVEDRAS.PUBLIC.addresses TO ROLE ACCOUNTADMIN;
GRANT USAGE ON WAREHOUSE SNOWFLAKE_LEARNING_WH TO ROLE ACCOUNTADMIN; 