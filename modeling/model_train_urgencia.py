from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import random

# Exemplos sintéticos de queixas com urgência
queixas_urgencia = [
    {"text": "Há um cano rebentado a inundar a rua.", "urgencia": "urgente"},
    {"text": "Fogo ativo perto de habitações.", "urgencia": "urgente"},
    {"text": "Pessoa presa no elevador do prédio.", "urgencia": "urgente"},
    {"text": "Buraco profundo na estrada principal, risco de acidente.", "urgencia": "urgente"},
    {"text": "Fuga de gás detetada no bairro.", "urgencia": "urgente"},
    {"text": "Cabo elétrico caído na via pública.", "urgencia": "urgente"},
    {"text": "Idoso caído na rua sem conseguir levantar-se.", "urgencia": "urgente"},
    {"text": "Esgoto a céu aberto junto a escola.", "urgencia": "urgente"},
    {"text": "Árvore prestes a cair sobre carros estacionados.", "urgencia": "urgente"},
    {"text": "Acidente de viação com feridos.", "urgencia": "urgente"},
    # Não urgente
    {"text": "Contentor de lixo cheio.", "urgencia": "nao_urgente"},
    {"text": "Falta de iluminação numa rua secundária.", "urgencia": "nao_urgente"},
    {"text": "Ervas altas no jardim municipal.", "urgencia": "nao_urgente"},
    {"text": "Banco partido no parque infantil.", "urgencia": "nao_urgente"},
    {"text": "Pintura de passadeira desbotada.", "urgencia": "nao_urgente"},
    {"text": "Reclamação sobre barulho de vizinhos.", "urgencia": "nao_urgente"},
    {"text": "Solicitação de mais árvores na praça.", "urgencia": "nao_urgente"},
    {"text": "Pedido de limpeza de sarjetas.", "urgencia": "nao_urgente"},
    {"text": "Sugestão de mais horários para autocarros.", "urgencia": "nao_urgente"},
    {"text": "Solicitação de rampa para mobilidade reduzida.", "urgencia": "nao_urgente"},
]

# Construir o dataset
random.shuffle(queixas_urgencia)
df = pd.DataFrame(queixas_urgencia)

le = LabelEncoder()
df["urgencia_id"] = le.fit_transform(df["urgencia"])

# Criar dataset Hugging Face
dataset = Dataset.from_pandas(df[["text", "urgencia_id"]].rename(columns={"urgencia_id": "labels"}))
dataset = dataset.train_test_split(test_size=0.2)

# Tokenizer
model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)

# Modelo
num_labels = len(le.classes_)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Treinamento
training_args = TrainingArguments(
    output_dir="output_urgencia",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",  # Save at the end of each epoch
    weight_decay=0.01,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

trainer.train()

# Guardar modelo treinado
model.save_pretrained("iVedras_urgencia")
tokenizer.save_pretrained("iVedras_urgencia")

# Guardar o LabelEncoder
import joblib
joblib.dump(le, "iVedras_urgencia_encoder.pkl") 