from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch

import pandas as pd
import random

topicos_queixas = {
    "Limpeza e Resíduos": [
        "Há lixo acumulado há dias na Rua das Amoreiras.",
        "Os contentores estão sempre cheios e a transbordar.",
        "Precisamos de mais recolha de resíduos na zona industrial.",
        "A limpeza das ruas não é feita regularmente.",
        "Há sacos de lixo espalhados junto ao mercado.",
        "O caixote de lixo público está partido.",
        "Os moradores não respeitam a separação de resíduos.",
        "Há muito lixo no parque infantil.",
        "Os recicláveis não estão a ser recolhidos.",
        "Mau cheiro vindo dos contentores na Avenida Principal."
    ],
    "Infraestruturas e Obras": [
        "Os passeios estão todos partidos e difíceis de andar.",
        "A rua está cheia de buracos e poças.",
        "Obras paradas há meses sem explicação.",
        "A iluminação pública não funciona nesta zona.",
        "O muro do cemitério está prestes a cair.",
        "O telhado do pavilhão municipal tem infiltrações.",
        "As obras na rotunda causam muitos transtornos.",
        "Falta uma passadeira junto à escola.",
        "O campo de futebol precisa de obras urgentes.",
        "A ponte da estrada municipal está em mau estado."
    ],
    "Trânsito e Mobilidade": [
        "O trânsito está sempre congestionado na rotunda central.",
        "Faltam sinais de trânsito perto da escola.",
        "Estacionamento em segunda fila causa problemas diários.",
        "Buraco na estrada",
        "Muitos carros em excesso de velocidade nesta rua.",
        "Os semáforos estão desregulados no cruzamento principal.",
        "Faltam lugares de estacionamento para moradores.",
        "O autocarro raramente cumpre os horários.",
        "A ciclovia está mal sinalizada.",
        "Há carros a bloquear a paragem de autocarro.",
        "As ruas estreitas dificultam a circulação dos autocarros."
    ],
    "Áreas Verdes e Espaços Públicos": [
        "O jardim junto à escola está mal cuidado e com ervas altas.",
        "Os bancos do parque estão partidos.",
        "Faltam árvores e sombras na praça principal.",
        "O parque infantil precisa de manutenção urgente.",
        "Não há iluminação no parque à noite.",
        "As fontes do jardim municipal estão sem funcionar.",
        "Há seringas no jardim público, é perigoso.",
        "O espaço para piqueniques está sujo e abandonado.",
        "Precisamos de mais espaços verdes na zona urbana.",
        "As vedações do parque estão danificadas."
    ],
    "Água e Saneamento": [
        "Tem havido cortes de água frequentes no bairro do Barro.",
        "A água sai com cor estranha das torneiras.",
        "O esgoto está a céu aberto junto à escola.",
        "Mau cheiro vindo das condutas de esgoto.",
        "Falta pressão da água nas casas da encosta.",
        "Rutura de cano na Rua das Flores.",
        "Há poças de esgoto na estrada.",
        "A estação de tratamento tem mau cheiro constante.",
        "As sarjetas estão entupidas com lixo.",
        "Água a escorrer continuamente de uma tampa na rua."
    ],
    "Animais e Ambiente": [
        "Há um cão abandonado no parque infantil.",
        "Muitos gatos vadios na zona antiga da cidade.",
        "Ruído constante de cães em quintal ao lado.",
        "Alguém está a envenenar animais no bairro.",
        "Falta de controlo de pombos no centro histórico.",
        "Precisamos de um canil municipal com urgência.",
        "A ribeira está poluída com resíduos industriais.",
        "Encontrado animal selvagem ferido na estrada.",
        "Pessoas a alimentar animais na via pública.",
        "Presença de javalis perto das moradias."
    ],
    "Serviços Sociais e Comunitários": [
        "Faltam atividades para idosos no centro comunitário.",
        "As instalações da creche estão degradadas.",
        "A biblioteca municipal tem horário muito reduzido.",
        "Faltam médicos no centro de saúde.",
        "A cantina social está sem condições.",
        "A fila para marcação médica é muito longa.",
        "Os jovens precisam de mais apoio no bairro.",
        "As aulas de ginástica sénior foram canceladas.",
        "Poucos apoios para famílias carenciadas.",
        "Não há acesso adequado para pessoas com mobilidade reduzida."
    ],
    "Segurança e Ordem Pública": [
        "Há assaltos frequentes nesta zona.",
        "A iluminação pública falha e deixa a rua insegura.",
        "Barulho de festas ilegais todas as semanas.",
        "Consumo de droga à vista no parque.",
        "Vandalismo constante em paragens de autocarro.",
        "Carros a circular em ruas pedonais.",
        "Falta patrulhamento da polícia em bairros afastados.",
        "Pinturas e grafitis ofensivos nos muros da escola.",
        "Zona escura e perigosa à noite.",
        "Agressões entre jovens perto da escola."
    ],
    "Comércio e Atividades Económicas": [
        "Muitos estabelecimentos encerrados no centro.",
        "Vendedores ambulantes ilegais na feira semanal.",
        "Falta de fiscalização no mercado municipal.",
        "Empresas sem licença a operar em zona residencial.",
        "Barulho excessivo de cafés durante a noite.",
        "Feira de velharias bloqueia as ruas.",
        "O mercado precisa de obras e organização.",
        "Pequenos negócios sem apoio municipal.",
        "Falta de espaço para comércio local.",
        "Fecho precoce das lojas aos fins de semana."
    ],
    "Outros": [
        "Sugestão: criar uma aplicação para reportar problemas.",
        "Gostaria de ter mais transparência nas decisões da junta.",
        "As informações do site da câmara estão desatualizadas.",
        "Falta sinalização turística no centro da cidade.",
        "O festival cultural precisa de melhor divulgação.",
        "Sugestão de instalar painéis solares nos edifícios públicos.",
        "Proposta: criar hortas urbanas nos bairros.",
        "Falta de comunicação com os moradores.",
        "Proponho mais assembleias participativas.",
        "Necessário apoio à cultura e artistas locais."
    ]
}

# Construir o dataset
data = []
for label, textos in topicos_queixas.items():
    for texto in textos:
        data.append({"text": texto, "label": label})

df = pd.DataFrame(data)


le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"])

# Save label mapping to a text file
with open("iVedras_label_mapping.txt", "w", encoding="utf-8") as f:
    for idx, label in enumerate(le.classes_):
        f.write(f"{idx}: {label}\n")

# 2. Criar dataset Hugging Face
dataset = Dataset.from_pandas(df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
dataset = dataset.train_test_split(test_size=0.2)

# 3. Tokenizer
model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)

# 4. Modelo
num_labels = len(le.classes_)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# 5. Treinamento
training_args = TrainingArguments(
    output_dir="output",
    learning_rate=1e-5,  # Lower learning rate for better convergence
    per_device_train_batch_size=4,  # Smaller batch size can help generalization
    per_device_eval_batch_size=4,
    num_train_epochs=10,  # More epochs for better learning
    weight_decay=0.05,  # Slightly higher weight decay to reduce overfitting
    save_total_limit=2,
    eval_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",  # Save at the end of each epoch
    load_best_model_at_end=True,  # Load best model according to eval loss
    metric_for_best_model="eval_loss",
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
model.save_pretrained("iVedras_queixas")
tokenizer.save_pretrained("iVedras_queixas")

# Também podes guardar o LabelEncoder
import joblib
joblib.dump(le, "iVedras_encoder.pkl")
