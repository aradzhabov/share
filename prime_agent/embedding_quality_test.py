'''
!!на windows для некоторых комбинаций железа падает молча
    нужно сделать так: pip install chromadb==0.5.0 chroma-hnswlib==0.7.3.
# metadata={"hnsw:space": "cosine"}) - МЕГАВАЖНО т.к. по умолчанию там другая метрика и она плохо работает


две очень полезные ссылки
https://python.langchain.com/docs/integrations/vectorstores/chroma/
https://api.python.langchain.com/en/latest/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html

1) Qwen2.5-Plus подготовка данных
    1.1) сформируй таблицу в следующем формате:
    столбец один это идентификатор группы вопросов, столбец два это качество ответа системы. столбец два заполнять не надо.
    придумай 50 вопросов по документу и для каждого вопроса придумай еще по два вопроса, которые сформулированы другими словами, но полностью идентичные по смыслу. для каждого набора используй уникальный идентификатор группы вопросов например 1, 2, 3.
    Подготовь данные в формате csv
    2.1) перемести вопросы в строки и используй одинаковый id для похожей группы вопросов

2) Mistral обработка данных
преобразуй код для решения следующей задачи:
нужно открыть excel файл. В файле есть следующие колонки:
sim_group_id - идентификатор группы схожих по смыслу вопросов
question - текст вопроса
is_in_vector_db - признак того, что необходимо данный вопрос загрузить в коллекцию векторной базы
sim_value - коэффициент похожести с поступившим запросом
returned_docs - дополнительный признак

нужно для каждого значения поля query для которого is_in_vector_db  = 1 занести в коллекцию на которую указывает переменная collection_name и сохранить в метаданных каждого документа коллекции значение поля sim_group_id

далее нужно для каждой записи  для значения в поле question  выполнить vector_store.similarity_search_with_score   и занести в поле sim_value  соответствующее query полученное значение score, а в поле returned_docs  занести из документа полученного в результате запроса его res.metadata и res.page_content

отформатируй значение  score  чтобы было пять знаков после запятой
добавь столбец is_fail и установи значение 1 если sim_group_id  в метаданных полученного из векторной базы документа не совпадает с sim_group_id записи

Далее идет код который нужно модифицировать:
import uuid
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
persistent_client = chromadb.PersistentClient()
collection_name = 'test_1'
collection = persistent_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})


query = 'просто тест'
query_id = str(uuid.uuid4())
collection.add(
    ids=[query_id],
    documents=[query],
    embeddings=[embeddings.embed_query(query)],
    metadatas=[{"query": query}]
)

vector_store = Chroma(
    client=persistent_client,
    collection_name=collection_name,
    embedding_function=embeddings,
)

print(f"-----{query}")
results = vector_store.similarity_search_with_score(
    query, k=1,
)
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

'''

'''
Итого:
fail/pass
    9/140  (paid remote) giga t=~0,60 сек секможно отсечь еще 4ре если не учитывать как правильные все, что выше 0,084
    10/140 (free local run) multilingual-e5-large t=~0,10 сек 2GB
    30/140 (free local run) rubert-tiny2 118 MB
    43/140 (free local run) sbert_large_nlu_ru  t=~0,9 сек 2GB
    52/140 (free local run) all-MiniLM-L6-v2
'''

import uuid
import pandas as pd
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
import cfg_common
import time

# Инициализация модели и клиента
collection_name = 'xxx'
file_with_data_for_test = '1.xlsx' #'embedding_data_for_test.xlsx'
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# embeddings = HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2")
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
# embeddings = HuggingFaceEmbeddings(model_name="ai-forever/sbert_large_nlu_ru")

# GIGA
embeddings = GigaChatEmbeddings(
    credentials=cfg_common.giga_api_key,
    verify_ssl_certs=False,
)

persistent_client = chromadb.PersistentClient()

collection = persistent_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

# Загрузка данных из Excel
df = pd.read_excel(file_with_data_for_test)

# Добавление вопросов в векторную базу данных
for index, row in df.iterrows():
    if row['is_in_vector_db'] == 1:
        query_id = str(uuid.uuid4())
        collection.add(
            ids=[query_id],
            documents=[row['question']],
            embeddings=[embeddings.embed_query(row['question'])],
            metadatas=[{"sim_group_id": row['sim_group_id']}]
        )

# Инициализация векторного хранилища
vector_store = Chroma(
    client=persistent_client,
    collection_name=collection_name,
    embedding_function=embeddings,
)

# Добавление нового столбца is_fail
df['is_fail'] = 0

# Выполнение поиска похожих вопросов и обновление данных
for index, row in df.iterrows():

    start_time = time.time()
    results = vector_store.similarity_search_with_score(row['question'], k=1)
    end_time = time.time()
    print(f"(smx) Время выполнения: {end_time - start_time:.2f} seconds")

    for res, score in results:
        df.at[index, 'sim_value'] = f"{score:.5f}"  # Форматирование score с пятью знаками после запятой
        df.at[index, 'returned_docs'] = f"{res.metadata} {res.page_content}"
        if res.metadata.get('sim_group_id') != row['sim_group_id']:
            df.at[index, 'is_fail'] = 1

# Сохранение обновленных данных обратно в Excel
df.to_excel(f'embedding_{collection_name}_test_v2.xlsx', index=False)
