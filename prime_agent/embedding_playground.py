import uuid
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
import cfg_common

import uuid
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
import cfg_common
from datetime import datetime
import hashlib

# Инициализация модели и клиента
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = GigaChatEmbeddings(
    credentials=cfg_common.giga_api_key,
    verify_ssl_certs=False,
)



# Инициализация модели и клиента
persistent_client = chromadb.PersistentClient()
collection_name = 'test_2'

# Создаем или получаем коллекцию
collection = persistent_client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
)

# Инициализация векторного хранилища LangChain
vector_store = Chroma(
    client=persistent_client,
    collection_name=collection_name,
    embedding_function=embeddings,
)

product_list = [" прайм+ ", " старт ", " прайм "]


def keywords_match(query, document):
    '''
    Нужно чтобы даже если предложения близки по смыслу, то учесть нюансы
    т.к. !!1даже один+!!! может полностью изменить контекст
    все следующие предложения очень близки по смыслу (с точки зрения embedding),
    но нам надо, чтобы одинаковыми были только:
        "чем отличается пакет прайм от старт"
        "чем отличается пакет прайм от прайм старт"
            "чем отличается пакет прайм+ от прайм старт"
            "чем отличается пакет прайм+ от старт"
        "чем отличается прайм от прайм+"
        "чем отличается прайм+ от прайм"
        "
    '''

    def normalize_keywords(text):
        if "прайм старт" in text:
            text = text.replace("прайм старт", "старт")
        return text

    query = f' {normalize_keywords(query)} '
    document = f' {normalize_keywords(document)} '

    query_keywords = set(keyword for keyword in product_list if keyword in query)
    doc_keywords = set(keyword for keyword in product_list if keyword in document)

    return query_keywords == doc_keywords


def generate_content_hash(text: str) -> str:
    """
    Генерирует хэш содержимого для уникальной идентификации.

    Args:
        text: Текст для хэширования

    Returns:
        str: Хэш строки
    """
    return hashlib.md5(text.encode()).hexdigest()


def check_if_exists_in_db(
        query: str,
        similarity_threshold: float = 0.1,
        k: int = 1,
        search_by_content: bool = False
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Проверяет, есть ли похожая запись в базе данных.

    Args:
        query: Текст запроса для проверки
        similarity_threshold: Порог схожести (чем меньше, тем строже)
        k: Количество результатов для возврата
        search_by_content: Если True, ищет по всему содержимому, а не только по query

    Returns:
        Tuple[bool, Optional[Dict]]:
        - Флаг существования записи
        - Содержимое найденной записи или None
    """
    try:
        # Ищем похожие записи
        results = vector_store.similarity_search_with_score(
            query,
            k=k
        )

        # Если есть результаты и они достаточно похожи
        if results:
            for res, score in results:
                # Используем порог схожести - чем меньше score, тем больше похоже
                if score < similarity_threshold:
                    return True, {
                        'page_content': res.page_content,
                        'metadata': res.metadata,
                        'similarity_score': score,
                        'id': res.metadata.get('id', 'unknown') if res.metadata else 'unknown',
                        'content_data': res.metadata.get('content_data', '') if res.metadata else ''
                    }

        return False, None

    except Exception as e:
        print(f"Ошибка при проверке записи в БД: {e}")
        return False, None


def add_new_record_to_db(
        query: str,
        content_data: Optional[str] = None,
        additional_data: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        custom_id: Optional[str] = None,
        check_duplicates: bool = True
) -> str:
    """
    Добавляет новую запись в базу данных с текстовым содержимым.

    Args:
        query: Текст запроса для поиска/индексации
        content_data: Основное текстовое содержимое (ответ, документ, статья и т.д.)
        additional_data: Дополнительные данные в виде словаря
        metadata: Дополнительные метаданные для Chroma
        custom_id: Пользовательский ID (если None, будет сгенерирован)
        check_duplicates: Проверять ли дубликаты перед добавлением

    Returns:
        str: ID добавленной записи
    """
    try:
        # Проверяем дубликаты если нужно
        if check_duplicates:
            exists, existing_record = check_if_exists_in_db(query)
            if exists:
                print(f"Запись уже существует в БД: {existing_record}")
                return existing_record['id'] if existing_record else "existing_record"

        # Генерируем ID если не предоставлен
        record_id = custom_id if custom_id else str(uuid.uuid4())

        # Подготавливаем полное содержимое для сохранения
        full_content = query

        # Добавляем content_data если есть
        if content_data:
            full_content = f"QUERY: {query}\n\nCONTENT: {content_data}"

        # Подготавливаем метаданные
        record_metadata = metadata.copy() if metadata else {}
        record_metadata['id'] = record_id
        record_metadata['query'] = query
        record_metadata['timestamp'] = datetime.now().isoformat()
        record_metadata['content_hash'] = generate_content_hash(full_content)

        # Сохраняем дополнительные данные в метаданные
        if content_data:
            record_metadata['content_data'] = content_data
            record_metadata['has_content'] = True
            record_metadata['content_length'] = len(content_data)
        else:
            record_metadata['has_content'] = False

        # Добавляем additional_data в метаданные
        if additional_data:
            for key, value in additional_data.items():
                if isinstance(value, (str, int, float, bool)):
                    record_metadata[f'data_{key}'] = str(value)

        # Добавляем запись
        collection.add(
            ids=[record_id],
            documents=[full_content],  # Сохраняем полное содержимое
            embeddings=[embeddings.embed_query(query)],  # Индексируем по query
            metadatas=[record_metadata]
        )

        print(f"Запись успешно добавлена с ID: {record_id}")
        if content_data:
            print(f"Добавлено содержимое длиной {len(content_data)} символов")
        return record_id

    except Exception as e:
        print(f"Ошибка при добавлении записи в БД: {e}")
        return ""


def add_document_to_db(
        title: str,
        content: str,
        document_type: Optional[str] = None,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        additional_info: Optional[Dict] = None,
        custom_id: Optional[str] = None
) -> str:
    """
    Специализированная функция для добавления документов.

    Args:
        title: Заголовок/название документа
        content: Содержимое документа
        document_type: Тип документа (статья, новость, инструкция и т.д.)
        source: Источник документа
        tags: Список тегов
        additional_info: Дополнительная информация
        custom_id: Пользовательский ID

    Returns:
        str: ID добавленного документа
    """
    # Подготавливаем метаданные
    metadata = {}

    if document_type:
        metadata['document_type'] = document_type
    if source:
        metadata['source'] = source
    if tags:
        metadata['tags'] = ','.join(tags)

    # Добавляем additional_info если есть
    additional_data = additional_info.copy() if additional_info else {}

    return add_new_record_to_db(
        query=title,  # Используем заголовок как поисковый запрос
        content_data=content,
        additional_data=additional_data,
        metadata=metadata,
        custom_id=custom_id
    )


def add_qa_pair_to_db(
        question: str,
        answer: str,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        source_url: Optional[str] = None,
        custom_id: Optional[str] = None
) -> str:
    """
    Специализированная функция для добавления вопросов и ответов.

    Args:
        question: Вопрос
        answer: Ответ
        category: Категория (техническая, общая и т.д.)
        difficulty: Сложность (легкая, средняя, сложная)
        source_url: URL источника
        custom_id: Пользовательский ID

    Returns:
        str: ID добавленной пары
    """
    metadata = {
        'record_type': 'qa_pair',
    }

    if category:
        metadata['category'] = category
    if difficulty:
        metadata['difficulty'] = difficulty
    if source_url:
        metadata['source_url'] = source_url

    additional_data = {
        'question': question,
        'answer': answer,
        'qa_length': len(question) + len(answer)
    }

    # Формируем содержимое для индексации
    content = f"ВОПРОС: {question}\n\nОТВЕТ: {answer}"

    return add_new_record_to_db(
        query=question,  # Индексируем по вопросу
        content_data=content,
        additional_data=additional_data,
        metadata=metadata,
        custom_id=custom_id
    )


def search_with_content(
        query: str,
        k: int = 5,
        min_similarity: float = 0.0,
        max_similarity: float = 0.5,
        filter_metadata: Optional[Dict] = None
) -> List[Dict[str, Any]]:
    """
    Ищет похожие записи и возвращает их с содержимым.

    Args:
        query: Текст запроса для поиска
        k: Количество результатов для возврата
        min_similarity: Минимальный порог схожести
        max_similarity: Максимальный порог схожести
        filter_metadata: Фильтр по метаданным

    Returns:
        List[Dict]: Список найденных записей с содержимым
    """
    try:
        # Подготавливаем фильтр для Chroma
        where_filter = None
        if filter_metadata:
            where_filter = {}
            for key, value in filter_metadata.items():
                where_filter[key] = value

        # Ищем записи
        if where_filter:
            results = vector_store.similarity_search_with_score(
                query,
                k=k,
                filter=where_filter
            )
        else:
            results = vector_store.similarity_search_with_score(
                query,
                k=k
            )

        found_records = []
        for res, score in results:
            # Фильтрация по порогам схожести
            if score < min_similarity or score > max_similarity:
                continue

            # Извлекаем содержимое из метаданных
            content_data = res.metadata.get('content_data', '') if res.metadata else ''

            # Если нет content_data, пробуем извлечь из page_content
            if not content_data and 'CONTENT:' in res.page_content:
                try:
                    # Извлекаем содержимое из форматированного текста
                    parts = res.page_content.split('CONTENT:', 1)
                    if len(parts) > 1:
                        content_data = parts[1].strip()
                except:
                    content_data = res.page_content

            record = {
                'id': res.metadata.get('id', 'unknown') if res.metadata else 'unknown',
                'query': res.metadata.get('query', '') if res.metadata else '',
                'content_data': content_data,
                'metadata': res.metadata,
                'similarity_score': float(score),
                'page_content': res.page_content,
                'has_content': res.metadata.get('has_content', False) if res.metadata else False
            }
            found_records.append(record)

        # Сортируем по схожести (меньший score = больше похоже)
        found_records.sort(key=lambda x: x['similarity_score'])
        return found_records

    except Exception as e:
        print(f"Ошибка при поиске записей: {e}")
        return []


def get_record_by_id(record_id: str) -> Optional[Dict[str, Any]]:
    """
    Получает запись по ID.

    Args:
        record_id: ID записи

    Returns:
        Optional[Dict]: Запись или None
    """
    try:
        result = collection.get(
            ids=[record_id],
            include=['documents', 'metadatas', 'embeddings']
        )

        if result['ids']:
            metadata = result['metadatas'][0] if result['metadatas'] else {}
            document = result['documents'][0] if result['documents'] else ""

            return {
                'id': record_id,
                'document': document,
                'metadata': metadata,
                'content_data': metadata.get('content_data', ''),
                'query': metadata.get('query', ''),
                'embedding': result['embeddings'][0] if result['embeddings'] else None
            }
        return None

    except Exception as e:
        print(f"Ошибка при получении записи по ID: {e}")
        return None


def update_record_content(
        record_id: str,
        new_content: str,
        update_query: bool = False,
        new_query: Optional[str] = None
) -> bool:
    """
    Обновляет содержимое существующей записи.

    Args:
        record_id: ID записи для обновления
        new_content: Новое содержимое
        update_query: Обновлять ли также поисковый запрос
        new_query: Новый поисковый запрос (если update_query=True)

    Returns:
        bool: Успешность обновления
    """
    try:
        # Получаем текущую запись
        current_record = get_record_by_id(record_id)
        if not current_record:
            print(f"Запись с ID {record_id} не найдена")
            return False

        # Подготавливаем новое содержимое
        current_metadata = current_record['metadata']
        current_query = new_query if update_query and new_query else current_metadata.get('query', '')

        # Обновляем метаданные
        current_metadata['content_data'] = new_content
        current_metadata['content_length'] = len(new_content)
        current_metadata['last_updated'] = datetime.now().isoformat()
        current_metadata['content_hash'] = generate_content_hash(new_content)

        # Формируем полное содержимое
        full_content = f"QUERY: {current_query}\n\nCONTENT: {new_content}"

        # Обновляем запись
        collection.update(
            ids=[record_id],
            documents=[full_content],
            embeddings=[embeddings.embed_query(current_query)] if update_query else None,
            metadatas=[current_metadata]
        )

        print(f"Запись {record_id} успешно обновлена")
        return True

    except Exception as e:
        print(f"Ошибка при обновлении записи: {e}")
        return False


# Пример использования функций
if __name__ == "__main__":
    print("=== Пример использования расширенного API для работы с Chroma DB ===\n")

    # 1. Добавление записи с содержимым
    print("1. Добавление записи с содержимым:")
    record_id = add_new_record_to_db(
        query="Что такое искусственный интеллект?",
        content_data="Искусственный интеллект (ИИ) — это область компьютерных наук, "
                     "которая занимается созданием систем, способных выполнять задачи, "
                     "требующие человеческого интеллекта. К таким задачам относятся обучение, "
                     "распознавание образов, понимание естественного языка и принятие решений.",
        additional_data={
            "category": "технологии",
            "author": "ИИ эксперт",
            "language": "русский"
        },
        metadata={
            "source": "википедия",
            "year": "2023"
        }
    )
    print(f"Добавлена запись с ID: {record_id}\n")

    # 2. Добавление документа
    print("2. Добавление документа:")
    doc_id = add_document_to_db(
        title="Введение в машинное обучение",
        content="Машинное обучение — это подраздел искусственного интеллекта, "
                "который позволяет компьютерам обучаться на данных без явного программирования. "
                "Основные типы машинного обучения: обучение с учителем, "
                "обучение без учителя и обучение с подкреплением.",
        document_type="учебный материал",
        source="Coursera",
        tags=["ML", "обучение", "данные"],
        additional_info={"pages": 15, "difficulty": "средняя"}
    )
    print(f"Добавлен документ с ID: {doc_id}\n")

    # 3. Добавление пары вопрос-ответ
    print("3. Добавление пары вопрос-ответ:")
    qa_id = add_qa_pair_to_db(
        question="Какие есть типы нейронных сетей?",
        answer="Основные типы нейронных сетей: полносвязные нейронные сети (FCN), "
               "сверточные нейронные сети (CNN) для обработки изображений, "
               "рекуррентные нейронные сети (RNN) для последовательных данных, "
               "и трансформеры для обработки естественного языка.",
        category="нейронные сети",
        difficulty="средняя",
        source_url="https://example.com/neural-nets"
    )
    print(f"Добавлена QA пара с ID: {qa_id}\n")

    # 4. Поиск с содержимым
    print("4. Поиск записей с содержимым:")
    search_results = search_with_content(
        query="машинное обучение",
        k=3,
        max_similarity=0.3
    )

    for i, result in enumerate(search_results, 1):
        print(f"{i}. Score: {result['similarity_score']:.5f}")
        print(f"   Запрос: {result['query'][:50]}...")
        print(f"   Содержимое: {result['content_data'][:100]}...")
        print(f"   ID: {result['id']}")
        print()

    # 5. Получение записи по ID
    print("5. Получение записи по ID:")
    if record_id:
        record = get_record_by_id(record_id)
        if record:
            print(f"Найдена запись: {record['query'][:50]}...")
            print(f"Длина содержимого: {len(record.get('content_data', ''))} символов")

    # 6. Обновление содержимого
    print("\n6. Обновление содержимого записи:")
    if record_id:
        updated = update_record_content(
            record_id=record_id,
            new_content="Обновленное содержимое об искусственном интеллекте. "
                        "ИИ теперь включает также генеративные модели и трансформеры.",
            update_query=True,
            new_query="Что включает современный искусственный интеллект?"
        )
        if updated:
            print("Содержимое успешно обновлено")