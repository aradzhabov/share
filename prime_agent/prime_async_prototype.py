cfg_common.py'''
 RAG&LLM
    + Когда основали компанию?
    - на сколько выросла чистая прибыль компании
    - за какой год отчет
    - номер 8-800 к кому относится

'''


import asyncio
import time
import uuid
from typing import Annotated, TypedDict, List, Optional, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver  # Для хранения истории диалога

from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import cfg_common
from langchain_gigachat.chat_models import GigaChat


from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import Literal
import chromadb

# --- 1. КОНФИГУРАЦИЯ ---
FIXED_SYSTEM_PROMPT = "Вы — официальный ассистент. Используйте предоставленные данные для ответа."
CHECK_INTERVAL = 3600  # 1 час

# Инициализация моделей (замените ключи/базовые URL если нужно)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
model = GigaChat(
    credentials=cfg_common.giga_api_key,
    model=cfg_common.model,
    verify_ssl_certs=False,
    streaming=False, # — необязательный параметр, который включает и отключает потоковую генерацию токенов. По умолчанию False. Потоковая генерация позволяет повысить отзывчивость интерфейса программы при работе с длинными текстами.
    scope=cfg_common.scope,
    # scope — необязательный параметр, в котором можно указать версию API. Возможные значения:
    #     GIGACHAT_API_PERS — версия API для физических лиц;
    #     GIGACHAT_API_B2B — доступ для ИП и юридических лиц по предоплате;
    #     GIGACHAT_API_CORP — доступ для ИП и юридических лиц по схеме pay-as-you-go
)


chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="qa_cache",
    metadata={"hnsw:space": "cosine"} # Переключаем на косинусное сходство
)

# Коллекция для RAG (знания)
rag_collection = chroma_client.get_or_create_collection(name="rag_storage", metadata={"hnsw:space": "cosine"})



# --- 2. УПРАВЛЕНИЕ ДАННЫМИ (8 Кб) ---
class AsyncGlobalDataContext:
    def __init__(self):
        self.cached_data = None
        self.current_version = None
        self.last_check_time = 0
        self.lock = asyncio.Lock()

    async def sync_data(self):
        async with self.lock:
            now = time.time()
            if now - self.last_check_time > CHECK_INTERVAL or self.cached_data is None:
                # Имитация получения документа (V1.05 - первые 5 символов)
                raw_doc = "V1.05\nквадраты в нашем дворе бывают красные и зеленые"
                new_version = raw_doc[:5]
                if new_version != self.current_version:
                    self.cached_data = raw_doc
                    self.current_version = new_version
                    try:
                        chroma_client.delete_collection("qa_cache")
                    except:
                        pass
                    global collection
                    collection = chroma_client.get_or_create_collection(name="qa_cache")
                self.last_check_time = now
            return self.cached_data


data_provider = AsyncGlobalDataContext()


# --- 3. СОСТОЯНИЕ ---
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    mode: Literal["RAG&LLM", "LOCAL&CACHE", "LOCAL&NO_CACHE"]  # Три режима
    current_info: str
    cache_hit: bool
    final_answer: Optional[str]


# --- 4. УЗЛЫ (NODES) ---

async def fetch_data_node(state: AgentState):
    """Всегда подтягиваем актуальные 8 Кб"""
    data = await data_provider.sync_data()
    return {"current_info": data}


async def check_cache_node(state: AgentState):
    """Безопасная проверка кэша"""
    if state["mode"] != "LOCAL&CACHE":
        return {"cache_hit": False}

    user_query = state["messages"][-1].content
    loop = asyncio.get_event_loop()

    # Эмбеддинг
    query_emb = await loop.run_in_executor(None, embeddings.embed_query, user_query)

    # Поиск
    results = await loop.run_in_executor(None, lambda: collection.query(
        query_embeddings=[query_emb],
        n_results=1
    ))

    # ПРОВЕРКА: есть ли вообще документы в ответе?
    distance_criteria = 0.35
    if results.get('documents') and len(results['documents'][0]) > 0:
        # Извлекаем дистанцию и ответ аккуратно
        distance = results['distances'][0][0]
        print(f"DEBUG: Дистанция до ближайшего ответа: {distance}, а дистанция до ближайшего должна быть < {distance_criteria}")
        if distance < distance_criteria:
            print(f"✅ Найдено в кэше (dist: {distance:.4f})")
            answer = results['metadatas'][0][0]['answer']
            return {"final_answer": answer, "cache_hit": True}

    print("❌ В кэше ничего не найдено или база пуста")
    return {"cache_hit": False}


async def llm_engine_node(state: AgentState):
    """Основной движок генерации ответа"""
    if state.get("cache_hit"): return {}

    user_query = state["messages"][-1].content
    mode = state["mode"]

    # Определяем контекст в зависимости от режима
    if mode == "RAG&LLM":
        # Имитация RAG поиска
        context = "Данные из внешней RAG-базы..."
        sys_info = "Используй данные RAG."
    else:
        # Режимы LOCAL (CACHE или NO_CACHE)
        context = state["current_info"]
        sys_info = "Используй локальные 8 Кб данные."

    prompt = [
        SystemMessage(content=f"{FIXED_SYSTEM_PROMPT} {sys_info}"),
        *state["messages"][:-1],
        HumanMessage(content=f"КОНТЕКСТ:\n{context}\n\nВОПРОС: {user_query}")
    ]

    response = await model.ainvoke(prompt)

    # Сохраняем в кэш ТОЛЬКО если режим LOCAL&CACHE
    if mode == "LOCAL&CACHE":
        loop = asyncio.get_event_loop()
        query_emb = await loop.run_in_executor(None, embeddings.embed_query, user_query)
        await loop.run_in_executor(None, lambda: collection.add(
            embeddings=[query_emb],
            documents=[user_query],
            metadatas=[{"answer": response.content}],
            ids=[str(uuid.uuid4())]
        ))

    return {"messages": [response], "final_answer": response.content}


async def rag_engine_node(state: AgentState):
    """Настоящая эмуляция RAG: поиск знаний + генерация от批вета"""
    user_query = state["messages"][-1].content
    loop = asyncio.get_event_loop()

    # 1. Поиск релевантных знаний в RAG-коллекции
    query_emb = await loop.run_in_executor(None, embeddings.embed_query, user_query)
    rag_results = await loop.run_in_executor(None, lambda: rag_collection.query(
        query_embeddings=[query_emb],
        n_results=2  # Берем 2 самых похожих документа
    ))

    # Собираем найденные куски в одну строку
    retrieved_docs = "\n".join(rag_results['documents'][0]) if rag_results['documents'] else "Информации не найдено."
    print(f"🔎 [RAG] Найдено в базе: {retrieved_docs[:100]}...")

    # 2. Формируем промпт для LLM с найденным контекстом
    prompt = [
        SystemMessage(content=f"{FIXED_SYSTEM_PROMPT} Ты используешь данные из ВНЕШНЕГО АРХИВА (RAG)."),
        *state["messages"][:-1],
        HumanMessage(content=f"НАЙДЕННЫЕ ЗНАНИЯ:\n{retrieved_docs}\n\nВОПРОС: {user_query}")
    ]


    response = await model.ainvoke(prompt)

    return {
        "messages": [response],
        "final_answer": response.content,
    }


# --- 5. СБОРКА ГРАФА ---

workflow = StateGraph(AgentState)

workflow.add_node("fetcher", fetch_data_node)
workflow.add_node("cache_lookup", check_cache_node)
workflow.add_node("llm_engine", llm_engine_node)
workflow.add_node("rag_engine", rag_engine_node)


workflow.add_edge(START, "fetcher")


def route_start(state: AgentState):
    if state["mode"] == "RAG&LLM": return "rag"
    if state["mode"] == "LOCAL&NO_CACHE": return "llm"
    return "cache"

workflow.add_conditional_edges("fetcher", route_start, {"rag": "rag_engine", "llm": "llm_engine", "cache": "cache_lookup"})
workflow.add_conditional_edges("cache_lookup", lambda s: "end" if s["cache_hit"] else "llm", {"end": END, "llm": "llm_engine"})
workflow.add_edge("llm_engine", END)
workflow.add_edge("rag_engine", END)


# # Развилка на старте
# def start_router(state: AgentState):
#     if state["mode"] == "RAG&LLM" or state["mode"] == "LOCAL&NO_CACHE":
#         return "direct_llm"
#     return "check_cache"
#
#
#
# workflow.add_conditional_edges(
#     "fetcher",
#     start_router,
#     {"direct_llm": "llm_engine", "check_cache": "cache_lookup"}
# )

# # Развилка после кэша
# workflow.add_conditional_edges(
#     "cache_lookup",
#     lambda s: "end" if s["cache_hit"] else "ask_llm",
#     {"end": END, "ask_llm": "llm_engine"}
# )
#
# workflow.add_edge("llm_engine", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
print(app.get_graph().draw_ascii())


async def prepare_rag_data():
    """Загрузка знаний в RAG-коллекцию"""
    if rag_collection.count() == 0:
        print("📥 [System] Наполнение RAG базы знаниями...")
        docs = [
            "Отчет 2023: Чистая прибыль выросла на 15% по сравнению с прошлым годом.",
            "Инструкция по безопасности: В здании запрещено курить и использовать нагреватели.",
            "Контакты: Техподдержка работает круглосуточно по номеру 8-800-555-35-35.",
            "История: Компания была основана в 2010 году в городе Иннополис."
        ]
        # Важно: добавляем списком!
        loop = asyncio.get_event_loop()
        # Генерируем эмбеддинги для всех доков сразу
        embs = await loop.run_in_executor(None, embeddings.embed_documents, docs)

        rag_collection.add(
            embeddings=embs,
            documents=docs,
            ids=[f"id_{i}" for i in range(len(docs))]
        )
        print(f"✅ [System] В RAG базу загружено {len(docs)} документов.")

# --- 6. ЗАПУСК ---

async def chat_loop():
    await prepare_rag_data()
    # Уникальный ID сессии для истории сообщений
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Режим по умолчанию
    current_mode = "LOCAL&CACHE"

    print(f"--- 🤖 Бот запущен (ID: {thread_id}) ---")
    print(f"--- Текущий режим: {current_mode} ---")
    print("Команды: '/mode [название]', 'exit' для выхода")
    print("Доступные режимы: RAG&LLM, LOCAL&CACHE, LOCAL&NO_CACHE")

    while True:
        try:
            user_text = input("\n👤 Вы: ").strip()
            if not user_text: continue

            # Команда смены режима
            if user_text.lower().startswith("/mode "):
                new_mode = user_text.split(" ")[1].upper()
                if new_mode in ["RAG&LLM", "LOCAL&CACHE", "LOCAL&NO_CACHE"]:
                    current_mode = new_mode
                    print(f"✅ Режим изменен на: {current_mode}")
                else:
                    print(f"❌ Неизвестный режим. Доступны: RAG&LLM, LOCAL&CACHE, LOCAL&NO_CACHE")
                continue

            if user_text.lower() in ["exit", "quit", "выход"]: break

            # Запуск графа с передачей ТЕКУЩЕГО режима
            inputs = {
                "messages": [HumanMessage(content=user_text)],
                "mode": current_mode
            }

            result = await app.ainvoke(inputs, config=config)

            # 1. Печать истории
            print("\n" + "=" * 30 + " КОНТЕКСТ ДИАЛОГА " + "=" * 30)
            for msg in result["messages"]:
                role = "USER" if isinstance(msg, HumanMessage) else "BOT"
                # Отрезаем 8кб контекст из логов истории, чтобы не спамить в консоль
                content = msg.content[:99] + "..." if len(msg.content) > 200 else msg.content
                print(f"[{role}]: {content}")
            print("=" * 78)

            # 2. Печать  ответа

            # Проверяем, откуда пришел ответ для наглядности
            source = "Источник ответа = КЭШ" if result.get("cache_hit") else current_mode
            print(f"🤖 Бот [{source}]: {result['final_answer']}")

        except Exception as e:
            print(f"⚠️ Ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(chat_loop())
