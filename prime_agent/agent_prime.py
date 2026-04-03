import cfg_common
from typing import Dict
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, ToolCall, AIMessage
from langchain_gigachat.chat_models import GigaChat
from langchain_core.tools import tool
from langchain_gigachat.tools.giga_tool import giga_tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
import time
from colorama import Fore, Style

import itertools
import threading

import embedding_playground as em

is_print_context = False
skip_system_message_when_print_context = False
cur_subscription_name = 'СберПрайм'
attached_document_sber_prime_info = ['41bb8217-bd4a-4b17-8949-39d352dedbfb']

path = cfg_common.ad_hoc_train_data
with open(path, 'r', encoding='utf-8') as file:
    prime_general_info = f"{file.read()}"

few_shot_examples = [
    {
        "request": "Расскажи о других пакетах",
        "params": {},
    },
    {
        "request": "Чем отличается старт от прайм+",
        "params": {},
    },
    {
        "request": "Чем отличается пакеты",
        "params": {},
    },
    {
        "request": "какова стоимость пакетов",
        "params": {},
    },
    {
        "request": "какие еще есть пакеты",
        "params": {},
    },
]

# пример для подписки СберПрайм
service_utilization = [
    {
        "music": {
            "value": 1,
            "description": "Звук: Музыка в HiFi-качестве"
        },
        "movie": {
            "value": 0,
            "description": "Okko: Десятки тысяч фильмов и сериалов"
        },
        "taxi": {
            "value": 1,
            "description": "Ситидрайв: Кешбэк бонусами до 10% за поездки"
        },
        "cashback_category": {
            "value": 0,
            "description": "Спасибо: 5 категорий повышенного кешбэка бонусами"
        },
        "savings_account_account_opened": {
            "value": 0,
            "description": "СБЕР: +1% к ставке СберВклада"
        }
    }
]

user_interests = [
    {
        "sport": "football",
        "team": "Спартак",
        "upcoming_matches": [
            {
                "match_date": "2026-12-15",
                "opponent": "Зенит",
                "location": "Москва, Открытие Арена"
            },
            {
                "match_date": "2026-12-22",
                "opponent": "ЦСКА",
                "location": "Москва, Лужники"
            },
            {
                "match_date": "2026-12-25",
                "opponent": "ЦСКА",
                "location": "онлайн сервис ОККО"
            }
        ]
    }
]


@giga_tool(few_shot_examples=few_shot_examples)
def get_prime_general_info() -> str:
    """
    Возвращает общую информацию о пакетах Старт, СберПрайм и СберПрайм+

    Returns:
        str: вся общая информация о пакетах Старт, СберПрайм и СберПрайм+ и их составляющих
    """
    return prime_general_info


@tool
def get_utilization() -> Dict:
    """
    Возвращает информацию о уже использованных и не использованных сервисах в пакете. Значение 0 означает, что сервис не использован

    Returns:
        Dict: Словарь с информацией о сервисах.
    """
    return service_utilization[0]


@tool
def get_interests() -> Dict:
    """
    Возвращает информацию о интересах пользователя и ближайших событиях которые могут быть ему интересны

    Returns:
        Dict: Словарь с информацией об интересах и ближайших событиях которые могут быть интересны клиенту.
    """
    return user_interests[0]


giga = GigaChat(
    credentials=cfg_common.giga_api_key,
    model=cfg_common.model,
    verify_ssl_certs=False,
    streaming=False,
    scope=cfg_common.scope,
)


def print_context(messages, skip_system_message=False):
    # https: // python.langchain.com / api_reference / core / messages.html
    print(f"Контекст:")
    for i, message in enumerate(messages, start=1):
        if isinstance(message, HumanMessage):
            message_type = "HumanMessage"
        elif isinstance(message, SystemMessage):
            message_type = "SystemMessage"
            if skip_system_message:
                continue
        elif isinstance(message, ToolMessage):
            message_type = "ToolMessage"
        elif isinstance(message, AIMessage):
            message_type = "AIMessage"
        elif isinstance(message, ToolCall):
            message_type = "ToolCall"
        else:
            message_type = "Другое"

        print(f"    {i}. {message_type}: {message.content} {message.additional_kwargs}")
        if False == skip_system_message:
            print(messages)


if cfg_common.take_care_about_cur_user_subscription_utilization_and_prefs:
    system_prompt = (f"Сегодня {datetime.now().strftime('Cегодня %d %B %Y года. текущее время %H:%M')}. "
                     f"Ты — бот, помогающий клиенту с информацией о его пакете услуг {cur_subscription_name} и информирующий клиента о ближайших интересных ему событиях. Ты консультируешь по пакетам Старт, СберПрайм и СберПрайм+. Далее описание трех пакетов СберПрайм: {prime_general_info}. В своих ответах учитывай, что у клиента подключен пакет  {cur_subscription_name}")
    tools = [get_utilization, get_interests]
else:
    system_prompt = (f"{datetime.now().strftime('Cегодня %d %B %Y года. текущее время %H:%M')}. "
                     f"Ты — бот. Ты консультируешь по пакетам Старт, СберПрайм и СберПрайм+. Далее описание трех пакетов СберПрайм: {prime_general_info}. Отвечай только на основе предоставленного выше описания пакетов. Если нет нужной информации, говори, что уточню информацию по вашему запросу и добавлю в свою базу знаний.")
    tools = []

agent = create_react_agent(giga,
                           tools=tools,
                           checkpointer=MemorySaver(),
                           prompt=system_prompt)


def chat(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    welcome_message = """
      ╭───────────────────╮
      │  Добро пожаловать │      /\\___/\\
      │  в СберПрайм R&D! │     ( =^.^= )       
      ╰───────────────────╯    (,,)❤️(,,)
    """

    # имитация исходного вызова
    if cfg_common.take_care_about_cur_user_subscription_utilization_and_prefs:
        print(
            f"{Fore.GREEN}СберПрайм: {Style.RESET_ALL} Добрый день! У Вас подключен пакет {Fore.GREEN}{cur_subscription_name}{Style.RESET_ALL}. Я помогу Вам с информацией о текущем пакете и общей информацией о других пакетах")
    else:
        print(
            f"{welcome_message}\n{Fore.GREEN} Добрый день! Я помогу Вам с общей информацией о пакетах Прайм, помогу сравнить пакеты и сделать правильный выбор!\n")

    if cfg_common.take_care_about_cur_user_subscription_utilization_and_prefs:
        resp = agent.invoke({"messages": [("user", 'Расскажи какими сервисами я еще не воспользовался')]},
                            config=config)
        print(f"{Fore.GREEN}СберПрайм: {Style.RESET_ALL}",
              f'{Fore.BLUE}Обратите внимание: {Style.RESET_ALL}{resp["messages"][-1].content}')

        resp = agent.invoke({"messages": [("user", 'какие события мне могут быть интересны')]}, config=config)
        print(f"{Fore.GREEN}СберПрайм: {Style.RESET_ALL}",
              f'{Fore.MAGENTA}Обратите внимание: {Style.RESET_ALL}{resp["messages"][-1].content}')

    frames = ["🐱", "🐱\\", "\\🐱", "\\🐱/", " ", "/🐱"]

    def animate_loading():
        for c in itertools.cycle(frames):
            if done_loading:
                break
            print(f'\r{Fore.GREEN}⏳... {c}{Style.RESET_ALL} "тут я могу отражать дополнительную полезную информацию"',
                  end='', flush=True)
            time.sleep(0.1)
        print('\r', end='', flush=True)

    while (True):
        rq = input(f"{' ' * 10}{Fore.CYAN}Вы: {Style.RESET_ALL}")
        if rq == "":
            break

        # Создаем флаг для завершения анимации
        global done_loading
        done_loading = False

        # Замеряем время начала обработки запроса
        start_time = time.time()

        # Запускаем анимацию в отдельном потоке
        loading_thread = threading.Thread(target=animate_loading)
        loading_thread.start()

        try:
            exists = False
            existing_record = None
            if cfg_common.use_cached_answers:
                exists, existing_record = em.check_if_exists_in_db(rq, cfg_common.similarity_threshold)

            if exists:
                answer = existing_record.get('page_content')
            else:
                resp = agent.invoke({"messages": [("user", rq)]}, config=config)
                answer = resp["messages"][-1].content
                if is_print_context:
                    print_context(resp, skip_system_message=skip_system_message_when_print_context)
                if cfg_common.use_cached_answers:
                    em.add_new_record_to_db(rq, answer)
        finally:
            done_loading = True
            loading_thread.join()

            # Замеряем время окончания и вычисляем длительность ответа
            end_time = time.time()
            response_time = end_time - start_time

        # Выводим ответ вместе с временем выполнения
        print(f"({exists}):{' ' * 2}{Fore.GREEN}СберПрайм: {Style.RESET_ALL}", f'{answer}')
        print(f"{' ' * 12}{Fore.YELLOW}⏱️ Время ответа: {response_time:.2f} секунд{Style.RESET_ALL}")


chat("1")