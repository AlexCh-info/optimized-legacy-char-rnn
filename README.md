# Async Streaming Refactor: char-rnn.pytorch

Рефакторинг legacy-реализации char-level RNN ([spro/char-rnn.pytorch](https://github.com/spro/char-rnn.pytorch), ~2017) под асинхронный, потоковый инференс — без изменения архитектуры или весов модели.

## Задача

Оригинальный `generate()` — синхронная функция, которая:
1. Инициализирует `hidden` state с нуля при каждом вызове.
2. Прогоняет весь `prime_str` через модель ("прогрев"), чтобы восстановить контекст.
3. Генерирует `predict_len` новых символов и возвращает **всё сразу**, одним блоком.

Это работает для одноразовой генерации, но ломается в сценарии стриминга: если нужно отдавать символы по мере готовности (например, много последовательных вызовов с растущим контекстом), каждый новый вызов вынужден **заново** прогонять весь накопленный текст через priming-цикл. Сложность растёт линейно с длиной контекста — O(n) на каждый новый чанк вместо O(1).

Дополнительно: как у любого синхронного PyTorch-инференса, вызов `decoder(...)` блокирует поток целиком — в асинхронном сервисе это означает, что во время генерации event loop не может обслуживать другие задачи/клиентов.

## Что изменено

### 1. Явное состояние вместо "спрятанного" внутри функции

**Было** (`generate.py`):
```python
def generate(decoder, prime_str='A', predict_len=100, temperature=0.8, cuda=False):
    hidden = decoder.init_hidden(1)          # <- пересоздаётся с нуля при КАЖДОМ вызове
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))
    ...
    for p in range(len(prime_str) - 1):       # <- прогрев целиком внутри одного вызова
        _, hidden = decoder(prime_input[:,p], hidden)

    inp = prime_input[:,-1]

    for p in range(predict_len):              # <- вся генерация тоже внутри одного вызова
        output, hidden = decoder(inp, hidden)
        ...
    return predicted                          # <- возвращает ВЕСЬ текст сразу, одним блоком
```

`hidden` и `inp` — локальные переменные функции. Они не переживают возврат из `generate()`, поэтому единственный способ "продолжить" генерацию — вызвать функцию заново с более длинным `prime_str`, заново оплатив весь прогрев.

**Стало** (`new_generate.py`):
```python
class Generation:
    def __init__(self, decoder, cuda: bool = False) -> None:
        self.decoder = decoder
        self.hidden = self.decoder.init_hidden(1)   # <- создаётся один раз на сессию
        self.cuda = cuda

    async def warm_up(self, prime_str: str) -> None:
        prime_input = Variable(char_tensor(prime_str).unsqueeze(0))
        if self.cuda:
            self.hidden = self.hidden.cuda()
            prime_input = prime_input.cuda()
        self.inp = await asyncio.to_thread(self._warm_up_with_no_grad, len(prime_str), prime_input)

    async def step(self, temperature: float = 0.8) -> str:
        if self.cuda:
            self.inp = self.inp.cuda()
        output, self.hidden = await asyncio.to_thread(self._forward_with_no_grad)
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1).item()
        predicted_char = all_characters[top_i]
        self.inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        return predicted_char                       # <- возвращает ОДИН новый символ
```

`hidden` и `inp` теперь — атрибуты объекта (`self.hidden`, `self.inp`). Они живут столько же, сколько живёт сама сессия `Generation`, и переживают множественные вызовы `step()`. Прогрев (`warm_up`) выполняется **один раз** за сессию, а не при каждом шаге генерации.

### 2. Разделение "прогрева" и "шага генерации"

Оригинальная функция объединяла priming-цикл и generation-цикл в одном вызове. Новый класс разносит их по разным методам с разной частотой вызова:

| | Legacy `generate()` | `Generation` класс |
|---|---|---|
| Прогрев контекста | Каждый вызов | Один раз (`warm_up`) |
| Генерация символа | Все `predict_len` сразу | По одному (`step`) |
| Возврат | Весь текст целиком | Один новый символ |
| Стоимость повторного вызова | O(n), n = длина контекста | O(1) |

### 3. Неблокирующий инференс через `asyncio.to_thread`

Сам forward-pass (`self.decoder(...)`) остаётся синхронным (PyTorch inference не переписан на async) — но вызов вынесен в отдельный поток через `asyncio.to_thread`, чтобы не блокировать event loop на время вычисления:

```python
output, self.hidden = await asyncio.to_thread(self._forward_with_no_grad)
```

Оборачивается **весь** цикл прогрева одним вызовом (а не поэлементно), поскольку с точки зрения внешнего кода прогрев атомарен — промежуточное состояние никого не интересует, и дробить его на множество потоков было бы избыточно.

### 4. Отключение autograd для инференса

```python
@torch.no_grad
def _forward_with_no_grad(self):
    return self.decoder(self.inp, self.hidden)
```

Учтён нюанс: `torch.no_grad()` — thread-local переключатель. Если бы `no_grad()` был обёрнут вокруг `await asyncio.to_thread(...)` снаружи, флаг выставился бы в потоке event loop, а не в worker-потоке, где реально исполняется `self.decoder(...)` — и не сработал бы. Поэтому `no_grad` применяется декоратором прямо на функции, которая передаётся в `to_thread` и исполняется внутри целевого потока. Дополнительно, при загрузке модели у всех параметров выставляется `requires_grad = False` — защита на уровне модели, не зависящая от потоков.

### 5. Изоляция состояния = потокобезопасность "бесплатно"

Поскольку `hidden`/`inp` — атрибуты **экземпляра**, а не общие/глобальные переменные, несколько параллельных объектов `Generation` (например, по одному на клиента) с одним и тем же `decoder` не создают race condition: веса модели read-only и разделяются свободно, а мутируемое состояние генерации у каждого клиента своё.

## Результаты бенчмарка

`benchmark_generation.py` сравнивает три сценария на реальной обученной модели (GRU, `hidden_size=128`, `n_layers=2`).

**1. Legacy `generate()` при растущем `prime_str` (имитация стриминга поверх старой функции):**

| prime_len | elapsed_sec |
|---|---|
| 4 | 0.0044 |
| 104 | 0.0276 |
| 204 | 0.0539 |
| 304 | 0.0799 |
| 404 | 0.1058 |
| 454 | 0.1192 |

Линейный рост — подтверждённый O(n) bottleneck.

**2. Новый класс — `warm_up` один раз + 10× `step()`:**

`warm_up`: 0.0014 сек. Каждый `step()`: ~0.0007–0.0009 сек, **без роста** от шага к шагу.

| step_idx | elapsed_sec|
|---|---|
| 1 | 0.0014 |
| 2 | 0.0009 |
| 3 | 0.0009 |
| 4 | 0.0008 |
| 5 | 0.0008 |

**Итог:** 10 шагов "стриминга" — 0.605 сек (legacy) против ~0.01 сек (новый класс) — **~60x** на этом сценарии, разрыв растёт с увеличением контекста.

**3. Параллельные клиенты (`asyncio.gather`, 3 клиента, общий `decoder`):**

Три независимых, неповторяющихся результата при одинаковом `prime_str` — эмпирическое подтверждение отсутствия гонки состояния между сессиями.

## Структура проекта

```
model.py               — архитектура CharRNN (GRU/LSTM), без изменений
helpers.py              — токенизация текста, без изменений
train.py                — обучение модели, без изменений (кроме .item() фикса под новый PyTorch)
generate.py             — оригинальная синхронная generate() [legacy, для сравнения]
new_generate.py         — новый класс Generation: async, стриминговый, потокобезопасный
benchmark_generation.py — сравнение legacy vs новый класс + тест параллельных клиентов
```

## Известные ограничения / TODO

- `warm_up("")` с пустой строкой вызовет `IndexError` при попытке взять `prime_input[:, -1]` — валидация входа не реализована.
- Очередь (`asyncio.Queue`) для доставки символов внешним потребителям (например, WebSocket) не входит в этот слой — `Generation` отвечает только за инкрементальный инференс, оркестрация доставки — отдельная задача.
- Профилирование проводилось на CPU; на CUDA `time.perf_counter()` без `torch.cuda.synchronize()` может давать неточные цифры из-за асинхронности CUDA-вызовов.
