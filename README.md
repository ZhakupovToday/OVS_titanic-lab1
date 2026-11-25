# Лабораторная работа 1. Синхронный инференс: выживание пассажиров «Титаника»

Веб-сервис на FastAPI по данным пассажира предсказывает, выжил бы он при крушении «Титаника».

- **Модель:** градиентный бустинг (`GradientBoostingClassifier`), бинарная классификация.
- **Данные:** `data/titanic.csv` (891 пассажир).
- **Признаки:** `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`.
- **Качество на отложенной выборке:** accuracy ≈ 0.80, ROC AUC ≈ 0.82.
- **Пайплайн:** в `models/pipeline.pkl` сохраняется весь пайплайн целиком. В него входят заполнение пропусков, стандартизация, one-hot кодирование и сам классификатор, поэтому сервис принимает **сырые** данные, в том числе с пропусками.

## Структура

```
data/            датасет
notebooks/       titanic.ipynb: анализ данных, сравнение моделей, сохранение пайплайна
src/train.py     то же обучение в виде скрипта
src/model_utils.py  загрузка пайплайна и инференс
src/main.py      FastAPI-сервис
test/            юнит-тесты (pytest)
.github/workflows/service-testing.yml   CI: pycodestyle + pytest + обучение
```

## Запуск

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-notebook.txt     # или requirements.txt без Jupyter

# 1. Обучить модель: прогнать все ячейки notebooks/titanic.ipynb или
PYTHONPATH=./src python src/train.py

# 2. Тесты
PYTHONPATH=./:./src pytest -v test

# 3. Сервис
cd src && MODEL_PATH=../models/pipeline.pkl uvicorn main:app --port 8000
```

Интерактивная документация: http://localhost:8000/docs

## API

| Метод | Ресурс | Авторизация | Ответ |
|---|---|---|---|
| GET | `/healthcheck` | нет | `200 {"status": "ok"}` |
| POST | `/predictions` | `Bearer 00000` | `200 {"survived": bool, "probability": float}` |

Тело запроса `/predictions`:

| Поле | Тип | Обязательное | Допустимые значения |
|---|---|---|---|
| `pclass` | int | да | 1, 2, 3 |
| `sex` | str | да | `male`, `female` |
| `age` | float | нет | 0–120 |
| `sibsp` | int | да | ≥ 0 (братья/сёстры, супруги на борту) |
| `parch` | int | да | ≥ 0 (родители, дети на борту) |
| `fare` | float | нет | ≥ 0 |
| `embarked` | str | нет | `S`, `C`, `Q` |

Коды ошибок: `401` — токена нет или он неверный, `422` — данные не прошли валидацию.

## Примеры запросов

```bash
curl http://localhost:8000/healthcheck

# Пассажирка первого класса → выжила
curl -X POST http://localhost:8000/predictions \
  -H "Authorization: Bearer 00000" -H "Content-Type: application/json" \
  -d '{"pclass": 1, "sex": "female", "age": 29, "sibsp": 0, "parch": 0, "fare": 211.34, "embarked": "S"}'

# Мужчина из третьего класса, возраст и порт неизвестны → не выжил
curl -X POST http://localhost:8000/predictions \
  -H "Authorization: Bearer 00000" -H "Content-Type: application/json" \
  -d '{"pclass": 3, "sex": "male", "sibsp": 0, "parch": 0}'

# Без токена → 401 Not authenticated
curl -X POST http://localhost:8000/predictions -H "Content-Type: application/json" \
  -d '{"pclass": 3, "sex": "male", "sibsp": 0, "parch": 0}'

# Неверный токен → 401 Invalid authentication credentials
curl -X POST http://localhost:8000/predictions \
  -H "Authorization: Bearer 12345" -H "Content-Type: application/json" \
  -d '{"pclass": 3, "sex": "male", "sibsp": 0, "parch": 0}'

# Неверные данные → 422
curl -X POST http://localhost:8000/predictions \
  -H "Authorization: Bearer 00000" -H "Content-Type: application/json" \
  -d '{"pclass": 5, "sex": "robot", "sibsp": 0, "parch": 0}'
```
