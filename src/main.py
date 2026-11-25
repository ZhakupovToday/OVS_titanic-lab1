# -*- coding: utf-8 -*-
"""Сервис синхронного инференса: выживет ли пассажир «Титаника»."""
import os
from functools import lru_cache
from typing import Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

import model_utils


class Passenger(BaseModel):
    """Сырые данные пассажира. age, fare и embarked могут отсутствовать."""
    pclass: Literal[1, 2, 3] = Field(description="Класс билета")
    sex: Literal["male", "female"]
    age: Optional[float] = Field(None, ge=0, le=120, description="Возраст")
    sibsp: int = Field(ge=0, description="Братья/сёстры и супруги на борту")
    parch: int = Field(ge=0, description="Родители и дети на борту")
    fare: Optional[float] = Field(None, ge=0, description="Стоимость билета")
    embarked: Optional[Literal["S", "C", "Q"]] = Field(
        None, description="Порт посадки: Southampton, Cherbourg, Queenstown")

    model_config = {"json_schema_extra": {"examples": [{
        "pclass": 1, "sex": "female", "age": 29, "sibsp": 0, "parch": 0,
        "fare": 211.34, "embarked": "S"}]}}


class Prediction(BaseModel):
    survived: bool
    probability: float = Field(description="Вероятность выживания")


model_path: Optional[str] = os.getenv("MODEL_PATH")
if model_path is None:
    raise ValueError("The environment variable $MODEL_PATH is empty!")

API_TOKEN: str = os.getenv("API_TOKEN", "00000")

app = FastAPI(title="Titanic survival inference",
              description="Синхронный инференс модели выживания пассажиров")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth")


@lru_cache(maxsize=1)
def get_model() -> Pipeline:
    """Модель загружается один раз, при первом запросе."""
    return model_utils.load_model(model_path)


async def check_token(token: str = Depends(oauth2_scheme)) -> None:
    if token != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predictions", response_model=Prediction,
          dependencies=[Depends(check_token)])
async def predictions(passenger: Passenger,
                      model: Pipeline = Depends(get_model)) -> Prediction:
    return Prediction(**model_utils.make_inference(model,
                                                   passenger.model_dump()))
