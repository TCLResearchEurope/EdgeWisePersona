import json

from pydantic import BaseModel
from typing import List, Literal, Optional


class SessionMeta(BaseModel):
    time_of_day: Literal["morning", "afternoon", "evening", "night"]
    day_of_week: Literal["weekday", "weekend"]
    sun_phase: Literal["before_sunrise", "daylight", "after_sunset"]
    weather: Literal["sunny", "cloudy", "rainy", "snowy"]
    outdoor_temp: Literal["very cold", "cold", "mild", "warm", "hot"]

SESSION_META_SCHEMA  = json.dumps(SessionMeta.model_json_schema(), indent=2)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    text: str


class Session(BaseModel):
    session_id: int = 0
    meta: SessionMeta
    messages: List[Message]
    applied_routines: Optional[List[int]] = None
    session_summarization: Optional[str] = None

SESSION_SCHEMA  = json.dumps(Session.model_json_schema(), indent=2)


class SessionsList(BaseModel):
    sessions: List[Session]

SESSIONS_LIST_SCHEMA  = json.dumps(SessionsList.model_json_schema(), indent=2)
