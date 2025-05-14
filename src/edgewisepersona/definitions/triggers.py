import json

from pydantic import BaseModel
from typing import Optional, Literal


class Triggers(BaseModel):
    time_of_day: Optional[Literal["morning", "afternoon", "evening", "night"]] = None
    day_of_week: Optional[Literal["weekday", "weekend"]] = None
    sun_phase: Optional[Literal["before_sunrise", "daylight", "after_sunset"]] = None
    weather: Optional[Literal["sunny", "cloudy", "rainy", "snowy"]] = None
    outdoor_temp: Optional[Literal["very cold", "cold", "mild", "warm", "hot"]] = None

TRIGGERS_SCHEMA  = json.dumps(Triggers.model_json_schema(), indent=2)
