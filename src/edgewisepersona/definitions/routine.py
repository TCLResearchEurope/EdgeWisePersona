import json

from pydantic import BaseModel
from typing import List

from edgewisepersona.definitions.device_state import Actions
from edgewisepersona.definitions.triggers import Triggers


class Routine(BaseModel):
    triggers: Triggers
    actions: Actions


class RoutinesList(BaseModel):
    routines: List[Routine]


ROUTINE_SCHEMA = json.dumps(Routine.model_json_schema(), indent=2)
ROUTINES_LIST_SCHEMA = json.dumps(RoutinesList.model_json_schema(), indent=2)
