import json

from pydantic import BaseModel
from typing import List


class Character(BaseModel):
    character: str


class CharactersList(BaseModel):
    characters: List[Character]


CHARACTER_SCHEMA = json.dumps(Character.model_json_schema(), indent=2)
CHARACTERS_LIST_SCHEMA = json.dumps(CharactersList.model_json_schema(), indent=2)
