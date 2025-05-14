import json
import random

from pydantic import BaseModel, conint
from typing import Optional, Literal


class TVSettings(BaseModel):
    volume: Optional[int] = None
    brightness: Optional[int] = None
    input_source: Optional[Literal["HDMI1", "HDMI2", "AV", "Netflix", "YouTube"]] = None


class ACSettings(BaseModel):
    temperature: Optional[conint(ge=16, le=30)] = None
    mode: Optional[Literal["cool", "heat", "auto"]] = None
    fan_speed: Optional[conint(ge=0, le=3)] = None


class LightsSettings(BaseModel):
    brightness: Optional[int] = None
    color: Optional[Literal["warm", "cool", "neutral"]] = None
    mode: Optional[Literal["static", "dynamic"]] = None


class SpeakerSettings(BaseModel):
    volume: Optional[int] = None
    equalizer: Optional[Literal["bass boost", "balanced", "treble boost"]] = None


class SecuritySettings(BaseModel):
    armed: Optional[bool] = None
    alarm_volume: Optional[int] = None


class Actions(BaseModel):
    tv: Optional[TVSettings] = None
    ac: Optional[ACSettings] = None
    lights: Optional[LightsSettings] = None
    speaker: Optional[SpeakerSettings] = None
    security: Optional[SecuritySettings] = None

ACTIONS_SCHEMA  = json.dumps(Actions.model_json_schema(), indent=2)


class DeviceState(BaseModel):
    tv: TVSettings
    ac: ACSettings
    lights: LightsSettings
    speaker: SpeakerSettings
    security: SecuritySettings

DEVICE_SCHEMA  = json.dumps(DeviceState.model_json_schema(), indent=2)


def random_device_state() -> DeviceState:
    return DeviceState(
        tv=TVSettings(
            volume=random.randint(0, 100),
            brightness=random.randint(0, 100),
            input_source=random.choice(["HDMI1","HDMI2","AV","Netflix"]),
        ),
        ac=ACSettings(
            temperature=random.randint(16, 30),
            mode=random.choice(["cool","heat","auto"]),
            fan_speed=random.randint(0, 3),
        ),
        lights=LightsSettings(
            brightness=random.randint(0, 100),
            color=random.choice(["warm","cool","neutral"]),
            mode=random.choice(["static","dynamic"]),
        ),
        speaker=SpeakerSettings(
            volume=random.randint(0, 100),
            equalizer=random.choice(["bass boost","balanced","treble boost"]),
        ),
        security=SecuritySettings(
            armed=random.choice([True, False]),
            alarm_volume=random.randint(0, 100),
        ),
    )
