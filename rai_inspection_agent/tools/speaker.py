from typing import Any, Literal, Type
from pydantic import BaseModel, Field
from rai.communication.ros2 import ROS2Message
from rai.tools.ros2.base import BaseROS2Tool


class ControlSpeakerAlarmInput(BaseModel):
    command: Literal["gas_leak", "temperature_abnormal", "stop"] = Field(
        description=(
            "Speaker command to execute. "
            "'gas_leak' plays the gas leak alarm, "
            "'temperature_abnormal' plays the temperature abnormal alarm, "
            "and 'stop' stops playback."
        )
    )


class ControlSpeakerAlarmTool(BaseROS2Tool):
    name: str = "control_speaker_alarm"
    description: str = (
        "Control the inspection speaker alarm. Use 'gas_leak' to play "
        "the gas leak warning, 'temperature_abnormal' to play the temperature "
        "abnormal warning, and 'stop' to stop playback."
    )
    args_schema: Type[ControlSpeakerAlarmInput] = ControlSpeakerAlarmInput

    service_name: str = Field(default="/alarm_aggregator_node/set_parameters")
    service_type: str = Field(default="rcl_interfaces/srv/SetParameters")
    timeout_sec: float = Field(default=5.0, ge=0.1)

    def _run(self, command: Literal["gas_leak", "temperature_abnormal", "stop"]) -> dict[str, Any]:
        if not self.is_writable(self.service_name):
            raise ValueError(f"Service {self.service_name} is not writable")

        request = self._build_request(command)

        try:
            response = self.connector.service_call(
                ROS2Message(payload=request),
                self.service_name,
                msg_type=self.service_type,
                timeout_sec=self.timeout_sec,
            )
            return {
                "status": "succeeded",
                "command": command,
                "service_name": self.service_name,
                "request": request,
                "response": str(response.payload),
                "error_message": "",
            }
        except Exception as exc:
            return {
                "status": "failed",
                "command": command,
                "service_name": self.service_name,
                "request": request,
                "response": "",
                "error_message": f"{type(exc).__name__}: {exc}",
            }

    def _build_request(
        self, command: Literal["gas_leak", "temperature_abnormal", "stop"]
    ) -> dict[str, Any]:
        if command == "gas_leak":
            return {
                "parameters": [
                    {
                        "name": "alarm_category",
                        "value": {"type": 4, "string_value": "gas"},
                    },
                    {
                        "name": "play",
                        "value": {"type": 1, "bool_value": True},
                    },
                ]
            }
        if command == "temperature_abnormal":
            return {
                "parameters": [
                    {
                        "name": "alarm_category",
                        "value": {"type": 4, "string_value": "camera"},
                    },
                    {
                        "name": "play",
                        "value": {"type": 1, "bool_value": True},
                    },
                ]
            }
        return {
            "parameters": [
                {
                    "name": "play",
                    "value": {"type": 1, "bool_value": False},
                }
            ]
        }


def create_control_speaker_alarm_tool(
    connector: Any,
    service_name: str = "/alarm_aggregator_node/set_parameters",
    timeout_sec: float = 5.0,
) -> ControlSpeakerAlarmTool:
    return ControlSpeakerAlarmTool(
        connector=connector,
        service_name=service_name,
        timeout_sec=timeout_sec,
    )
