from types import MethodType, SimpleNamespace
import rclpy
from rai.communication.ros2 import ROS2Connector
from rai_inspection_agent.tools.speaker import ControlSpeakerAlarmTool


def test_control_speaker_alarm_tool_gas_leak_request():
    if not rclpy.ok():
        rclpy.init()
    connector = ROS2Connector(node_name="test_speaker_tool_gas")
    calls = []

    def fake_service_call(self, message, target, msg_type, timeout_sec):
        calls.append(
            {
                "payload": message.payload,
                "target": target,
                "msg_type": msg_type,
                "timeout_sec": timeout_sec,
            }
        )
        return SimpleNamespace(
            payload=(
                "rcl_interfaces.srv.SetParameters_Response("
                "results=[rcl_interfaces.msg.SetParametersResult(successful=True, reason=''), "
                "rcl_interfaces.msg.SetParametersResult(successful=True, reason='')])"
            )
        )

    connector.service_call = MethodType(fake_service_call, connector)
    try:
        tool = ControlSpeakerAlarmTool(connector=connector)
        result = tool._run(command="gas_leak")
    finally:
        connector.shutdown()

    assert calls == [
        {
            "payload": {
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
            },
            "target": "/alarm_aggregator_node/set_parameters",
            "msg_type": "rcl_interfaces/srv/SetParameters",
            "timeout_sec": 5.0,
        }
    ]
    assert result["status"] == "succeeded"
    assert result["command"] == "gas_leak"
    assert "SetParameters_Response" in result["response"]


def test_control_speaker_alarm_tool_temperature_abnormal_request():
    if not rclpy.ok():
        rclpy.init()
    connector = ROS2Connector(node_name="test_speaker_tool_camera")
    calls = []

    def fake_service_call(self, message, target, msg_type, timeout_sec):
        calls.append(message.payload)
        return SimpleNamespace(
            payload=(
                "rcl_interfaces.srv.SetParameters_Response("
                "results=[rcl_interfaces.msg.SetParametersResult(successful=True, reason=''), "
                "rcl_interfaces.msg.SetParametersResult(successful=True, reason='')])"
            )
        )

    connector.service_call = MethodType(fake_service_call, connector)
    try:
        tool = ControlSpeakerAlarmTool(connector=connector)
        result = tool._run(command="temperature_abnormal")
    finally:
        connector.shutdown()

    assert calls == [
        {
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
    ]
    assert result["status"] == "succeeded"
    assert result["command"] == "temperature_abnormal"


def test_control_speaker_alarm_tool_stop_request():
    if not rclpy.ok():
        rclpy.init()
    connector = ROS2Connector(node_name="test_speaker_tool_stop")
    calls = []

    def fake_service_call(self, message, target, msg_type, timeout_sec):
        calls.append(message.payload)
        return SimpleNamespace(
            payload=(
                "rcl_interfaces.srv.SetParameters_Response("
                "results=[rcl_interfaces.msg.SetParametersResult(successful=True, reason='')])"
            )
        )

    connector.service_call = MethodType(fake_service_call, connector)
    try:
        tool = ControlSpeakerAlarmTool(connector=connector)
        result = tool._run(command="stop")
    finally:
        connector.shutdown()

    assert calls == [
        {
            "parameters": [
                {
                    "name": "play",
                    "value": {"type": 1, "bool_value": False},
                }
            ]
        }
    ]
    assert result["status"] == "succeeded"
    assert result["command"] == "stop"


def test_control_speaker_alarm_tool_handles_service_exception():
    if not rclpy.ok():
        rclpy.init()
    connector = ROS2Connector(node_name="test_speaker_tool_error")

    def fake_service_call(self, message, target, msg_type, timeout_sec):
        raise RuntimeError("service unavailable")

    connector.service_call = MethodType(fake_service_call, connector)
    try:
        tool = ControlSpeakerAlarmTool(connector=connector)
        result = tool._run(command="gas_leak")
    finally:
        connector.shutdown()

    assert result["status"] == "failed"
    assert result["command"] == "gas_leak"
    assert result["response"] == ""
    assert "RuntimeError: service unavailable" == result["error_message"]
