from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
    ReasoningEffort
)
import json

def json_to_harmony_string(json_data):
    """
    Converts a generic conversation JSON into OpenAI Harmony format.

    json_data: dict corresponding to the kwargs of `client.chat.completions.create`, with keys like:
        - model
        - messages: list of {role, content, tool_calls?, reasoning?, ...}
        - tools: list of {type, function:{name, description, parameters}}
        - reasoning_effort
        - seed
    """

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Map reasoning effort
    effort_map = {
        "low": ReasoningEffort.LOW,
        "medium": ReasoningEffort.MEDIUM,
        "high": ReasoningEffort.HIGH
    }
    reasoning_effort = effort_map.get(json_data.get("reasoning_effort", "").lower(),
                                      ReasoningEffort.MEDIUM)

    # Create system content (you can adjust model identity and other defaults)
    system_message = (
        SystemContent.new()
        .with_model_identity(None)
        .with_knowledge_cutoff(None)
        .with_reasoning_effort(reasoning_effort)
        .with_channel_config(None)
    )

    # Convert tools to Harmony ToolDescription
    tool_descriptions = []
    for t in json_data.get("tools", []):
        if t.get("type") == "function":
            fn = t.get("function", {})
            tool_descriptions.append(
                ToolDescription.new(
                    fn.get("name", ""),
                    fn.get("description", ""),
                    parameters=fn.get("parameters")
                )
            )

    developer_message = (
        DeveloperContent.new()
        .with_function_tools(tool_descriptions)
    )

    # Convert messages
    harmony_messages = [
        Message.from_role_and_content(Role.SYSTEM, system_message),
        Message.from_role_and_content(Role.DEVELOPER, developer_message)
    ]

    for m in json_data.get("messages", []):
        role_map = {
            "system": Role.SYSTEM,
            "developer": Role.DEVELOPER,
            "user": Role.USER,
            "assistant": Role.ASSISTANT,
            "tool": Role.TOOL
        }
        role = role_map.get(m.get("role", "").lower(), Role.USER)

        if role == Role.ASSISTANT and m.get("content"):
            msg = Message.from_role_and_content(role, m.get("content", ""))
            harmony_messages.append(msg)

        # If there's reasoning, store it as an analysis channel
        if m.get("reasoning"):
            msg = Message.from_role_and_content(role, m.get("reasoning", "")).with_channel("analysis")
            harmony_messages.append(msg)

        # If it's a tool call
        if m.get("tool_calls"):
            # Treat as commentary channel to the tool
            for call in m["tool_calls"]:
                msg = Message.from_role_and_content(role, json.dumps(call['function']['arguments'])).with_channel("commentary")
                msg = msg.with_recipient(f"functions.{call['function']['name']}")
                msg = msg.with_content_type("json")
                harmony_messages.append(msg)

        # If it's a tool response
        if role == Role.TOOL:
            author = Author.new(Role.TOOL, m.get("name"))
            msg = Message.from_author_and_content(author, m.get("content", ""))
            msg = msg.with_recipient("assistant").with_channel("commentary")
            harmony_messages.append(msg)

    # Construct the conversation
    convo = Conversation.from_messages(harmony_messages)
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    return json.dumps(encoding.decode_utf8(encoding.render_conversation(convo)))
