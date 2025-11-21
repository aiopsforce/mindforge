from enum import Enum

class MemoryType(Enum):
    """
    Enumeration of available memory levels/types in MindForge.
    """
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    USER_SPECIFIC = "user"
    SESSION_SPECIFIC = "session"
    AGENT_SPECIFIC = "agent"
    PERSONA = "persona"
    TOOLBOX = "toolbox"
    CONVERSATION = "conversation"
    WORKFLOW = "workflow"
    EPISODIC = "episodic"
    AGENT_REGISTRY = "agent_registry"
    ENTITY = "entity"
