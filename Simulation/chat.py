from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Tuple
if TYPE_CHECKING:
    from .player import Player
    from .enums import Time

class ChannelOpenState(Enum):
    READ = auto()
    WRITE = auto()

class ChatChannelType(Enum):
    DAY_PUBLIC = auto()
    MAFIA_NIGHT = auto()
    COVEN_NIGHT = auto()
    VAMPIRE_NIGHT = auto()
    JAILED = auto()
    DEAD = auto()
    WHISPER = auto()  # one-off dynamic channels; key = (src_id,dst_id,day)

class ChatMessage:
    def __init__(self, sender: 'Player', message: str, channel_type: ChatChannelType, *, is_environment: bool = False):
        self.sender = sender
        self.message = message
        self.channel_type = channel_type
        self.is_environment = is_environment  # True for system/death messages

    def __repr__(self):
        prefix = "[ENV]" if self.is_environment else f"[{self.channel_type.name}]"
        sender_name = "SYSTEM" if self.is_environment else self.sender.name
        return f"{prefix} {sender_name}: {self.message}"

class ChatChannel:
    """Light wrapper holding message history and r/w flags."""
    def __init__(self, channel_type: ChatChannelType):
        self.channel_type = channel_type
        self.messages: List[ChatMessage] = []
        # open state per player_id: {id: {READ, WRITE}}
        self.members: Dict[int, set[ChannelOpenState]] = {}

    # ------------------------------------------------------------------
    def add_member(self, player: 'Player', *, can_write: bool = True, can_read: bool = True):
        states = set()
        if can_read:
            states.add(ChannelOpenState.READ)
        if can_write:
            states.add(ChannelOpenState.WRITE)
        self.members.setdefault(player.id, set()).update(states)

    def remove_member(self, player: 'Player'):
        self.members.pop(player.id, None)

    def broadcast(self, sender: 'Player', text: str, *, is_environment: bool = False):
        self.messages.append(ChatMessage(sender, text, self.channel_type, is_environment=is_environment))

    def get_visible(self, player: 'Player') -> List[ChatMessage]:
        if ChannelOpenState.READ not in self.members.get(player.id, set()):
            return []
        return self.messages

class ChatHistory:
    """Stores chat history for a specific day/night period."""
    def __init__(self, day: int, is_night: bool):
        self.day = day
        self.is_night = is_night
        self.messages: List[ChatMessage] = []
        self.whispers: List[ChatMessage] = []

    def add_message(self, message: ChatMessage):
        if message.channel_type == ChatChannelType.WHISPER:
            self.whispers.append(message)
        else:
            self.messages.append(message)

    def get_period_name(self) -> str:
        return f"Night {self.day}" if self.is_night else f"Day {self.day}"

class ChatManager:
    def __init__(self):
        # Static channels
        self.channels: Dict[ChatChannelType, ChatChannel] = {t: ChatChannel(t) for t in ChatChannelType if t != ChatChannelType.WHISPER}
        # dynamic whispers: key -> ChatChannel
        self.whispers: Dict[Tuple[int,int,int], ChatChannel] = {}
        
        # Historical chat storage: key = (day, is_night) -> ChatHistory
        self.history: Dict[Tuple[int, bool], ChatHistory] = {}
        
        # Current period tracking
        self.current_day: int = 0
        self.current_is_night: bool = False

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------
    def start_new_period(self, day: int, is_night: bool):
        """Start a new day/night period, archiving previous messages."""
        # Archive current messages if this isn't the very first period
        if hasattr(self, 'current_day'):
            self._archive_current_messages()
        
        # Clean up old history (remove periods from 2+ days ago)
        self._cleanup_old_history(day)
        
        # Update current period
        self.current_day = day
        self.current_is_night = is_night
        
        # Clear current message buffers for new period
        for channel in self.channels.values():
            channel.messages.clear()
        self.whispers.clear()

    def _archive_current_messages(self):
        """Archive all current messages to history."""
        period_key = (self.current_day, self.current_is_night)
        history = ChatHistory(self.current_day, self.current_is_night)
        
        # Archive all channel messages
        for channel in self.channels.values():
            for msg in channel.messages:
                history.add_message(msg)
        
        # Archive whisper messages
        for channel in self.whispers.values():
            for msg in channel.messages:
                history.add_message(msg)
        
        self.history[period_key] = history

    def _cleanup_old_history(self, current_day: int):
        """Remove chat history from 2+ days ago."""
        keys_to_remove = []
        for (day, is_night) in self.history.keys():
            if day < current_day - 1:  # Remove anything older than previous day
                keys_to_remove.append((day, is_night))
        
        for key in keys_to_remove:
            del self.history[key]

    def get_chat_history(self, player: 'Player', day: int, is_night: bool) -> str:
        """Get formatted chat history for a specific day/night period."""
        period_key = (day, is_night)
        history = self.history.get(period_key)
        
        if not history:
            period_name = f"Night {day}" if is_night else f"Day {day}"
            return f"No chat history available for {period_name}."
        
        # Filter messages the player can see based on channel permissions
        visible_messages = []
        
        for msg in history.messages:
            # Check if player had read access to this channel type
            if self._player_could_see_channel(player, msg.channel_type, day, is_night):
                visible_messages.append(msg)
        
        # Add whispers the player was involved in
        for whisper in history.whispers:
            if whisper.sender.id == player.id or any(
                member_id == player.id for member_id in self._get_whisper_participants(whisper)
            ):
                visible_messages.append(whisper)
        
        # Sort by insertion order (using object id as proxy)
        visible_messages.sort(key=id)
        
        # Format the history
        if not visible_messages:
            return f"No visible messages from {history.get_period_name()}."
        
        formatted = [f"=== {history.get_period_name()} Chat History ==="]
        for msg in visible_messages:
            if msg.is_environment:
                formatted.append(f"[ENV] {msg.message}")
            elif msg.channel_type == ChatChannelType.WHISPER:
                # Don't show whisper content in history, just indicate it happened
                other_participant = "someone" if msg.sender.id == player.id else msg.sender.name
                formatted.append(f"[WHISPER] Whispered with {other_participant}")
            else:
                formatted.append(f"{msg.sender.name}: {msg.message}")
        
        return "\n".join(formatted)

    def _player_could_see_channel(self, player: 'Player', channel_type: ChatChannelType, day: int, is_night: bool) -> bool:
        """Determine if a player could see a channel during a specific period."""
        if not player.is_alive:
            return channel_type == ChatChannelType.DEAD
        
        if channel_type == ChatChannelType.DAY_PUBLIC:
            return True  # All living players can see day public
        elif channel_type == ChatChannelType.DEAD:
            return player.role.name.value == "Medium"  # Only mediums can see dead chat
        elif channel_type == ChatChannelType.MAFIA_NIGHT:
            return is_night and player.role.faction.name == "MAFIA"
        elif channel_type == ChatChannelType.COVEN_NIGHT:
            return is_night and player.role.faction.name == "COVEN"
        elif channel_type == ChatChannelType.VAMPIRE_NIGHT:
            return is_night and player.role.faction.name == "VAMPIRE"
        elif channel_type == ChatChannelType.JAILED:
            return False  # Jail messages are private to that night only
        
        return False

    def _get_whisper_participants(self, whisper: ChatMessage) -> List[int]:
        """Extract participant IDs from a whisper message (placeholder)."""
        # This would need to be enhanced to track whisper participants properly
        return []

    # ------------------------------------------------------------------
    # Environment message helpers
    # ------------------------------------------------------------------
    def add_environment_message(self, message: str, channel_type: ChatChannelType = ChatChannelType.DAY_PUBLIC):
        """Add a system/environment message to the specified channel."""
        # Create a dummy player for environment messages
        env_player = type('EnvironmentPlayer', (), {'name': 'SYSTEM', 'id': -1})()
        channel = self.channels[channel_type]
        channel.broadcast(env_player, message, is_environment=True)

    # ------------------------------------------------------------------
    # Membership helpers
    # ------------------------------------------------------------------
    def move_player_to_channel(self, player: 'Player', channel_type: ChatChannelType, *, write: bool = True, read: bool = True):
        # ensure channel exists
        if channel_type == ChatChannelType.WHISPER:
            raise ValueError("WHISPER channels are dynamic; use create_whisper_channel")
        chan = self.channels[channel_type]
        chan.add_member(player, can_write=write, can_read=read)

    def remove_player_from_channel(self, player: 'Player', channel_type: ChatChannelType):
        chan = self.channels.get(channel_type)
        if chan:
            chan.remove_member(player)

    # ------------------------------------------------------------------
    # Speaking APIs
    # ------------------------------------------------------------------
    def send_speak(self, player: 'Player', text: str) -> ChatMessage | str:
        """Player speaks in whichever static channel they have WRITE perms.

        If they are in none, returns an error string."""
        # find first writable channel (there should be exactly one in practice)
        for chan in self.channels.values():
            if ChannelOpenState.WRITE in chan.members.get(player.id, set()):
                chan.broadcast(player, text)
                return chan.messages[-1]
        return "Error: you cannot speak right now."

    def send_whisper(self, src: 'Player', dst: 'Player', text: str, *, day: int, is_night: bool) -> ChatMessage | str:
        """Create (or fetch) a private WHISPER channel and deliver text.

        • Whispers are day-only; at night returns error.
        • Dead ↔ living whispers are invalid.
        """
        if is_night:
            return "Error: You cannot whisper at night."
        if not src.is_alive or not dst.is_alive:
            return "Error: Whisper target must be alive."
        key = (src.id, dst.id, day) if src.id < dst.id else (dst.id, src.id, day)
        chan = self.whispers.setdefault(key, ChatChannel(ChatChannelType.WHISPER))
        chan.add_member(src, can_write=True, can_read=True)
        chan.add_member(dst, can_write=True, can_read=True)
        chan.broadcast(src, text)
        return chan.messages[-1]

    # ------------------------------------------------------------------
    def get_visible_messages(self, player: 'Player') -> List[ChatMessage]:
        """Get all currently visible messages for a player (current period only)."""
        msgs: List[ChatMessage] = []
        for chan in self.channels.values():
            msgs.extend(chan.get_visible(player))
        for chan in self.whispers.values():
            msgs.extend(chan.get_visible(player))
        # order by insertion (id of obj stable)
        return sorted(msgs, key=id) 