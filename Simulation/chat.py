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
    def __init__(self, sender: 'Player', message: str, channel_type: ChatChannelType):
        self.sender = sender
        self.message = message
        self.channel_type = channel_type

    def __repr__(self):
        return f"[{self.channel_type.name}] {self.sender.name}: {self.message}"

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

    def broadcast(self, sender: 'Player', text: str):
        self.messages.append(ChatMessage(sender, text, self.channel_type))

    def get_visible(self, player: 'Player') -> List[ChatMessage]:
        if ChannelOpenState.READ not in self.members.get(player.id, set()):
            return []
        return self.messages

class ChatManager:
    def __init__(self):
        # Static channels
        self.channels: Dict[ChatChannelType, ChatChannel] = {t: ChatChannel(t) for t in ChatChannelType if t != ChatChannelType.WHISPER}
        # dynamic whispers: key -> ChatChannel
        self.whispers: Dict[Tuple[int,int,int], ChatChannel] = {}

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
        msgs: List[ChatMessage] = []
        for chan in self.channels.values():
            msgs.extend(chan.get_visible(player))
        for chan in self.whispers.values():
            msgs.extend(chan.get_visible(player))
        # order by insertion (id of obj stable)
        return sorted(msgs, key=id) 