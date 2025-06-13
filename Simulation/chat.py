from enum import Enum, auto
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .player import Player

class ChatChannelType(Enum):
    PUBLIC = auto()
    MAFIA = auto()
    COVEN = auto()
    VAMPIRE = auto()
    JAILED = auto()
    GRAVEYARD = auto()

class ChatMessage:
    def __init__(self, sender: 'Player', message: str, channel_type: ChatChannelType):
        self.sender = sender
        self.message = message
        self.channel_type = channel_type

    def __repr__(self):
        return f"[{self.channel_type.name}] {self.sender.name}: {self.message}"

class ChatManager:
    def __init__(self):
        self.channels = {channel_type: [] for channel_type in ChatChannelType}
        self.player_channels = {} # {player_id: [channel_types]}

    def add_player_to_channel(self, player: 'Player', channel_type: ChatChannelType):
        if player.id not in self.player_channels:
            self.player_channels[player.id] = []
        if channel_type not in self.player_channels[player.id]:
            self.player_channels[player.id].append(channel_type)
            player.chat_channels.append(channel_type)

    def remove_player_from_channel(self, player: 'Player', channel_type: ChatChannelType):
        if player.id in self.player_channels and channel_type in self.player_channels[player.id]:
            self.player_channels[player.id].remove(channel_type)
            player.chat_channels.remove(channel_type)

    def send_message(self, sender: 'Player', message: str, channel_type: ChatChannelType):
        if channel_type not in self.player_channels.get(sender.id, []):
            print(f"Warning: {sender.name} is not in channel {channel_type.name}")
            return None
        
        chat_message = ChatMessage(sender, message, channel_type)
        self.channels[channel_type].append(chat_message)
        return chat_message

    def get_messages(self, player: 'Player'):
        visible_messages = []
        for channel_type in self.player_channels.get(player.id, []):
            visible_messages.extend(self.channels[channel_type])
        return sorted(visible_messages, key=lambda msg: id(msg)) # sort by creation order 