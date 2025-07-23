from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Tuple
import time # Import time for timestamping
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
    MEDIUM_SEANCE = auto()  # Medium-dead player communication
    PLAYER_PRIVATE_NOTIFICATION = auto() # For player-specific notifications (e.g., roleblocked, doused)

class ChatMessage:
    _message_counter = 0 # For deterministic temporal sorting

    def __init__(self, sender: 'Player', message: str, channel_type: ChatChannelType, *, is_environment: bool = False):
        self.sender = sender
        self.message = message
        self.channel_type = channel_type
        self.is_environment = is_environment  # True for system/death messages
        self.timestamp = ChatMessage._message_counter # Use a counter for deterministic order
        ChatMessage._message_counter += 1

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
    def __init__(self, all_players=None, logger=None):
        # Store reference to all players for blackmailer whisper access
        self._all_players = all_players or []
        # Store reference to logger for chat logging
        self.logger = logger
        # Static channels
        self.channels: Dict[ChatChannelType, ChatChannel] = {t: ChatChannel(t) for t in ChatChannelType if t not in [ChatChannelType.WHISPER, ChatChannelType.PLAYER_PRIVATE_NOTIFICATION]}
        # dynamic whispers: key -> ChatChannel
        self.whispers: Dict[Tuple[int,int,int], ChatChannel] = {}
        
        # dynamic seances: key = (medium_id, target_id, day) -> ChatChannel
        self.seances: Dict[Tuple[int,int,int], ChatChannel] = {}
        
        # dynamic player-specific notification channels: key = player_id -> ChatChannel
        self.player_notifications_channels: Dict[int, ChatChannel] = {}
        
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
        self.seances.clear()
        # Clear player-specific notification channels
        for channel in self.player_notifications_channels.values():
            channel.messages.clear()

    def _archive_current_messages(self):
        """Archive all current messages to history, preserving temporal order."""
        period_key = (self.current_day, self.current_is_night)
        history = ChatHistory(self.current_day, self.current_is_night)
        
        all_messages_in_period: List[ChatMessage] = []

        # Collect messages from all channel types
        for channel in self.channels.values():
            all_messages_in_period.extend(channel.messages)
        for channel in self.whispers.values():
            all_messages_in_period.extend(channel.messages)
        for channel in self.seances.values():
            all_messages_in_period.extend(channel.messages)
        for channel in self.player_notifications_channels.values():
            all_messages_in_period.extend(channel.messages)
        
        # Sort all messages by their timestamp to preserve temporal order
        all_messages_in_period.sort(key=lambda msg: msg.timestamp)
        
        # Add sorted messages to history
        for msg in all_messages_in_period:
            history.add_message(msg)
        
        self.history[period_key] = history

    def _cleanup_old_history(self, current_day: int):
        """Remove chat history from 3+ days ago (keep two full days)."""
        keys_to_remove = []
        for (day, is_night) in self.history.keys():
            if day < current_day - 2:  # Remove anything older than two days ago
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
        
        # Add seances the player was involved in (they're also stored in whispers list)
        # Note: seances are archived with whispers in _archive_current_messages
        
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

    def get_multi_period_chat_history(self, player: 'Player', current_day: int, current_is_night: bool) -> str:
        """Get formatted chat history for the last two full days (all day and night periods), with phase/day separators."""
        # Determine which days to include
        # If current_day <= 2, just show all available history
        if current_day <= 2:
            days_to_show = list(range(1, current_day + 1))
        else:
            days_to_show = list(range(current_day - 2, current_day + 1))
        # For each day, include both day and night periods if present
        periods = []
        for day in days_to_show:
            for is_night in [False, True]:
                key = (day, is_night)
                if key in self.history:
                    periods.append((day, is_night))
        # Sort periods chronologically
        periods.sort()
        # Build the full history
        output = []
        for day, is_night in periods:
            period_key = (day, is_night)
            history = self.history.get(period_key)
            if not history:
                continue
            # Filter messages the player can see based on channel permissions
            visible_messages = []
            for msg in history.messages:
                if self._player_could_see_channel(player, msg.channel_type, day, is_night):
                    visible_messages.append(msg)
            for whisper in history.whispers:
                if whisper.sender.id == player.id or any(
                    member_id == player.id for member_id in self._get_whisper_participants(whisper)
                ):
                    visible_messages.append(whisper)
            visible_messages.sort(key=id)
            # Add separator
            sep = f"\n--- {history.get_period_name()} ---\n"
            output.append(sep)
            if not visible_messages:
                output.append(f"No visible messages from {history.get_period_name()}.")
            else:
                for msg in visible_messages:
                    if msg.is_environment:
                        output.append(f"[ENV] {msg.message}")
                    elif msg.channel_type == ChatChannelType.WHISPER:
                        other_participant = "someone" if msg.sender.id == player.id else msg.sender.name
                        output.append(f"[WHISPER] Whispered with {other_participant}")
                    else:
                        output.append(f"{msg.sender.name}: {msg.message}")
            # Optionally, add night actions if this is a night period and actions are tracked elsewhere
            # (Assume night actions are logged as environment messages or similar)
        return "\n".join(output).strip()

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
        elif channel_type == ChatChannelType.MEDIUM_SEANCE:
            return False  # Seance messages are handled separately like whispers
        elif channel_type == ChatChannelType.PLAYER_PRIVATE_NOTIFICATION:
            return False  # Private notifications are handled separately in get_visible_messages
        
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
        """Send a public message from a player."""
        # Check if player can write to DAY_PUBLIC
        if ChannelOpenState.WRITE not in self.channels[ChatChannelType.DAY_PUBLIC].members.get(player.id, set()):
            return "You cannot speak right now."
        
        # Create and broadcast the message
        message = ChatMessage(player, text, ChatChannelType.DAY_PUBLIC)
        self.channels[ChatChannelType.DAY_PUBLIC].broadcast(player, text)
        
        # Log the chat message if logger is available
        if self.logger:
            turn_name = f"Night {self.current_day}" if self.current_is_night else f"Day {self.current_day}"
            self.logger.log_chat(player.name, text, turn_name, is_whisper=False)
        
        return message

    def send_whisper(self, src: 'Player', dst: 'Player', text: str, *, day: int, is_night: bool) -> ChatMessage | str:
        """Send a whisper from src to dst."""
        # Check if both players are alive
        if not src.is_alive or not dst.is_alive:
            return "You cannot whisper to dead players."
        
        # Check if src is blackmailed (cannot whisper)
        if hasattr(src, 'is_blackmailed') and src.is_blackmailed:
            return "You are blackmailed and cannot whisper."
        
        # Create whisper channel key
        whisper_key = (src.id, dst.id, day)
        
        # Create whisper channel if it doesn't exist
        if whisper_key not in self.whispers:
            whisper_channel = ChatChannel(ChatChannelType.WHISPER)
            whisper_channel.add_member(src, can_write=True, can_read=True)
            whisper_channel.add_member(dst, can_write=True, can_read=True)
            self.whispers[whisper_key] = whisper_channel
        
        # Send the whisper
        whisper_channel = self.whispers[whisper_key]
        message = ChatMessage(src, text, ChatChannelType.WHISPER)
        whisper_channel.broadcast(src, text)
        
        # Log the whisper if logger is available
        if self.logger:
            turn_name = f"Night {day}" if is_night else f"Day {day}"
            self.logger.log_chat(src.name, text, turn_name, is_whisper=True)
        
        return message

    def create_seance_channel(self, medium: 'Player', target: 'Player'):
        """Create a seance channel between Medium and target."""
        key = (medium.id, target.id, self.current_day)
        
        if key in self.seances:
            return  # Seance already exists
        
        # Create new seance channel
        channel = ChatChannel(ChatChannelType.MEDIUM_SEANCE)
        
        # Add both players to the seance
        channel.add_member(medium, can_write=True, can_read=True)
        channel.add_member(target, can_write=True, can_read=True)
        
        self.seances[key] = channel
        
        # Add environment message to announce seance
        channel.broadcast(medium, f"[SEANCE] {medium.name} has initiated a seance with {target.name}.", is_environment=True)
        
        print(f"[Chat] Seance channel created between {medium.name} (Medium) and {target.name}")

    def send_seance(self, sender: 'Player', text: str) -> ChatMessage | str:
        """Send a message in the seance channel this player is part of."""
        # Find seance channel this player is in
        for (medium_id, target_id, day), channel in self.seances.items():
            if self.current_day == day and (sender.id == medium_id or sender.id == target_id):
                channel.broadcast(sender, text)
                return channel.messages[-1]
        
        return "You are not in any active seance."

    # ------------------------------------------------------------------
    # Player-specific notification API
    # ------------------------------------------------------------------
    def add_player_notification(self, player: 'Player', message: str, *, is_environment: bool = False):
        """Add a private notification message visible only to a specific player."""
        if player.id not in self.player_notifications_channels:
            self.player_notifications_channels[player.id] = ChatChannel(ChatChannelType.PLAYER_PRIVATE_NOTIFICATION)
        
        channel = self.player_notifications_channels[player.id]
        # Use a dummy sender for environment-like notifications, or the player themselves for self-generated ones
        sender_for_notification = type('NotificationSender', (), {'name': 'SYSTEM', 'id': -2})() if is_environment else player
        channel.broadcast(sender_for_notification, message, is_environment=is_environment)

    # ------------------------------------------------------------------
    def get_visible_messages(self, player: 'Player') -> List[ChatMessage]:
        """Get all currently visible messages for a player (current period only)."""
        msgs: List[ChatMessage] = []
        for chan in self.channels.values():
            msgs.extend(chan.get_visible(player))
        for chan in self.whispers.values():
            msgs.extend(chan.get_visible(player))
        for chan in self.seances.values():
            msgs.extend(chan.get_visible(player))
        
        # Add player-specific notifications
        if player.id in self.player_notifications_channels:
            msgs.extend(self.player_notifications_channels[player.id].messages)

        # order by timestamp
        return sorted(msgs, key=lambda msg: msg.timestamp)

    def get_current_player_notifications(self, player: 'Player') -> List[ChatMessage]:
        """Get all private notifications for a player for the current period."""
        if player.id in self.player_notifications_channels:
            return sorted(self.player_notifications_channels[player.id].messages, key=lambda msg: msg.timestamp)
        return [] 