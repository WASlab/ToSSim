from enum import Enum, auto

class Faction(Enum):
    TOWN = auto()
    MAFIA = auto()
    COVEN = auto()
    NEUTRAL = auto()
    VAMPIRE = auto()
    # PESTILENCE is a transformed role, might not be a faction itself, but let's consider it.
    PESTILENCE = auto() 

class RoleAlignment(Enum):
    # Town Sub-alignments
    TOWN_INVESTIGATIVE = auto()
    TOWN_PROTECTIVE = auto()
    TOWN_KILLING = auto()
    TOWN_SUPPORT = auto()
    
    # Mafia Sub-alignments
    MAFIA_KILLING = auto()
    MAFIA_SUPPORT = auto()
    MAFIA_DECEPTION = auto()

    # Coven is all unique, no real sub-alignments mentioned other than Coven Leader.
    COVEN = auto()

    # Neutral Sub-alignments
    NEUTRAL_KILLING = auto()
    NEUTRAL_EVIL = auto()
    NEUTRAL_CHAOS = auto()
    NEUTRAL_BENIGN = auto()

class RoleName(Enum):
    # Town Investigative
    INVESTIGATOR = "Investigator"
    LOOKOUT = "Lookout"
    SHERIFF = "Sheriff"
    SPY = "Spy"
    PSYCHIC = "Psychic"
    TRACKER = "Tracker"

    # Town Protective
    DOCTOR = "Doctor"
    BODYGUARD = "Bodyguard"
    CRUSADER = "Crusader"
    TRAPPER = "Trapper"

    # Town Killing
    JAILOR = "Jailor"
    VIGILANTE = "Vigilante"
    VETERAN = "Veteran"
    VAMPIRE_HUNTER = "Vampire Hunter"

    # Town Support
    MAYOR = "Mayor"
    MEDIUM = "Medium"
    ESCORT = "Escort"
    TRANSPORTER = "Transporter"
    RETRIBUTIONIST = "Retributionist"

    # Mafia Killing
    GODFATHER = "Godfather"
    MAFIOSO = "Mafioso"
    AMBUSHER = "Ambusher"

    # Mafia Support
    CONSIGLIERE = "Consigliere"
    BLACKMAILER = "Blackmailer"
    HYPNOTIST = "Hypnotist"
    CONSORT = "Consort"

    # Mafia Deception
    FORGER = "Forger"
    JANITOR = "Janitor"
    DISGUISER = "Disguiser"

    # Coven
    COVEN_LEADER = "Coven Leader"
    POTION_MASTER = "Potion Master"
    POISONER = "Poisoner"
    MEDUSA = "Medusa"
    NECROMANCER = "Necromancer"
    HEX_MASTER = "Hex Master"

    # Neutral Killing
    SERIAL_KILLER = "Serial Killer"
    ARSONIST = "Arsonist"
    WEREWOLF = "Werewolf"
    JUGGERNAUT = "Juggernaut"
    PESTILENCE = "Pestilence"

    # Neutral Evil
    WITCH = "Witch"
    EXECUTIONER = "Executioner"
    JESTER = "Jester"

    # Neutral Chaos
    VAMPIRE = "Vampire"
    PIRATE = "Pirate"
    PLAGUEBEARER = "Plaguebearer"
    GUARDIAN_ANGEL = "Guardian Angel"

    # Neutral Benign
    SURVIVOR = "Survivor"
    AMNESIAC = "Amnesiac"

class Attack(Enum):
    NONE = 0
    BASIC = 1
    POWERFUL = 2
    UNSTOPPABLE = 3

class Defense(Enum):
    NONE = 0
    BASIC = 1
    POWERFUL = 2
    INVINCIBLE = 3

class Priority(Enum):
    # Based on the document's night order
    JAILOR_JAIL = 0 # Day action, but affects night start
    VETERAN_ALERT = 1
    TRANSPORTER_SWAP = 1
    WITCH_CONTROL = 2
    COVEN_LEADER_CONTROL = 2
    ROLEBLOCK = 3
    SELF_PROTECT = 4
    PROTECT = 4
    MISC_NON_LETHAL = 5
    KILLING = 6
    POST_ATTACK = 7
    REMEMBER_ROLE = 8
    OBSERVATION = 9
    FINALIZATION = 10 