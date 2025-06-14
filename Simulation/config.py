import random
from .enums import RoleName, RoleAlignment, Faction

#This is a master list that defines which roles fall into whcih categories
#All role lists will be generated from this list, consider it to be a source of truth

ROLE_CATEGORIES = {
    RoleAlignment.TOWN_INVESTIGATIVE: [
        RoleName.INVESTIGATOR,
        RoleName.LOOKOUT,
        RoleName.SHERIFF,
        RoleName.SPY,
        RoleName.PSYCHIC,
        RoleName.TRACKER,
    ],
    RoleAlignment.TOWN_PROTECTIVE: [
        RoleName.DOCTOR,
        RoleName.BODYGUARD,
        RoleName.CRUSADER,
        RoleName.TRAPPER,
    ],
    RoleAlignment.TOWN_KILLING: [
        RoleName.JAILOR,
        RoleName.VIGILANTE,
        RoleName.VETERAN,
        RoleName.VAMPIRE_HUNTER,
    ],
    RoleAlignment.TOWN_SUPPORT: [
        RoleName.MAYOR,
        RoleName.MEDIUM,
        RoleName.ESCORT,
        RoleName.TRANSPORTER,
        RoleName.RETTRIBUTIONIST,
    ],
    RoleAlignment.MAFIA_KILLING: [
        RoleName.GODFATHER,
        RoleName.MAFIOSO,
        RoleName.AMBUSHER,
    ],
    RoleAlignment.MAFIA_SUPPORT: [
        RoleName.BLACKMAILER,
        RoleName.CONSIGLIERE,
        RoleName.BOOTLEGGER,
    ],
    RoleAlignment.MAFIA_DECEPTION: [
        RoleName.CONSORT,
        RoleName.FORGER,
        RoleName.JANITOR,
        RoleName.FRAMER,
        RoleName.DISGUISER,
        RoleName.HYPNOTIST,
        
    ],
    RoleAlignment.NEUTRAL_BENIGN: [
        RoleName.AMNESIAC,
        RoleName.GUARDIAN_ANGEL,
        RoleName.SURVIVOR,
    ],
    RoleAlignment.NEUTRAL_EVIL: [
        RoleName.EXECUTIONER,
        RoleName.JESTER,
        RoleName.WITCH
    ],
    RoleAlignment.NEUTRAL_KILLING: [
        RoleName.ARSONIST,
        RoleName.JUGGERNAUT,
        RoleName.SERIAL_KILLER,
        RoleName.WEREWOLF
    ],
    RoleAlignment.NEUTRAL_CHAOS: [
        RoleName.PIRATE,
        RoleName.VAMPIRE,
        RoleName.PLAGUEBEARER,
        RoleName.PESTILENCE
    ],
    RoleAlignment.COVEN_EVIL: [
        RoleName.COVEN_LEADER,
        RoleName.HEX_MASTER,
        RoleName.MEDUSA,
        RoleName.NECROMANCER,
        RoleName.POISONER,
        RoleName.POTION_MASTER,
    ],
    
}
UNIQUE_ROLES = [
    RoleName.GODFATHER,
    RoleName.JAILOR,
    RoleName.VETERAN,
    RoleName.WEREWOLF,
    RoleName.GODFATHER,
    RoleName.MAYOR,
    RoleName.RETRIBUTIONIST,
    RoleName.MAFIOSO,
    RoleName.COVEN_LEADER,
    RoleName.POISONER,
    RoleName.POTION_MASTER,
    RoleName.MEDUSA,
    RoleName.NECROMANCER,
    RoleName.HEX_MASTER,
    
]
class GameConfiguration:
    def __init__(self, config: dict):
        if not isinstance(config, dict):
            raise TypeError("Configuration must be a dictionary.")
        self.config = config
        self.num_players = sum(self.config.values())

    def generate_role_list(self) -> list[RoleName]:
        role_list = []
        used_unique_roles = set()

        def add_role(role):
            if role in UNIQUE_ROLES:
                if role in used_unique_roles:
                    return False
                used_unique_roles.add(role)
            role_list.append(role)
            return True

        for role, count in self.config.items():
            if isinstance(role, RoleName):
                for _ in range(count):
                    if not add_role(role):
                        raise ValueError(f"Configuration error: Tried to add more than one unique role: {role.value}")

        special_categories = {
            "RANDOM_TOWN": [role for cat in (RoleAlignment.TOWN_INVESTIGATIVE, RoleAlignment.TOWN_PROTECTIVE, RoleAlignment.TOWN_KILLING, RoleAlignment.TOWN_SUPPORT) for role in ROLE_CATEGORIES.get(cat, [])],
            "RANDOM_MAFIA": [role for cat in (RoleAlignment.MAFIA_KILLING, RoleAlignment.MAFIA_SUPPORT, RoleAlignment.MAFIA_DECEPTION) for role in ROLE_CATEGORIES.get(cat, [])],
            "RANDOM_COVEN": ROLE_CATEGORIES.get(RoleAlignment.COVEN_EVIL, []),
            "ANY": [role for sublist in ROLE_CATEGORIES.values() for role in sublist]
        }
        
        for category_key, count in self.config.items():
            if isinstance(category_key, RoleName):
                continue

            pool = ROLE_CATEGORIES.get(category_key) or special_categories.get(category_key)
            if not pool:
                print(f"Warning: Unrecognized role category '{category_key}' in config. Skipping.")
                continue
            
            for _ in range(count):
                available_roles = [r for r in pool if r not in used_unique_roles]
                if not available_roles:
                    raise ValueError(f"Not enough unique roles available for category '{category_key}'")
                
                chosen_role = random.choice(available_roles)
                add_role(chosen_role)

        if len(role_list) != self.num_players:
            raise ValueError(f"Generated {len(role_list)} roles, but expected {self.num_players}.")

        random.shuffle(role_list)
        return role_list
    
#########GAME MODE PRESETS ################

CLASSIC_15_PLAYER_CONFIG = {
    RoleName.SHERIFF: 1,
    RoleName.LOOKOUT: 1,
    RoleName.INVESTIGATOR: 1,
    RoleName.JAILOR: 1,
    RoleName.DOCTOR: 1,
    #Tavern Keeper is not a role in this game, so we will use Escort instead
    RoleName.ESCORT: 1,
    RoleName.MEDIUM: 1,
    RoleAlignment.TOWN_KILLING: 1,
    "RANDOM_TOWN": 1,
    RoleName.GODFATHER: 1,
    RoleName.MAFIOSO: 1,
    RoleName.FRAMER: 1,
    RoleName.SERIAL_KILLER: 1,
    RoleName.EXECUTIONER: 1,
    RoleName.JESTER: 1,
}
RANKED_PRACTICE_15_PLAYER_CONFIG = {
    RoleName.JAILOR: 1,
    RoleAlignment.TOWN_INVESTIGATIVE: 2,
    RoleAlignment.TOWN_PROTECTIVE: 1,
    "RANDOM_TOWN": 5,
    RoleName.GODFATHER: 1,
    RoleName.MAFIOSO: 1,
    RoleAlignment.MAFIA_SUPPORT: 1,
    "RANDOM_MAFIA": 1,
    RoleAlignment.NEUTRAL_EVIL: 1,
    RoleAlignment.NEUTRAL_KILLING: 1,
}
RANKED_15_PLAYER_CONFIG = RANKED_PRACTICE_15_PLAYER_CONFIG

ALL_ANY_15_PLAYER_CONFIG = {
    "ANY": 15
}
RAINBOW_15_PLAYER_CONFIG = {
    RoleName.GODFATHER: 1,
    RoleName.ARSONIST: 2,
    RoleName.SURVIVOR: 2,
    RoleName.JAILOR: 1,
    RoleName.AMNESIAC: 2,
    RoleName.SERIAL_KILLER: 2,
    RoleName.WITCH: 2,
    "ANY": 1,
    RoleName.VETERAN: 1,
    RoleName.MAFIOSO: 1
}
DRACULAS_PALACE_15_PLAYER_CONFIG = {
    RoleName.DOCTOR: 1,
    RoleName.LOOKOUT: 2,
    RoleName.JAILOR: 1,
    RoleName.VIGILANTE: 1,
    RoleAlignment.TOWN_PROTECTIVE: 1,
    RoleAlignment.TOWN_SUPPORT: 2,
    RoleName.VAMPIRE_HUNTER: 1,
    RoleName.JESTER: 1,
    RoleName.WITCH: 1,
    RoleName.VAMPIRE: 4
}
TOWN_TRAITOR_15_PLAYER_CONFIG = {
    RoleName.SHERIFF: 1,
    RoleName.JAILOR: 1,
    RoleName.DOCTOR: 1,
    RoleName.LOOKOUT: 1,
    RoleAlignment.TOWN_INVESTIGATIVE: 1,
    RoleAlignment.TOWN_PROTECTIVE: 1,
    RoleAlignment.TOWN_KILLING: 1,
    RoleAlignment.TOWN_SUPPORT: 1,
    "RANDOM_TOWN": 3,
    RoleName.GODFATHER: 1,
    RoleName.MAFIOSO: 1,
    "RANDOM_MAFIA": 1,
    RoleName.WITCH: 1,
}

COVEN_CLASSIC_15_PLAYER_CONFIG = {
    RoleName.SHERIFF: 1,
    RoleName.LOOKOUT: 1,
    RoleName.PSYCHIC: 1,
    RoleName.JAILOR: 1,
    RoleAlignment.TOWN_PROTECTIVE: 1,
    "RANDOM_TOWN": 3,
    RoleName.COVEN_LEADER: 1,
    RoleName.MEDUSA: 1,
    RoleName.POTION_MASTER: 1,
    "RANDOM_COVEN": 1,
    RoleName.EXECUTIONER: 1,
    RoleName.PIRATE: 1,
    RoleName.PLAGUEBEARER: 1,
}

COVEN_RANKED_PRACTICE_15_PLAYER_CONFIG = {
    RoleName.JAILOR: 1,
    RoleAlignment.TOWN_INVESTIGATIVE: 2,
    RoleAlignment.TOWN_SUPPORT: 1,
    RoleAlignment.TOWN_PROTECTIVE: 1,
    RoleAlignment.TOWN_KILLING: 1,
    "RANDOM_TOWN": 3,
    RoleName.COVEN_LEADER: 1,
    RoleName.MEDUSA: 1,
    "RANDOM_COVEN": 2,
    RoleAlignment.NEUTRAL_KILLING: 1,
    RoleAlignment.NEUTRAL_EVIL: 1,
}

COVEN_RANKED_15_PLAYER_CONFIG = COVEN_RANKED_PRACTICE_15_PLAYER_CONFIG

MAFIA_RETURNS_15_PLAYER_CONFIG = {
    RoleName.SHERIFF: 1,
    RoleName.LOOKOUT: 1,
    RoleName.PSYCHIC: 1,
    RoleName.JAILOR: 1,
    RoleAlignment.TOWN_PROTECTIVE: 1,
    "RANDOM_TOWN": 3,
    RoleName.GODFATHER: 1,
    RoleName.AMBUSHER: 1,
    RoleName.HYPNOTIST: 1,
    "RANDOM_MAFIA": 1,
    RoleName.EXECUTIONER: 1,
    RoleName.PIRATE: 1,
    RoleName.PLAGUEBEARER: 1,
}

COVEN_ALL_ANY_15_PLAYER_CONFIG = {
    "ANY": 15
}
    