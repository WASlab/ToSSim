import random
from .enums import RoleName, Faction, RoleAlignment
from .alignment import ROLE_ALIGNMENT_MAP, get_role_faction, get_role_alignment

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
        RoleName.RETRIBUTIONIST,
    ],
    RoleAlignment.MAFIA_KILLING: [
        RoleName.GODFATHER,
        RoleName.MAFIOSO,
        RoleName.AMBUSHER,
    ],
    RoleAlignment.MAFIA_SUPPORT: [
        RoleName.BLACKMAILER,
        RoleName.CONSIGLIERE,
        RoleName.CONSORT,
    ],
    RoleAlignment.MAFIA_DECEPTION: [
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

CLASSIC_INVESTIGATOR_RESULTS = (
    (RoleName.VIGILANTE, RoleName.VETERAN, RoleName.MAFIOSO, RoleName.AMBUSHER),
    (RoleName.MEDIUM, RoleName.JANITOR, RoleName.RETRIBUTIONIST),
    (RoleName.SURVIVOR, RoleName.VAMPIRE_HUNTER, RoleName.AMNESIAC),
    (RoleName.SPY, RoleName.BLACKMAILER, RoleName.JAILOR),
    (RoleName.SHERIFF, RoleName.EXECUTIONER, RoleName.WEREWOLF),
    (RoleName.FRAMER, RoleName.VAMPIRE, RoleName.JESTER),
    (RoleName.LOOKOUT, RoleName.FORGER, RoleName.WITCH),
    (RoleName.ESCORT, RoleName.TRANSPORTER, RoleName.HYPNOTIST, RoleName.CONSORT), #Tavern Keeper maps to Escort
    (RoleName.DOCTOR, RoleName.DISGUISER, RoleName.SERIAL_KILLER),
    (RoleName.INVESTIGATOR, RoleName.CONSIGLIERE, RoleName.MAYOR),
    (RoleName.BODYGUARD, RoleName.GODFATHER, RoleName.ARSONIST),
)

COVEN_INVESTIGATOR_RESULTS = (
    (RoleName.VIGILANTE, RoleName.VETERAN, RoleName.MAFIOSO, RoleName.PIRATE, RoleName.AMBUSHER),
    (RoleName.MEDIUM, RoleName.JANITOR, RoleName.RETRIBUTIONIST, RoleName.NECROMANCER, RoleName.TRAPPER),
    (RoleName.SURVIVOR, RoleName.VAMPIRE_HUNTER, RoleName.AMNESIAC, RoleName.MEDUSA, RoleName.PSYCHIC),
    (RoleName.SPY, RoleName.BLACKMAILER, RoleName.JAILOR, RoleName.GUARDIAN_ANGEL),
    (RoleName.SHERIFF, RoleName.EXECUTIONER, RoleName.WEREWOLF, RoleName.POISONER),
    (RoleName.FRAMER, RoleName.VAMPIRE, RoleName.JESTER, RoleName.HEX_MASTER),
    (RoleName.LOOKOUT, RoleName.FORGER, RoleName.JUGGERNAUT, RoleName.COVEN_LEADER),
    (RoleName.ESCORT, RoleName.TRANSPORTER, RoleName.HYPNOTIST, RoleName.CONSORT), #Tavern Keeper maps to Escort
    (RoleName.DOCTOR, RoleName.DISGUISER, RoleName.SERIAL_KILLER, RoleName.POTION_MASTER),
    (RoleName.INVESTIGATOR, RoleName.CONSIGLIERE, RoleName.MAYOR, RoleName.TRACKER, RoleName.PLAGUEBEARER, RoleName.PESTILENCE),
    (RoleName.BODYGUARD, RoleName.GODFATHER, RoleName.ARSONIST, RoleName.CRUSADER),
)

CLASSIC_CONSIGLIERE_RESULTS = {
    RoleName.BODYGUARD: "Your target is a trained protector. They must be a Bodyguard.",
    RoleName.DOCTOR: "Your target is a professional surgeon. They must be a Doctor.",
    RoleName.ESCORT: "Your target is a beautiful person working for the town. They must be an Tavern Keeper.",
    RoleName.INVESTIGATOR: "Your target gathers information about people. They must be an Investigator.",
    RoleName.JAILOR: "Your target detains people at night. They must be a Jailor.",
    RoleName.LOOKOUT: "Your target watches who visits people at night. They must be a Lookout.",
    RoleName.MAYOR: "Your target is the leader of the town. They must be the Mayor.",
    RoleName.MEDIUM: "Your target speaks with the dead. They must be a Medium.",
    RoleName.RETRIBUTIONIST: "Your target wields mystical powers. They must be a Retributionist.",
    RoleName.SHERIFF: "Your target is a protector of the town. They must be a Sheriff.",
    RoleName.SPY: "Your target secretly watches who someone visits. They must be a Spy.",
    RoleName.TRANSPORTER: "Your target specializes in transportation. They must be a Transporter.",
    RoleName.VAMPIRE_HUNTER: "Your target tracks Vampires. They must be a Vampire Hunter!",
    RoleName.VETERAN: "Your target is a paranoid war hero. They must be a Veteran.",
    RoleName.VIGILANTE: "Your target will bend the law to enact justice. They must be a Vigilante.",
    RoleName.AMBUSHER: "Your target lies in wait. They must be an Ambusher.",
    RoleName.BLACKMAILER: "Your target uses information to silence people. They must be a Blackmailer.",
    RoleName.CONSIGLIERE: "Your target gathers information for the Mafia. They must be a Consigliere.",
    RoleName.CONSORT: "Your target is a beautiful person working for the Mafia. They must be a Bootlegger.",
    RoleName.DISGUISER: "Your target makes other people appear to be someone they're not. They must be a Disguiser.",
    RoleName.FORGER: "Your target is good at forging documents. They must be a Forger.",
    RoleName.FRAMER: "Your target has a desire to deceive. They must be a Framer!",
    RoleName.GODFATHER: "Your target is the leader of the Mafia. They must be the Godfather.",
    RoleName.HYPNOTIST: "Your target is skilled at disrupting others. They must be a Hypnotist.",
    RoleName.JANITOR: "Your target cleans up dead bodies. They must be a Janitor.",
    RoleName.MAFIOSO: "Your target does the Godfather's dirty work. They must be a Mafioso.",
    RoleName.AMNESIAC: "Your target does not remember their role. They must be an Amnesiac.",
    RoleName.ARSONIST: "Your target likes to watch things burn. They must be an Arsonist.",
    RoleName.EXECUTIONER: "Your target wants someone to be lynched at any cost. They must be an Executioner.",
    RoleName.JESTER: "Your target wants to be lynched. They must be a Jester.",
    RoleName.SERIAL_KILLER: "Your target wants to kill everyone. They must be a Serial Killer.",
    RoleName.SURVIVOR: "Your target simply wants to live. They must be a Survivor.",
    RoleName.VAMPIRE: "Your target drinks blood. They must be a Vampire!",
    RoleName.WEREWOLF: "Your target howls at the moon. They must be a Werewolf.",
    RoleName.WITCH: "Your target casts spells on people. They must be a Witch.",
}

COVEN_CONSIGLIERE_RESULTS = {
    **CLASSIC_CONSIGLIERE_RESULTS,
    RoleName.CRUSADER: "Your target is a divine protector. They must be a Crusader.",
    RoleName.PSYCHIC: "Your target has the sight. They must be a Psychic.",
    RoleName.TRACKER: "Your target is a skilled in the art of tracking. They must be a Tracker.",
    RoleName.TRAPPER: "Your target is waiting for a big catch. They must be a Trapper.",
    RoleName.GUARDIAN_ANGEL: "Your target is watching over someone. They must be a Guardian Angel.",
    RoleName.JUGGERNAUT: "Your target gets more powerful with each kill. They must be a Juggernaut.",
    RoleName.PESTILENCE: "Your target reeks of disease. They must be Pestilence, Horseman of the Apocalypse.",
    RoleName.PIRATE: "Your target wants to plunder the town. They must be a Pirate.",
    RoleName.PLAGUEBEARER: "Your target is a carrier of disease. They must be the Plaguebearer.",
    RoleName.COVEN_LEADER: "Your target leads the mystical. They must be the Coven Leader.",
    RoleName.HEX_MASTER: "Your target is versed in the ways of hexes. They must be the Hex Master.",
    RoleName.MEDUSA: "Your target has a gaze of stone. They must be Medusa.",
    RoleName.NECROMANCER: "Your target uses the deceased to do their dirty work. They must be the Necromancer.",
    RoleName.POISONER: "Your target uses herbs and plants to kill their victims. They must be the Poisoner.",
    RoleName.POTION_MASTER: "Your target works with alchemy. They must be a Potion Master.",
}

#Base configuration for investigation results
#This structure maps a role to a list of other roles it could appear as.
INVESTIGATOR_RESULTS = {
    #Coven Expansion Results
    "coven": {
        RoleName.VIGILANTE: [RoleName.VIGILANTE, RoleName.VETERAN, RoleName.MAFIOSO, RoleName.PIRATE, RoleName.AMBUSHER],
        RoleName.VETERAN: [RoleName.VIGILANTE, RoleName.VETERAN, RoleName.MAFIOSO, RoleName.PIRATE, RoleName.AMBUSHER],
        RoleName.MAFIOSO: [RoleName.VIGILANTE, RoleName.VETERAN, RoleName.MAFIOSO, RoleName.PIRATE, RoleName.AMBUSHER],
        RoleName.PIRATE: [RoleName.VIGILANTE, RoleName.VETERAN, RoleName.MAFIOSO, RoleName.PIRATE, RoleName.AMBUSHER],
        RoleName.AMBUSHER: [RoleName.VIGILANTE, RoleName.VETERAN, RoleName.MAFIOSO, RoleName.PIRATE, RoleName.AMBUSHER],
        
        RoleName.LOOKOUT: [RoleName.LOOKOUT, RoleName.FORGER, RoleName.COVEN_LEADER],
        RoleName.FORGER: [RoleName.LOOKOUT, RoleName.FORGER, RoleName.COVEN_LEADER],
        RoleName.COVEN_LEADER: [RoleName.LOOKOUT, RoleName.FORGER, RoleName.COVEN_LEADER],
    },
    #Classic (Non-Coven) Results
    "classic": {
        RoleName.VIGILANTE: [RoleName.VIGILANTE, RoleName.VETERAN, RoleName.MAFIOSO],
        RoleName.VETERAN: [RoleName.VIGILANTE, RoleName.VETERAN, RoleName.MAFIOSO],
        RoleName.MAFIOSO: [RoleName.VIGILANTE, RoleName.VETERAN, RoleName.MAFIOSO],
    }
}

#Consigliere results are direct role reveals
CONSILIERE_RESULTS = {role: f"Your target is a {role.value}." for role in RoleName}

#--- Game Mode Role Lists ---

def get_classic_roles():
    return [
        RoleName.JAILOR, "TOWN_INVESTIGATIVE", "TOWN_INVESTIGATIVE", "TOWN_SUPPORT", 
        "TOWN_SUPPORT", "TOWN_PROTECTIVE", "TOWN_KILLING", "RANDOM_TOWN",
        RoleName.GODFATHER, RoleName.MAFIOSO, "MAFIA_SUPPORT", "RANDOM_MAFIA",
        "NEUTRAL_KILLING", "NEUTRAL_EVIL", "ANY"
    ]

#--- Helper Functions ---

def get_role_list(game_mode: str, coven: bool) -> list[RoleName]:
    """Returns the role list for a given game mode."""
    if game_mode.lower() == "classic":
        return get_classic_roles()
    #Add other game modes here
    return get_classic_roles() #Default to classic

def get_investigator_result_group(role: RoleName, coven: bool) -> list[RoleName] | None:
    """Gets the investigator result group for a given role.

    First consult the fast look-up dictionary (INVESTIGATOR_RESULTS). If the role
    is not present there (because the dictionary only stores the most common
    groups) fall back to scanning the full classic / coven result tuples so that
    we never return None for any canonical role.
    """
    #Fast path via mapping – covers most roles.
    mapping = INVESTIGATOR_RESULTS["coven"] if coven else INVESTIGATOR_RESULTS["classic"]
    for key_role, group in mapping.items():
        if role in group:
            return group

    #Fall-back: iterate through the exhaustive group tuples defined above.
    groups = COVEN_INVESTIGATOR_RESULTS if coven else CLASSIC_INVESTIGATOR_RESULTS
    for grp in groups:
        if role in grp:
            return list(grp)

    return None

def get_consigliere_result(role: RoleName) -> str:
    """Gets the consigliere's result message for a given role."""
    return CONSILIERE_RESULTS.get(role, "Your target's role is unknown.")

class GameConfiguration:
    def __init__(self, game_mode="Classic", coven=False):
        self.game_mode = game_mode
        self.is_coven = coven
        self.role_list = self._resolve_role_list(get_role_list(game_mode, coven))

    def _resolve_role_list(self, raw_list: list) -> list[RoleName]:
        """Resolves random/any roles into specific roles."""
        resolved_list = []
        for item in raw_list:
            if isinstance(item, RoleName):
                resolved_list.append(item)
            elif isinstance(item, str):
                #Handle string-based random roles
                if item == "RANDOM_TOWN":
                    resolved_list.append(self._get_random_role_by_faction(Faction.TOWN))
                elif item == "RANDOM_MAFIA":
                    resolved_list.append(self._get_random_role_by_faction(Faction.MAFIA))
                elif item == "RANDOM_COVEN":
                     resolved_list.append(self._get_random_role_by_faction(Faction.COVEN))
                elif item == "ANY":
                    resolved_list.append(random.choice(list(ROLE_ALIGNMENT_MAP.keys())))
                else:
                    #Handle specific alignment strings like "TOWN_INVESTIGATIVE"
                    try:
                        alignment_enum = RoleAlignment[item]
                        resolved_list.append(self._get_random_role_by_alignment(alignment_enum))
                    except KeyError:
                        #Handle cases where the string is not a valid RoleAlignment
                        print(f"Warning: Could not resolve role string '{item}'")
        return resolved_list

    def _get_random_role_by_alignment(self, alignment: RoleAlignment) -> RoleName:
        """Gets a random role from a specified alignment."""
        alignment_roles = [role for role, align in ROLE_ALIGNMENT_MAP.items() if align == alignment]
        return random.choice(alignment_roles) if alignment_roles else None

    def _get_random_role_by_faction(self, faction: Faction) -> RoleName:
        """Gets a random role from a specified faction."""
        faction_roles = [role for role in ROLE_ALIGNMENT_MAP.keys() if get_role_faction(role) == faction]
        return random.choice(faction_roles) if faction_roles else None

    def get_investigator_result_group(self, role: RoleName) -> list[RoleName] | None:
        return get_investigator_result_group(role, self.is_coven)

    def get_consigliere_result(self, role: RoleName) -> str:
        return get_consigliere_result(role)

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
    