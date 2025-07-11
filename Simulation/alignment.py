from .enums import RoleName, RoleAlignment, Faction

#Canonical mapping of each role to its alignment
ROLE_ALIGNMENT_MAP = {
    #Town Investigative
    RoleName.INVESTIGATOR: RoleAlignment.TOWN_INVESTIGATIVE,
    RoleName.LOOKOUT: RoleAlignment.TOWN_INVESTIGATIVE,
    RoleName.PSYCHIC: RoleAlignment.TOWN_INVESTIGATIVE,
    RoleName.SHERIFF: RoleAlignment.TOWN_INVESTIGATIVE,
    RoleName.SPY: RoleAlignment.TOWN_INVESTIGATIVE,
    RoleName.TRACKER: RoleAlignment.TOWN_INVESTIGATIVE,
    #Town Protective
    RoleName.BODYGUARD: RoleAlignment.TOWN_PROTECTIVE,
    RoleName.CRUSADER: RoleAlignment.TOWN_PROTECTIVE,
    RoleName.DOCTOR: RoleAlignment.TOWN_PROTECTIVE,
    RoleName.TRAPPER: RoleAlignment.TOWN_PROTECTIVE,
    #Town Killing
    RoleName.JAILOR: RoleAlignment.TOWN_KILLING,
    RoleName.VAMPIRE_HUNTER: RoleAlignment.TOWN_KILLING,
    RoleName.VETERAN: RoleAlignment.TOWN_KILLING,
    RoleName.VIGILANTE: RoleAlignment.TOWN_KILLING,
    #Town Support
    RoleName.MAYOR: RoleAlignment.TOWN_SUPPORT,
    RoleName.MEDIUM: RoleAlignment.TOWN_SUPPORT,
    RoleName.RETRIBUTIONIST: RoleAlignment.TOWN_SUPPORT,
    RoleName.TAVERN_KEEPER: RoleAlignment.TOWN_SUPPORT,
    RoleName.TRANSPORTER: RoleAlignment.TOWN_SUPPORT,
    #Mafia Deception
    RoleName.DISGUISER: RoleAlignment.MAFIA_DECEPTION,
    RoleName.FORGER: RoleAlignment.MAFIA_DECEPTION,
    RoleName.FRAMER: RoleAlignment.MAFIA_DECEPTION,
    RoleName.HYPNOTIST: RoleAlignment.MAFIA_DECEPTION,
    RoleName.JANITOR: RoleAlignment.MAFIA_DECEPTION,
    #Mafia Killing
    RoleName.AMBUSHER: RoleAlignment.MAFIA_KILLING,
    RoleName.GODFATHER: RoleAlignment.MAFIA_KILLING,
    RoleName.MAFIOSO: RoleAlignment.MAFIA_KILLING,
    #Mafia Support
    RoleName.BLACKMAILER: RoleAlignment.MAFIA_SUPPORT,
    RoleName.BOOTLEGGER: RoleAlignment.MAFIA_SUPPORT,
    RoleName.CONSIGLIERE: RoleAlignment.MAFIA_SUPPORT,
    #Neutral Benign
    RoleName.AMNESIAC: RoleAlignment.NEUTRAL_BENIGN,
    RoleName.GUARDIAN_ANGEL: RoleAlignment.NEUTRAL_BENIGN,
    RoleName.SURVIVOR: RoleAlignment.NEUTRAL_BENIGN,
    #Neutral Evil
    RoleName.EXECUTIONER: RoleAlignment.NEUTRAL_EVIL,
    RoleName.JESTER: RoleAlignment.NEUTRAL_EVIL,
    RoleName.WITCH: RoleAlignment.NEUTRAL_EVIL,
    #Neutral Killing
    RoleName.ARSONIST: RoleAlignment.NEUTRAL_KILLING,
    RoleName.JUGGERNAUT: RoleAlignment.NEUTRAL_KILLING,
    RoleName.SERIAL_KILLER: RoleAlignment.NEUTRAL_KILLING,
    RoleName.WEREWOLF: RoleAlignment.NEUTRAL_KILLING,
    #Neutral Chaos
    RoleName.PIRATE: RoleAlignment.NEUTRAL_CHAOS,
    RoleName.PESTILENCE: RoleAlignment.NEUTRAL_CHAOS,
    RoleName.PLAGUEBEARER: RoleAlignment.NEUTRAL_CHAOS,
    RoleName.VAMPIRE: RoleAlignment.NEUTRAL_CHAOS,
    #Coven Evil
    RoleName.COVEN_LEADER: RoleAlignment.COVEN_EVIL,
    RoleName.MEDUSA: RoleAlignment.COVEN_EVIL,
    RoleName.HEX_MASTER: RoleAlignment.COVEN_EVIL,
    RoleName.POISONER: RoleAlignment.COVEN_EVIL,
    RoleName.POTION_MASTER: RoleAlignment.COVEN_EVIL,
    RoleName.NECROMANCER: RoleAlignment.COVEN_EVIL,
}

#Helper function to get the faction from an alignment
def get_faction_from_alignment(alignment: RoleAlignment) -> Faction:
    if alignment in [
        RoleAlignment.TOWN_INVESTIGATIVE, RoleAlignment.TOWN_PROTECTIVE,
        RoleAlignment.TOWN_KILLING, RoleAlignment.TOWN_SUPPORT
    ]:
        return Faction.TOWN
    elif alignment in [
        RoleAlignment.MAFIA_DECEPTION, RoleAlignment.MAFIA_KILLING, RoleAlignment.MAFIA_SUPPORT
    ]:
        return Faction.MAFIA
    elif alignment == RoleAlignment.COVEN_EVIL:
        return Faction.COVEN
    elif alignment == RoleAlignment.NEUTRAL_CHAOS and RoleName.VAMPIRE in ROLE_ALIGNMENT_MAP: #Special case for Vampire faction
         return Faction.VAMPIRE
    elif alignment == RoleAlignment.NEUTRAL_CHAOS and RoleName.PESTILENCE in ROLE_ALIGNMENT_MAP: #Special case for Pestilence
         return Faction.PESTILENCE
    else:
        return Faction.NEUTRAL

def get_role_alignment(role_name: RoleName) -> RoleAlignment:
    return ROLE_ALIGNMENT_MAP.get(role_name)

def get_role_faction(role_name: RoleName) -> Faction:
    alignment = get_role_alignment(role_name)
    if alignment:
        # Special cases for specific roles that have their own factions
        if role_name == RoleName.VAMPIRE:
            return Faction.VAMPIRE
        elif role_name == RoleName.PESTILENCE:
            return Faction.PESTILENCE
        return get_faction_from_alignment(alignment)
    return None 