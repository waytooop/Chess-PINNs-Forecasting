"""
Configuration file for grandmaster data.

Contains:
- FIDE IDs for each grandmaster
- Mapping between GM keys and their information
- Rating milestones for analysis
"""
from typing import Dict, List, Any

# List of grandmasters with their FIDE IDs
GRANDMASTERS = [
    {
        "key": "GM001",
        "name": "Magnus Carlsen",
        "id": "1503014",
        "country": "Norway",
        "born": 1990
    },
    {
        "key": "GM002",
        "name": "Hikaru Nakamura",
        "id": "2016192",
        "country": "USA",
        "born": 1987
    },
    {
        "key": "GM003", 
        "name": "Gukesh D",
        "id": "46616543",
        "country": "India",
        "born": 2006
    },
    {
        "key": "GM004",
        "name": "Fabiano Caruana",
        "id": "2020009",
        "country": "USA",
        "born": 1992
    },
    {
        "key": "GM005",
        "name": "Alireza Firouzja",
        "id": "12573981",
        "country": "France",
        "born": 2003
    },
    {
        "key": "GM006",
        "name": "Ding Liren",
        "id": "8603677",
        "country": "China",
        "born": 1992
    },
    {
        "key": "GM007",
        "name": "Wesley So",
        "id": "5202213",
        "country": "USA",
        "born": 1993
    }
]

# Dictionary for easy lookup by GM key
GM_DICT = {gm["key"]: gm for gm in GRANDMASTERS}

# Dictionary for lookup by FIDE ID
GM_BY_FIDE_ID = {gm["id"]: gm for gm in GRANDMASTERS}

# Rating milestones to analyze
RATING_MILESTONES = [2200, 2300, 2400, 2500, 2600, 2700, 2800, 2850, 2900]

# Reference points for analysis
REFERENCE_RATINGS = {
    "Club Player": 1600, 
    "Expert": 2000,
    "FIDE Master": 2300,
    "International Master": 2400,
    "Grandmaster": 2500,
    "Super Grandmaster": 2700,
    "World Elite": 2800
}

def get_gm_by_fide_id(fide_id: str) -> Dict[str, Any]:
    """Get GM info by FIDE ID."""
    return GM_BY_FIDE_ID.get(fide_id, None)

def get_gm_by_key(key: str) -> Dict[str, Any]:
    """Get GM info by key."""
    return GM_DICT.get(key, None)

def get_all_fide_ids() -> List[str]:
    """Get list of all GM FIDE IDs."""
    return [gm["id"] for gm in GRANDMASTERS]
