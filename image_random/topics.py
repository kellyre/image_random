"""Seed pools used to force diversity across prompt batches.

Instead of a fixed topic list (which repeats across runs), each Ollama call
gets a random sample of broad DOMAINS and must invent a fresh, specific scene
concept in each one, steering clear of every concept already generated.
Styles are all photographic/physically-based: subjects may be fantastical,
but surfaces and materials should read as real.
"""

import random

DOMAINS = [
    "deep ocean or underwater worlds",
    "mythology or folklore brought to life",
    "megafauna, creatures, or impossible beasts",
    "ancient ruins or lost civilizations",
    "cosmic, astronomical, or planetary scenes",
    "dense cities, markets, or street life",
    "wilderness, weather, or natural forces",
    "flying machines, airships, or impossible vehicles",
    "enchanted forests, flora, or fungal worlds",
    "workshops, laboratories, or places of craft",
    "festivals, rituals, or gatherings",
    "architecture that defies physics",
    "the very small: macro, insects, mechanisms",
    "deserts, ice fields, or extreme landscapes",
    "islands, coasts, or floating landmasses",
    "underground realms: caves, mines, buried cities",
    "libraries, archives, or repositories of knowledge",
    "harbors, shipwrecks, or seafaring life",
    "mountains, monasteries, or high places",
    "dreamlike or surreal juxtapositions",
    "post-human or reclaimed-by-nature places",
    "kitchens, feasts, or food as landscape",
    "clockwork, automata, or living machines",
    "portals, thresholds, or between-worlds",
    "trains, stations, or great journeys",
    "storms of unusual things",
    "gardens: overgrown, celestial, or impossible",
    "nomads, caravans, or migrations",
    "light phenomena: bioluminescence, auroras, refraction",
    "colossal statues, relics, or forgotten monuments",
]

# All photographic / physically-based: fantastical subjects, real surfaces.
STYLES = [
    "ultra-detailed photorealistic photography, shallow depth of field",
    "cinematic film still, anamorphic lens, moody color grading",
    "national geographic expedition photography",
    "long-exposure night photography",
    "aerial drone photography, golden hour",
    "large-format landscape photography, razor sharp front to back",
    "documentary photography, natural light, decisive moment",
    "vintage kodachrome slide film photograph",
    "macro photography with visible surface texture and dust",
    "overcast soft-light photography, muted natural palette",
    "high-speed photography freezing motion mid-action",
    "twilight blue-hour photography with practical lights",
]


def sample_domains(n: int, rng: random.Random | None = None) -> list[str]:
    rng = rng or random
    return rng.sample(DOMAINS, min(n, len(DOMAINS)))


def sample_style(rng: random.Random | None = None) -> str:
    rng = rng or random
    return rng.choice(STYLES)
