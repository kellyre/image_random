"""Topic and style seed pools used to force diversity across prompt batches.

Each Ollama call gets a random sample of topics + a style direction, so a run
of 50 prompts spans many unrelated subjects instead of drifting into one theme.
"""

import random

TOPICS = [
    "a bioluminescent deep-sea ecosystem",
    "a bustling night market in Southeast Asia",
    "an abandoned Soviet-era space facility",
    "a macro view of frost crystals forming",
    "a solarpunk city integrated with forests",
    "a medieval blacksmith's workshop at dawn",
    "an alien desert with impossible geology",
    "a 1920s jazz club interior",
    "a colossal ancient tree housing a village",
    "a storm chaser's view of a supercell",
    "an art nouveau greenhouse conservatory",
    "a cyberpunk street food vendor",
    "a whale migration seen from above",
    "a Moroccan riad courtyard in afternoon light",
    "an ice cave lit by aurora borealis",
    "a retro-futurist 1960s vision of Mars",
    "a hidden Japanese mountain shrine in fog",
    "a clockwork automaton orchestra",
    "a coral reef reclaiming a sunken airliner",
    "a Venetian carnival at midnight",
    "a prairie thunderstorm over wheat fields",
    "an Afrofuturist royal palace",
    "a lighthouse keeper's room during a gale",
    "a floating sky archipelago with waterfalls",
    "a paleontology dig site at golden hour",
    "an Antarctic research station under stars",
    "a baroque library with impossible architecture",
    "a street in Havana with vintage cars after rain",
    "a nomad caravan crossing singing dunes",
    "a volcanic eruption seen from a safe ridge",
    "a Scandinavian fishing village in winter",
    "an overgrown post-human Manhattan",
    "a hummingbird frozen mid-flight in a garden",
    "a Mughal palace reflecting pool at dusk",
    "a steampunk airship docking tower",
    "a mycelium network glowing underground",
    "a Formula 1 pit stop in dramatic lighting",
    "an Andean village during a festival",
    "a glassblower shaping molten glass",
    "a tide pool teeming with life at sunset",
    "a monastery carved into a cliff face",
    "a quantum computer laboratory",
    "a Mississippi riverboat in the 1880s",
    "an origami world where everything is folded paper",
    "a leopard resting in a baobab tree",
    "a Grand Central-style train station on another planet",
    "a chocolatier's kitchen mid-creation",
    "a Viking longship in a fjord under midnight sun",
    "a satellite's view of city lights and lightning",
    "a wildflower superbloom in a desert valley",
]

STYLES = [
    "ultra-detailed photorealistic photography, shallow depth of field",
    "cinematic film still, anamorphic lens, moody color grading",
    "dramatic chiaroscuro oil painting",
    "vibrant studio ghibli inspired animation still",
    "national geographic wildlife photography",
    "architectural digest editorial photograph",
    "impressionist painting with bold brushwork",
    "long-exposure night photography",
    "hyperrealistic 3D render, octane, volumetric lighting",
    "vintage kodachrome slide film photograph",
    "detailed matte painting for a fantasy film",
    "aerial drone photography, golden hour",
    "watercolor and ink illustration",
    "high-fashion editorial photography, dramatic lighting",
    "documentary street photography, decisive moment",
    "art deco poster illustration",
]


def sample_topics(n: int, rng: random.Random | None = None) -> list[str]:
    rng = rng or random
    return rng.sample(TOPICS, min(n, len(TOPICS)))


def sample_style(rng: random.Random | None = None) -> str:
    rng = rng or random
    return rng.choice(STYLES)
