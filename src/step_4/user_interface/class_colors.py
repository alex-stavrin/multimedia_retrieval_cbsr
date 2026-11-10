import os

# Mapping from shape_class to a semantically matching hex color.
# All colors are chosen with high enough brightness/luminance to remain clearly
# visible against a black/dark UI background.
SHAPE_CLASS_COLORS: dict[str, str] = {
    # Human & body parts (warm human skin tones)
    "Human": "#F4C29F",
    "Humanoid": "#F4C29F",
    "HumanHead": "#F7D1B5",
    "Hand": "#E8B490",
    # Animals
    "Quadruped": "#D99B66",  # Warm fawn / fur brown
    "Bird": "#4FC3F7",  # Sky feather blue
    "AquaticAnimal": "#26C6DA",  # Ocean teal
    "Fish": "#FF8A65",  # Coral salmon
    "Insect": "#C0CA33",  # Beetle chartreuse green
    # Plants & nature
    "Tree": "#4CAF50",  # Forest leaf green
    "PlantIndoors": "#66BB6A",  # Houseplant jade green
    "PlantWildNonTree": "#9CCC65",  # Wild meadow light green
    # Furniture & home
    "Bed": "#B39DDB",  # Soft lavender linen
    "NonWheelChair": "#D4A373",  # Warm teak wood
    "WheelChair": "#64B5F6",  # Accessible cobalt-sky blue
    "MultiSeat": "#F06292",  # Plush sofa rose
    "RectangleTable": "#DEB887",  # Burlywood / oak table
    "RoundTable": "#CD853F",  # Peru / cherry wood (brightened)
    "Shelf": "#E6BE8A",  # Pale gold / pine wood
    "Door": "#E09F67",  # Cedar wood
    # Lighting
    "DeskLamp": "#FFEE58",  # Warm bulb yellow
    "FloorLamp": "#FFD54F",  # Ambient amber glow
    # Musical instruments
    "Musical_Instrument": "#FFD740",  # Brass gold
    "ClassicPiano": "#FFF59D",  # Ivory / champagne key
    "PianoBoard": "#F5E6CC",  # Warm ivory keyboard
    "Guitar": "#F4A261",  # Acoustic honey wood
    "Violin": "#E76F51",  # Varnished maple / sienna
    "Drum": "#FFB74D",  # Warm bronze / drumhead
    # Road vehicles & transport
    "Car": "#FF5252",  # Sports car red
    "Motorcycle": "#FF6E40",  # Racing orange-red
    "Bicycle": "#AEEA00",  # High-vis lime green
    "Bus": "#FFC107",  # School bus amber-yellow
    "Truck": "#5C9CE6",  # Freight steel blue
    "TruckNonContainer": "#FF9800",  # Utility truck orange
    "Train": "#7986CB",  # Rail indigo-blue
    "Wheel": "#B0BEC5",  # Alloy / tire silver-grey
    # Aviation, marine & space
    "AircraftBuoyant": "#90CAF9",  # Blimp light sky blue
    "Biplane": "#E6C27A",  # Vintage canvas ochre
    "Monoplane": "#29B6F6",  # Aviation azure
    "Jet": "#80DEEA",  # Supersonic cyan-silver
    "Helicopter": "#FF7043",  # Rescue orange
    "Rocket": "#FF5722",  # Thruster flame orange
    "Starship": "#CE93D8",  # Cosmic orchid purple
    "Ship": "#42A5F5",  # Nautical sea blue
    # Military & weapons
    "MilitaryVehicle": "#A3C167",  # Bright olive drab
    "Gun": "#90A4AE",  # Gunmetal blue-grey
    "SubmachineGun": "#A1B5C1",  # Tactical light slate
    "Sword": "#CFD8DC",  # Polished blade silver
    "Knife": "#B8C9D9",  # Stainless chrome steel
    # Buildings & urban
    "House": "#E57373",  # Terracotta brick red
    "Apartment": "#FF8A80",  # Sandstone coral
    "BuildingNonResidential": "#D7CCC8",  # Concrete / limestone
    "Skyscraper": "#84FFFF",  # Glass tower cyan
    "City": "#9FA8DA",  # Urban skyline periwinkle
    # Electronics & office
    "Computer": "#40C4FF",  # Silicon electric blue
    "Monitor": "#18FFFF",  # Screen glow aqua
    "ComputerKeyboard": "#B8C6DB",  # Keycap cool silver
    "Cellphone": "#8C9EFF",  # Modern indigo-lilac
    "DeskPhone": "#80CBC4",  # Office mint-teal
    # Everyday objects & tableware
    "Bookset": "#D4AC6E",  # Leather-bound parchment
    "Bottle": "#69F0AE",  # Glass bottle mint green
    "Cup": "#A7FFEB",  # Ceramic pastel aqua
    "Mug": "#BCAAA4",  # Latte / ceramictaupe
    "Vase": "#F48FB1",  # Rose ceramic
    "Spoon": "#E0E0E0",  # Sterling silver
    "Glasses": "#F8BBD0",  # Rose-gold frame
    "Hat": "#FFE082",  # Straw sun-hat gold
    "Chess": "#EA80FC",  # Royal violet
    "Sign": "#FF80AB",  # Neon sign pink
    "Tool": "#FFAB40",  # Hardware safety orange
}

DEFAULT_CLASS_COLOR = "#BDBDBD"


def get_shape_class(shape_path: str) -> str:
    """Extract the shape class name from the parent directory of a shape file."""
    return os.path.basename(os.path.dirname(os.path.abspath(shape_path)))


def get_class_color(shape_class: str) -> str:
    """Return the hex color associated with a shape class."""
    return SHAPE_CLASS_COLORS.get(shape_class, DEFAULT_CLASS_COLOR)
