"""
Simple sample puzzles designed to be solvable by the model.
These puzzles use clear semantic relationships that word embeddings can capture.
"""
from data.load_dataset import Puzzle

# note: used AI to generate these
SAMPLE_PUZZLES = [
    # Puzzle 1: Basic categories
    Puzzle(
        puzzle_id="sample_1",
        words=["RED", "BLUE", "GREEN", "YELLOW", "CAT", "DOG", "BIRD", "FISH", 
               "HAPPY", "SAD", "ANGRY", "EXCITED", "CAR", "BIKE", "TRAIN", "PLANE"],
        groups={
            1: ["RED", "BLUE", "GREEN", "YELLOW"],
            2: ["CAT", "DOG", "BIRD", "FISH"],
            3: ["HAPPY", "SAD", "ANGRY", "EXCITED"],
            4: ["CAR", "BIKE", "TRAIN", "PLANE"]
        },
        category_descriptions={
            1: "COLORS",
            2: "ANIMALS",
            3: "EMOTIONS",
            4: "VEHICLES"
        },
        difficulty=1.0,
        contest="Sample Puzzle 1 - Basic Categories"
    ),
    
    # Puzzle 2: Body parts
    Puzzle(
        puzzle_id="sample_2",
        words=["HEAD", "ARM", "LEG", "FOOT", "APPLE", "BANANA", "ORANGE", "GRAPE",
               "TABLE", "CHAIR", "DESK", "SOFA", "SUN", "MOON", "STAR", "PLANET"],
        groups={
            1: ["HEAD", "ARM", "LEG", "FOOT"],
            2: ["APPLE", "BANANA", "ORANGE", "GRAPE"],
            3: ["TABLE", "CHAIR", "DESK", "SOFA"],
            4: ["SUN", "MOON", "STAR", "PLANET"]
        },
        category_descriptions={
            1: "BODY PARTS",
            2: "FRUITS",
            3: "FURNITURE",
            4: "CELESTIAL BODIES"
        },
        difficulty=1.2,
        contest="Sample Puzzle 2 - Clear Categories"
    ),
    
    # Puzzle 3: Time-related
    Puzzle(
        puzzle_id="sample_3",
        words=["MORNING", "AFTERNOON", "EVENING", "NIGHT", "SPRING", "SUMMER", "FALL", "WINTER",
               "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "JANUARY", "FEBRUARY", "MARCH", "APRIL"],
        groups={
            1: ["MORNING", "AFTERNOON", "EVENING", "NIGHT"],
            2: ["SPRING", "SUMMER", "FALL", "WINTER"],
            3: ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY"],
            4: ["JANUARY", "FEBRUARY", "MARCH", "APRIL"]
        },
        category_descriptions={
            1: "TIMES OF DAY",
            2: "SEASONS",
            3: "DAYS OF WEEK",
            4: "MONTHS"
        },
        difficulty=1.3,
        contest="Sample Puzzle 3 - Time Concepts"
    ),
    
    # Puzzle 4: Actions
    Puzzle(
        puzzle_id="sample_4",
        words=["RUN", "WALK", "JUMP", "CLIMB", "READ", "WRITE", "DRAW", "PAINT",
               "SING", "DANCE", "PLAY", "ACT", "COOK", "BAKE", "FRY", "GRILL"],
        groups={
            1: ["RUN", "WALK", "JUMP", "CLIMB"],
            2: ["READ", "WRITE", "DRAW", "PAINT"],
            3: ["SING", "DANCE", "PLAY", "ACT"],
            4: ["COOK", "BAKE", "FRY", "GRILL"]
        },
        category_descriptions={
            1: "PHYSICAL ACTIVITIES",
            2: "CREATIVE ACTIVITIES",
            3: "PERFORMANCE ACTIVITIES",
            4: "COOKING METHODS"
        },
        difficulty=1.4,
        contest="Sample Puzzle 4 - Actions"
    ),
    
    # Puzzle 5: Nature
    Puzzle(
        puzzle_id="sample_5",
        words=["TREE", "FLOWER", "GRASS", "LEAF", "OCEAN", "RIVER", "LAKE", "STREAM",
               "MOUNTAIN", "HILL", "VALLEY", "PLAIN", "CLOUD", "RAIN", "SNOW", "WIND"],
        groups={
            1: ["TREE", "FLOWER", "GRASS", "LEAF"],
            2: ["OCEAN", "RIVER", "LAKE", "STREAM"],
            3: ["MOUNTAIN", "HILL", "VALLEY", "PLAIN"],
            4: ["CLOUD", "RAIN", "SNOW", "WIND"]
        },
        category_descriptions={
            1: "PLANTS",
            2: "BODIES OF WATER",
            3: "LAND FORMS",
            4: "WEATHER PHENOMENA"
        },
        difficulty=1.5,
        contest="Sample Puzzle 5 - Nature"
    ),
    
    # Puzzle 6: Food categories
    Puzzle(
        puzzle_id="sample_6",
        words=["BREAD", "RICE", "PASTA", "CEREAL", "BEEF", "CHICKEN", "PORK", "FISH",
               "MILK", "CHEESE", "YOGURT", "BUTTER", "SALAD", "SOUP", "STEW", "CURRY"],
        groups={
            1: ["BREAD", "RICE", "PASTA", "CEREAL"],
            2: ["BEEF", "CHICKEN", "PORK", "FISH"],
            3: ["MILK", "CHEESE", "YOGURT", "BUTTER"],
            4: ["SALAD", "SOUP", "STEW", "CURRY"]
        },
        category_descriptions={
            1: "GRAINS",
            2: "MEATS",
            3: "DAIRY PRODUCTS",
            4: "DISHES"
        },
        difficulty=1.6,
        contest="Sample Puzzle 6 - Food"
    ),
    
    # Puzzle 7: Clothing
    Puzzle(
        puzzle_id="sample_7",
        words=["SHIRT", "PANTS", "DRESS", "SKIRT", "SHOE", "BOOT", "SANDAL", "SNEAKER",
               "HAT", "CAP", "HELMET", "CROWN", "GLOVE", "MITTEN", "SCARF", "TIE"],
        groups={
            1: ["SHIRT", "PANTS", "DRESS", "SKIRT"],
            2: ["SHOE", "BOOT", "SANDAL", "SNEAKER"],
            3: ["HAT", "CAP", "HELMET", "CROWN"],
            4: ["GLOVE", "MITTEN", "SCARF", "TIE"]
        },
        category_descriptions={
            1: "CLOTHING ITEMS",
            2: "FOOTWEAR",
            3: "HEADWEAR",
            4: "ACCESSORIES"
        },
        difficulty=1.7,
        contest="Sample Puzzle 7 - Clothing"
    ),
    
    # Puzzle 8: Sports
    Puzzle(
        puzzle_id="sample_8",
        words=["BASKETBALL", "FOOTBALL", "SOCCER", "BASEBALL", "TENNIS", "GOLF", "SWIM", "RUN",
               "BALL", "NET", "RACKET", "CLUB", "HELMET", "PAD", "GLOVE", "SHOE"],
        groups={
            1: ["BASKETBALL", "FOOTBALL", "SOCCER", "BASEBALL"],
            2: ["TENNIS", "GOLF", "SWIM", "RUN"],
            3: ["BALL", "NET", "RACKET", "CLUB"],
            4: ["HELMET", "PAD", "GLOVE", "SHOE"]
        },
        category_descriptions={
            1: "TEAM SPORTS",
            2: "INDIVIDUAL SPORTS",
            3: "SPORTS EQUIPMENT",
            4: "PROTECTIVE GEAR"
        },
        difficulty=1.8,
        contest="Sample Puzzle 8 - Sports"
    ),
    
    # Puzzle 9: School subjects
    Puzzle(
        puzzle_id="sample_9",
        words=["MATH", "SCIENCE", "HISTORY", "ENGLISH", "BOOK", "PEN", "PAPER", "PENCIL",
               "TEACHER", "STUDENT", "PROFESSOR", "SCHOLAR", "READ", "WRITE", "STUDY", "LEARN"],
        groups={
            1: ["MATH", "SCIENCE", "HISTORY", "ENGLISH"],
            2: ["BOOK", "PEN", "PAPER", "PENCIL"],
            3: ["TEACHER", "STUDENT", "PROFESSOR", "SCHOLAR"],
            4: ["READ", "WRITE", "STUDY", "LEARN"]
        },
        category_descriptions={
            1: "SCHOOL SUBJECTS",
            2: "SCHOOL SUPPLIES",
            3: "PEOPLE IN EDUCATION",
            4: "LEARNING ACTIVITIES"
        },
        difficulty=1.9,
        contest="Sample Puzzle 9 - Education"
    ),
    
    # Puzzle 10: Technology
    Puzzle(
        puzzle_id="sample_10",
        words=["COMPUTER", "PHONE", "TABLET", "LAPTOP", "KEYBOARD", "MOUSE", "MONITOR", "PRINTER",
               "EMAIL", "TEXT", "CALL", "VIDEO", "INTERNET", "WEBSITE", "APP", "SOFTWARE"],
        groups={
            1: ["COMPUTER", "PHONE", "TABLET", "LAPTOP"],
            2: ["KEYBOARD", "MOUSE", "MONITOR", "PRINTER"],
            3: ["EMAIL", "TEXT", "CALL", "VIDEO"],
            4: ["INTERNET", "WEBSITE", "APP", "SOFTWARE"]
        },
        category_descriptions={
            1: "DEVICES",
            2: "COMPUTER PERIPHERALS",
            3: "COMMUNICATION METHODS",
            4: "DIGITAL CONCEPTS"
        },
        difficulty=2.0,
        contest="Sample Puzzle 10 - Technology"
    )
]


def get_sample_puzzles():
    """Get list of sample puzzles."""
    return SAMPLE_PUZZLES


def get_sample_puzzle_by_index(index: int) -> Puzzle:
    """Get a specific sample puzzle by index."""
    if 0 <= index < len(SAMPLE_PUZZLES):
        return SAMPLE_PUZZLES[index]
    raise IndexError(f"Sample puzzle index {index} out of range. Available: 0-{len(SAMPLE_PUZZLES)-1}")

