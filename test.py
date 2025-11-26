from main import solve_puzzle

# my_words = ["MICKEY MOUSE", "BUG BITE", "HAPPY MEAL", "BARBIE DREAMHOUSE",
#             "LOTTERY TICKET", "GLAD-HAND", "RINKY-DINK", "MERRY-GO-ROUND",
#             "CHERRY BLOSSOM", "TRIVIAL", "SUNNY-SIDE UP", "CALAMINE LOTION",
#             "VINYL RECORD", "FLAMINGO", "TWO-BIT", "YOUR HEAD"]

my_words = ["BOLT", "ATTENTION", "DART", "DASH",
            "BALL", "THRILL", "BRAND", "DROP",
            "KICK", "BLAST", "GAME", "NAMES",
            "DEAR", "FOR", "TO", "FLY"]

# my_words = [
#   "ARTY", "KISS", "ENAMEL", "ESSAY", "CROWN", 
#   "DECAY", "BRUSH", "PASTE", "ANY", "SKIM", 
#   "PLASTER", "PULP", "STICK", "STROKE", "ROOT", 
#   "FIX"]

solution = solve_puzzle(my_words)
print("Solution:")
for group_id, group_words in solution.items():
    print(f"Group {group_id}: {group_words}")