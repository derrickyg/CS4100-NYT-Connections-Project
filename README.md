install dependencies:
```bash
pip install -r requirements.txt
```

run program:
```bash
python main.py --mistakes-allowed 4 --num-puzzles 10
```

^ where 

`--mistakes-allowed` is the number of mistakes allowed in the game (default is 4, just like the real nyt connections game)

`--num-puzzles` is the number of puzzles to evaluate on (default is all ~652 puzzles in the dataset)