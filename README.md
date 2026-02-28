# SousChef

AI cooking assistant that watches your stovetop through a camera and gives real-time feedback.

## Setup

### 1. Install dependencies

```bash
pip install openai python-dotenv
```

### 2. Create a `.env` file

Create a `.env` file in the project root with your OpenRouter API key:

```
OPENROUTER_API_KEY=your_key_here
```

Get an API key at [openrouter.ai](https://openrouter.ai/).

## Simulator

`simulate.py` is an interactive terminal assistant that lets you test SousChef using photos from disk instead of a live camera.

```bash
python simulate.py
```

### Commands

| Command | Description |
|---------|-------------|
| `load <path>` | Load an image as the current camera frame |
| `temp <min> <max> <avg>` | Set mock thermal readings in °C |
| `help` | Show available commands |
| `quit` / `exit` | Exit the simulator |

### Proactive feedback (system-initiated)

SousChef automatically analyzes the scene whenever you load an image or update thermal readings — no question needed.

```
> load data/stove.webp
Loaded: data/stove.webp
The pot on the left burner has a small blue flame underneath it, suggesting it's heating up.

> temp 25 180 90
Thermal set: min=25.0°C  max=180.0°C  avg=90.0°C
That pot is very hot at 180°C — make sure to use oven mitts if you need to touch it.
```

This simulates the real Pi system's proactive observation loop, which periodically checks the camera and speaks up when something notable is happening.

### Natural language questions (user-initiated)

Type anything that isn't a command and SousChef treats it as a question. It answers using the current image, thermal data, and recent conversation history.

```
> what should I cook?
With that pot nice and hot, you could get a good sear on some meat or start sautéing aromatics for a soup.

> is it ready to flip?
Looking at your egg, the whites are still translucent near the yolk. Give it another 30-60 seconds.
```

You can ask questions without loading an image first, but responses will be more useful with visual context.

### Example session

```bash
python simulate.py
```

```
SousChef Simulator
Type 'help' for commands.

> load data/stove.webp          # proactive: SousChef describes what it sees
> temp 25 85 50                 # proactive: comments on temperature if notable
> is the pan hot enough?        # reactive: answers your question
> load data/egg_frying.jpg      # proactive: new scene analysis
> is it ready to flip?          # reactive: answers using image + thermal + history
> quit
```
