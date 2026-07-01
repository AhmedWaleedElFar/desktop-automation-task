# Vision-Based Desktop Automation with Dynamic Icon Grounding

A robust, AI-driven desktop automation pipeline that utilizes Vision Language Models (VLMs) to dynamically locate and interact with applications on a Windows desktop. 

This project goes beyond simple hardcoded coordinates or basic image-template matching. It implements the cutting-edge **ScreenSeekeR** visual grounding architecture, allowing it to locate targets (like the Notepad icon) regardless of their position, background wallpaper, or unexpected UI pop-ups.

---

## 🎯 The Automation Task
This system automates a full data entry workflow:
1. Ground and double-click the **Notepad** icon on the desktop.
2. Fetch blog posts from the [JSONPlaceholder API](https://jsonplaceholder.cypress.io/posts).
3. Type the post Title and Body into Notepad.
4. Save the file natively (using `Ctrl+S` and OS dialog navigation) to a specific output folder.
5. Close the window and repeat for the next post.

---

## 🧠 Theoretical Foundation: The ScreenSeekeR Architecture
This project is built around the methodologies described in the paper [ScreenSeekeR](https://arxiv.org/pdf/2504.07981).

To evaluate general, scalable visual grounding ability (rather than task-specific shortcuts), the core vision engine has been entirely decoupled into its own reusable `screenseeker` package. The pipeline executes in four distinct phases:

1. **Phase 1: Planning**  
   The VLM Planner analyzes the full 1920x1080 screenshot and proposes coarse "candidate regions" (neighborhoods) where the target icon might be located.
2. **Phase 2: Grounding**  
   The VLM Grounder analyzes the image and extracts precise spatial "voting boxes" for anything resembling the target instruction.
3. **Phase 3: Scoring (Gaussian Centrality)**  
   A mathematical scoring algorithm ranks the Planner's candidate regions based on the density and confidence of the Grounder's voting boxes inside them. *Crucially, to prevent misclicks caused by oversized Planner regions, the system extracts the exact `(x,y)` click coordinates from the highest-scoring Grounder vote.*
4. **Phase 4: Verification**  
   The highest-ranking region is cropped and verified to ensure it matches the target before execution.

---

## 📁 Project Structure

```text
src/
├── screenseeker/              # Task-agnostic Visual Grounding Engine
│   ├── __init__.py
│   ├── recursive_search.py    # Coordinator engine chaining the phases
│   ├── planner.py             # Phase 1: Candidate region generation
│   ├── grounder.py            # Phase 2: Voting box generation
│   ├── scoring.py             # Phase 3: Gaussian centrality math
│   ├── verifier.py            # Phase 4: Target verification
│   └── visualizer.py          # Decoupled utility for rendering debug images
├── automation.py              # Notepad-specific automation & typing logic
├── main.py                    # CLI entry point
└── screenshot.py              # OS-level screenshot capture
```

---

## 🚀 How to Run

### Prerequisites
- Windows OS (target resolution: 1920x1080)
- `uv` package manager installed
- Python 3.10+
- A "Notepad" shortcut placed anywhere on your desktop.

### Execution
Run the full automation pipeline via the CLI. It accepts an argument for the number of posts you want to process.

```bash
cd src
uv run python main.py --automate 3
```

You can also run component tests or test the visual search engine in isolation:
```bash
uv run python main.py --test
uv run python main.py --search
```

### Logging & Visual Outputs
All runs generate rich, annotated debugging images inside the `src/logs/` directory.
- `logs/planner_regions/`: Shows the coarse neighborhoods proposed by Phase 1.
- `logs/grounder_boxes/`: Shows the raw voting boxes proposed by Phase 2.
- `logs/detections/`: Shows the final, verified `(x, y)` target coordinate that the automation will click.
