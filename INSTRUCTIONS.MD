# Project Title: Vision-Based Desktop Automation with Dynamic Icon Grounding
## Objective:
Develop a Python application that uses computer vision to dynamically locate and interact with desktop icons on Windows (1920x1080 resolution). The system must find the Notepad icon regardless of its position on the desktop, enabling robust automation even when icon positions change.

## Part 1: Design Doc (up to 90 minutes, no code)
Write a 1–4 page design document for the system described below. 
No code required.
You may structure the document however you like. Examples of topics include: assumptions, system design, grounding strategy, tradeoffs, and more.

## Part 2: Implementation (up to 3 hours, Git repo)
### Requirements:
#### Setup:
Target OS: Windows 10/11 at 1920x1080 resolution


Required capabilities: screenshot capture, mouse/keyboard control, image processing


Pre-requisite: Create a Notepad shortcut icon on the desktop before running

#### Core Automation Task:

Application Target: Notepad (launched via desktop icon)

Data Source: Fetch blog posts from JSONPlaceholder API
(https://jsonplaceholder.typicode.com/posts)

Visual Grounding Requirements:

Capture a screenshot of the desktop
Implement a grounding system that can:
1- Locate the Notepad icon on the desktop (primary challenge)
2- Identify the icon regardless of its position
3- Return the center coordinates (x, y) for clicking
4- Handle cases where the icon might be in different locations

#### Automation Workflow:

Take a screenshot of the desktop
Ground the Notepad desktop icon → double-click it to launch
For each of the first 10 posts from the API:
- Type the post content (format: "Title: {title}\n\n{body}") in the text area
- Save the post in this format: Type filename: "post_{id}.txt", Directory: Desktop/tjm-project directory
- Close Notepad
- Repeat: Take fresh screenshot → ground Notepad icon → launch

#### Desktop Icon Grounding Implementation:

Be ready to discuss Grounding Implementations during the interview as a focus area. We are looking for the most flexible implementation, where we can bypass things like unexpected pop-ups without knowing what they look like in advance. The assignment is for a notepad, but it should work for any icon or button even if we don’t have the exact image or text beforehand.


For an example implementation, please see: https://arxiv.org/pdf/2504.07981. 


Side note: This approach may feel overkill for launching Notepad. This is intentional. The goal is to evaluate general, scalable visual grounding ability, not task-specific shortcuts.

Please note that a submission without a grounding implementation based on the referenced paper, will be rejected.


#### Error Handling & Robustness:
Develop your own methodology and cases. 

#### Deliverables:
- Source code with clear structure on Github
- uv configuration
- At least 3 annotated screenshots showing:
1- Icon detected in top-left area
2- Icon detected in bottom-right area
3- Icon detected in center of screen

#### Interview Test Scenario:

During the interview, we will:
Move the Notepad icon to different desktop positions
Run your executable
Verify it correctly locates and clicks the icon
Discuss what happens if detection fails

#### Bonus Challenges (Optional):

Detect multiple desktop icons and select the correct one
Handle different icon sizes (small, medium, large Windows settings)
Support both light and dark desktop themes

## Discussion Topics:
Be prepared to discuss:

- Icon detection approach: Why this method over alternatives?
- Failure cases: When would your detection fail? How would you improve it?
- Performance: How long does icon detection take? Optimization strategies?
- Robustness: How does your solution handle different scenarios?
- Scaling: How would you extend this to other scenarios?
- Alternative approaches: What would you do differently with more time?


