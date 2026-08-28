# RSEcon 2026 Causal Testing Walkthrough
Jupyter notebook "slides" for our RSEcon26 walkthrough.
We use this rather than normal slides so that we can have interactive code elemets.

## Setup
1. Start by creating a virtual environment, e.g.
```
virtualenv -p python3.12 --download venv
source venv/bin/activate
```

2. Install the dependencies:
```
pip install -e .
```

## Viewing the Slides
1. Run jupyter notebook:
```
jupyter notebook slides.ipynb
```
You should then see the jupyter notebook in the standard view.
You can edit and run the cells as you wish.

2. Press `alt + R` to view the slides. On a mac, you may need to press `⌘ + R` instead, but try `alt` first.
You should then see the slide view.
You can navigate with "page up" and "page down".
**DO NOT USE THE ARROW KEYS!** They will appear to work, but will miss out slide fragments.

> [!NOTE]
> If neither `alt + R` nor `⌘ + R` seems to do anything, you may need to explictly use the "classic" notebook view.
> To do this, click `View > Open in NbClassic`.

## Editing Slide Metadata
When you open the notebook, you should see a little dropdown box on each cell with "Slide", "Subslide", and "Fragment" options.
As above, if you do not, try `View > Open in NbClassic` and then `View > Cell Toolbar`.

- The **slide** option marks the start of a new slide. Use this just like you would for a new slide on Powerpoint etc.
- The **subslide** option marks a new part of a slide. This will appear on a fresh screen (previous content will disappear), but will be part of the same "slide". This is useful for showing code and its output since the amount you can actually fit on the screen is very small.
- The **fragment** option is for little snippets that will reveal themselves sequentially while the previous elements remain onscreen.

## Converting to HTML slides
To convert the notebook to HTML slides, uncomment `hv.output(fig='png')` in the first cell of the notebook and run
```
jupyter nbconvert slides.ipynb --to slides --execute --SlidesExporter.reveal_width=1600   --SlidesExporter.reveal_height=900
```
> [!WARNING]
> Don't forget to comment it back out again!
