# PENSIEVE-AI FAKE DEMO

A low-friction, web-based dementia screening mockup interface. This is a **demonstration only** and not for actual clinical use.

## Features

- **Start Test Page**: Simple, clean interface to begin the screening
- **Results Page**: Visual risk assessment with an axis showing:
  - 🟢 **Safe** (0.00 - 0.13): Low risk for cognitive decline
  - 🟡 **At Risk** (0.13 - 0.45): Moderate risk - consultation recommended
  - 🔴 **Confident** (0.45 - 1.00): High risk - professional evaluation needed

## Design

- **Color Scheme**: White background with green primary color for accessibility and clarity
- **Simple Layout**: Minimal design for low-friction user experience
- **Responsive**: Works on desktop and mobile devices

## Files

- `index.html` - Main HTML structure with start and results pages
- `style.css` - Styling with white and green color scheme
- `app.js` - Interactivity and mock score generation

## How to Use

1. Open `index.html` in a web browser
2. Click "Start Test" to proceed to the results page
3. View your mock risk score on the visual axis
4. Click "Redo Drawing" to generate a new random score
5. Click "Back to Home" to return to the start page

## Mock Behavior

- Each test generates a **random risk score** between 0 and 1
- The score determines which risk category you fall into
- Recommendations are automatically generated based on the score

## Notes

- This is a **FAKE DEMO** for UI/UX mockup purposes only
- Real implementation would integrate actual drawing analysis
- Risk thresholds (0.13 and 0.45) are based on project specifications
