// Generate a random risk score for the mock demo
function generateMockRiskScore() {
    // Return a random score in the "at risk" range (0.13 - 0.45)
    return 0.13 + Math.random() * (0.45 - 0.13);
}

// Update the results display based on the risk score
function updateResultsDisplay(riskScore) {
    const scoreValue = document.getElementById('scoreValue');
    const indicator = document.getElementById('indicator');
    const recommendationBox = document.getElementById('recommendationBox');
    const recommendationTitle = document.getElementById('recommendationTitle');
    const recommendationText = document.getElementById('recommendationText');
    
    // Update the score display with 2 decimal places
    scoreValue.textContent = riskScore.toFixed(2);
    
    // Move the indicator to the correct position (as percentage of 0-1 range)
    const indicatorPosition = riskScore * 100;
    indicator.style.left = indicatorPosition + '%';
    
    // Determine risk level and update recommendation
    let riskLevel;
    let riskClass;
    
    if (riskScore < 0.13) {
        riskLevel = 'Low Risk';
        riskClass = '';
        recommendationTitle.textContent = 'Low Risk';
        recommendationText.textContent = 
            'Your screening results suggest a low risk for cognitive decline. ' +
            'Continue with regular cognitive exercises and healthy lifestyle habits.';
    } else if (riskScore < 0.45) {
        riskLevel = 'At Risk';
        riskClass = 'at-risk';
        recommendationTitle.textContent = 'Moderate Risk';
        recommendationText.textContent = 
            'Your screening results indicate a moderate risk. ' +
            'We recommend consulting with a healthcare professional for further evaluation and cognitive assessment.';
    } else {
        riskLevel = 'High Risk';
        riskClass = 'confident';
        recommendationTitle.textContent = 'High Risk';
        recommendationText.textContent = 
            'Your screening results suggest a higher risk for cognitive decline. ' +
            'Please schedule an appointment with a healthcare professional for a comprehensive evaluation.';
    }
    
    // Update recommendation box styling
    recommendationBox.className = 'recommendation-box ' + riskClass;
}

// Start the test
function startTest() {
    // Hide start page
    document.getElementById('startPage').classList.remove('active');
    
    // Show results page
    document.getElementById('resultsPage').classList.add('active');
    
    // Generate mock result and display it
    const mockScore = generateMockRiskScore();
    
    // Simulate a drawing delay for realism
    setTimeout(() => {
        updateResultsDisplay(mockScore);
    }, 500);
}

// Redo the drawing (generate new mock score)
function retestDrawing() {
    const mockScore = generateMockRiskScore();
    updateResultsDisplay(mockScore);
}

// Go back to home page
function goToHome() {
    // Hide results page
    document.getElementById('resultsPage').classList.remove('active');
    
    // Show start page
    document.getElementById('startPage').classList.add('active');
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('PENSIEVE-AI Demo initialized');
});
