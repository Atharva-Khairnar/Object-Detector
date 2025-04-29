// This script handles the loading animation and messages
document.addEventListener('DOMContentLoaded', function() {
    // Array of loading messages to cycle through
    const loadingMessages = [
        "Analyzing image...",
        "Detecting objects...",
        "Identifying patterns...",
        "Applying object recognition...",
        "Creating color-coded tags...",
        "Processing almost complete..."
    ];
    
    const loadingText = document.getElementById('loading-text');
    if (loadingText) {
        let messageIndex = 0;
        
        // Change loading message every 2 seconds
        setInterval(function() {
            loadingText.textContent = loadingMessages[messageIndex];
            messageIndex = (messageIndex + 1) % loadingMessages.length;
        }, 2000);
    }
    
    // If we're on the result page, scroll to results
    if (window.location.pathname.includes('result')) {
        const resultContainer = document.getElementById('result-container');
        if (resultContainer) {
            resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }
});
