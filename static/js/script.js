// Common JavaScript functions for the application

// Function to scroll to the bottom of a container
function scrollToBottom(containerElement) {
    if (containerElement) {
        containerElement.scrollTop = containerElement.scrollHeight;
    }
}

// Format a symptom string for display (replace underscores with spaces, capitalize)
function formatSymptom(symptom) {
    return symptom
        .replace(/_/g, ' ')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// Add an event listener for showing tooltips
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
});