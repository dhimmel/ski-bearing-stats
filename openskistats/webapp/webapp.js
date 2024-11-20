function matchesNumericFilter(value, filterValue) {
    filterValue = filterValue.trim();
    let match;

    // Handle "-" and "-0" as shorthand for <= 0
    if (filterValue === "-" || filterValue === "-0") {
        return value <= 0;
    }

    // Handle basic numbers (>= for positive, <= for negative)
    if ((match = filterValue.match(/^(-?\d+)$/))) {
        const threshold = parseFloat(match[1]);
        return threshold >= 0 ? value >= threshold : value <= threshold;
    }

    // Handle explicit ranges with brackets (inclusive) or parentheses (exclusive)
    if ((match = filterValue.match(/^([\[\(])\s*(-?\d+)?\s*,\s*(-?\d+)?\s*([\]\)])$/))) {
        const lower = match[2] !== undefined ? parseFloat(match[2]) : -Infinity;
        const upper = match[3] !== undefined ? parseFloat(match[3]) : Infinity;
        const lowerInclusive = match[1] === '['; // True if inclusive lower
        const upperInclusive = match[4] === ']'; // True if inclusive upper

        return (
            (lowerInclusive ? value >= lower : value > lower) &&
            (upperInclusive ? value <= upper : value < upper)
        );
    }

    // Default: invalid filterValue returns true
    return true;
}

function showTooltip(event) {
    const tooltip = event.currentTarget.querySelector('.tooltip-image');
    const rect = event.currentTarget.getBoundingClientRect();
    const tooltipWidth = 300; // Assume tooltip image has a fixed width
    const tooltipHeight = 300; // Assume tooltip image has a fixed height
    const previewHeight = rect.height; // Height of the preview image

    // Position to the left of the preview image
    let left = rect.left - tooltipWidth - 10; // Add a small gap of 10px

    // Center vertically on the preview image
    let top = rect.top + (previewHeight / 2) - (tooltipHeight / 2);

    // Check if the tooltip overflows the left edge of the viewport
    if (left < 0) {
        left = 10; // Adjust to prevent overflow with padding
    }

    // Check if the tooltip overflows the top edge of the viewport
    if (top < 0) {
        top = 10; // Adjust to prevent overflow with padding
    }

    // Check if the tooltip overflows the bottom edge of the viewport
    if (top + tooltipHeight > window.innerHeight) {
        top = window.innerHeight - tooltipHeight - 10; // Adjust to prevent overflow
    }

    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
    tooltip.style.visibility = 'visible';
    tooltip.style.opacity = '1';
}


function hideTooltip(event) {
    const tooltip = event.currentTarget.querySelector('.tooltip-image');
    tooltip.style.visibility = 'hidden';
    tooltip.style.opacity = '0';
}
