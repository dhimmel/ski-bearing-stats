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

// Generates HTML for a donut chart given a percentage value (out of 1) and color
// Modified from <https://glin.github.io/reactable/articles/popular-movies/popular-movies.html>
function donutChart(value, color = '#a100bf') {
    // Format as percentage
    const pct = (value * 100).toFixed(0)
    // All units are in rem for relative scaling
    const radius = 1.5
    const diameter = 3.75
    const center = diameter / 2
    const width = 0.25
    const sliceLength = 2 * Math.PI * radius
    const sliceOffset = sliceLength * (1 - value)
    const donutChart = `
      <svg width="${diameter}rem" height="${diameter}rem" style="transform: rotate(-90deg)" focusable="false">
        <circle cx="${center}rem" cy="${center}rem" r="${radius}rem" fill="none" stroke-width="${width}rem" stroke="rgba(0,0,0,0.1)"></circle>
        <circle cx="${center}rem" cy="${center}rem" r="${radius}rem" fill="none" stroke-width="${width}rem" stroke="${color}"
         stroke-dasharray="${sliceLength}rem" stroke-dashoffset="${sliceOffset}rem"></circle>
      </svg>
    `
    const label = `
      <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%)">
        ${pct}%
      </div>
    `
    return `
      <div style="display: inline-flex; position: relative">
        ${donutChart}
        ${label}
      </div>
    `
}
  