import cv2
import numpy as np

def get_color_name(rgb):
    """Map an RGB tuple to a simple human-readable color label using threshold rules."""
    r, g, b = rgb

    if r > 200 and g < 100 and b < 100:
        return "Red"
    elif g > 200 and r < 100 and b < 100:
        return "Green"
    elif b > 200 and r < 100 and g < 100:
        return "Blue"
    elif r > 200 and g > 200 and b < 100:
        return "Yellow"
    elif r > 150 and b > 150:
        return "Purple"
    elif g > 150 and b > 150:
        return "Cyan"
    elif r > 200 and g > 150 and b > 150:
        return "Pink"
    elif r > 180 and g > 180 and b > 180:
        return "White"
    elif r < 50 and g < 50 and b < 50:
        return "Black"
    else:
        return "Mixed"


def get_dominant_color_name(image_bgr, mask):
    """
    Extract dominant color from a masked region using HSV analysis.
    
    Args:
        image_bgr: Input image in BGR format (OpenCV)
        mask: Binary mask indicating the region of interest
        
    Returns:
        Color name (string) of the dominant color
    """
    if cv2.countNonZero(mask) == 0:
        return "Unknown"
    
    # Extract the masked region
    masked_region = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    
    # Convert to HSV
    hsv_region = cv2.cvtColor(masked_region, cv2.COLOR_BGR2HSV)
    
    # Sample only pixels inside the object mask.
    h_vals = hsv_region[mask == 255, 0]
    s_vals = hsv_region[mask == 255, 1]
    v_vals = hsv_region[mask == 255, 2]
    
    if len(h_vals) == 0:
        return "Unknown"
    
    avg_h = np.mean(h_vals)
    avg_s = np.mean(s_vals)
    avg_v = np.mean(v_vals)
    
    # Convert average HSV signature into a color category.
    return _hsv_to_color_name(avg_h, avg_s, avg_v)


def _hsv_to_color_name(h, s, v):
    """
    Convert HSV values to color name.
    
    Args:
        h: Hue (0-180 in OpenCV)
        s: Saturation (0-255)
        v: Value/Brightness (0-255)
        
    Returns:
        Color name (string)
    """
    # Check for achromatic colors (low saturation)
    if s < 50:
        if v < 50:
            return "Black"
        elif v > 200:
            return "White"
        else:
            return "Gray"
    
    # Very low brightness is treated as dark regardless of hue.
    if v < 50:
        return "Dark"
    
    # Determine hue-based color
    # Hue ranges: 0-30 (Red), 30-60 (Orange/Yellow), 60-90 (Yellow/Green), 90-150 (Green), 150-170 (Cyan), 170-180 (Blue-Magenta)
    if h < 10 or h > 170:
        return "Red"
    elif 10 <= h < 25:
        return "Orange"
    elif 25 <= h < 60:
        return "Yellow"
    elif 60 <= h < 90:
        return "Yellow-Green"
    elif 90 <= h < 130:
        return "Green"
    elif 130 <= h < 160:
        return "Cyan"
    elif 160 <= h <= 170:
        return "Blue"
    else:  # h > 170
        return "Purple"


def color_matches_filter(color_name, color_filter):
    """
    Check if a detected color matches the filter (allows partial matches).
    
    Args:
        color_name: Name of the detected color
        color_filter: Filter color name (can be partial, e.g., "Red" matches "Red")
        
    Returns:
        Boolean indicating if the color matches the filter
    """
    if not color_filter or color_filter.lower() == "all":
        return True
    
    color_name_lower = color_name.lower()
    filter_lower = color_filter.lower()
    
    # Support broad filters (e.g., "Green" can match "Yellow-Green").
    return filter_lower in color_name_lower or color_name_lower == filter_lower


def get_available_colors():
    """Return available color filter names used by the UI/API."""
    return [
        "All",
        "Red",
        "Orange",
        "Yellow",
        "Green",
        "Cyan",
        "Blue",
        "Purple",
        "White",
        "Black",
        "Gray"
    ]