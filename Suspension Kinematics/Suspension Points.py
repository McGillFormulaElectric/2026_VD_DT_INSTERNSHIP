# Author: Ludih
# Summary: This script holds the suspension points.

import numpy as np

hardpointsFL = {
    # Lower Wishbone
    "lower_wishbone_fore":    np.array([0.7544, 0.2105, 0.1441]),
    "lower_wishbone_aft":     np.array([0.4436, 0.2502, 0.1401]),
    "lower_ball_joint":       np.array([0.762, 0.5012, 0.1143]),

    # Upper Wishbone
    "upper_wishbone_fore":    np.array([0.8144, 0.2318, 0.2955]),
    "upper_wishbone_aft":     np.array([0.507, 0.2597, 0.2945]),
    "upper_ball_joint":       np.array([0.7468, 0.4775, 0.2845]),

    # Tie Rod
    "tie_rod_chassis":        np.array([0.838, 0.179, 0.1458]),
    "tie_rod_outer":          np.array([0.8179, 0.5012, 0.1143]),

    # Wheel Points
    "upper_wheel_pt":         np.array([0.7625, 0.575, 0.406]),
    "contact_patch":          np.array([0.7625, 0.575, 0]),
    "fore_wheel_pt":          np.array([0.9655, 0.575, 0.203]),
    "aft_wheel_pt":           np.array([0.5595, 0.575, 0.203]),

    # Pushrod and Rocker
    "pushrod_outboard":       np.array([0.7467, 0.4522, 0.3014]),
    "rocker_fore_mount":      np.array([x, y, z]),
    "rocker_aft_mount":       np.array([x, y, z]),
    "rocker_pushrod_arm":     np.array([x, y, z]),
    "rocker_damper_arm":      np.array([x, y, z]),
    "damper_chassis":         np.array([x, y, z]),
}

hardpointsRL = {
    # Lower Wishbone
    "lower_wishbone_fore":    np.array([-0.4943, 0.2626, 0.1496]),
    "lower_wishbone_aft":     np.array([-0.8755, 0.2098, 0.1504]),
    "lower_ball_joint":       np.array([-0.7417, 0.4775, 0.1092]),

    # Upper Wishbone
    "upper_wishbone_fore":    np.array([-0.5128, 0.2728, 0.293]),
    "upper_wishbone_aft":     np.array([-0.8706, 0.2376, 0.2927]),
    "upper_ball_joint":       np.array([-0.7417, 0.4775, 0.1092]),

    # Tie Rod
    "tie_rod_chassis":        np.array([-0.8685, 0.2393, 0.2603]),
    "tie_rod_outer":          np.array([-0.8331, 0.5232, 0.2603]),

     # Wheel Points
    "upper_wheel_pt":         np.array([-0.7625, 0.575, 0.406]),
    "contact_patch":          np.array([-0.7625, 0.575, 0]),
    "fore_wheel_pt":          np.array([-0.5595, 0.575, 0.203]),
    "aft_wheel_pt":           np.array([-0.9655, 0.575, 0.203]),

    # Pushrod and Rocker
    "pushrod_outboard":       np.array([-0.7417, 0.4507, 0.3012]),
    "rocker_fore_mount":      np.array([x, y, z]),
    "rocker_aft_mount":       np.array([x, y, z]),
    "rocker_pushrod_arm":     np.array([x, y, z]),
    "rocker_damper_arm":      np.array([x, y, z]),
    "damper_chassis":         np.array([x, y, z]),
}


metadata = {
    "corner": "FR",
    "units": "mm",
}
