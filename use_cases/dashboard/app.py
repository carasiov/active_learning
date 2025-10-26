"""Dashboard with intelligent proportional panel resizing."""

from __future__ import annotations

import sys
from pathlib import Path

from dash import Dash, Input, Output
import dash_bootstrap_components as dbc

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(APP_DIR) in sys.path:
    sys.path = [p for p in sys.path if p != str(APP_DIR)]

from use_cases.dashboard.layouts import build_dashboard_layout  # noqa: E402
from use_cases.dashboard.state import initialize_model_and_data  # noqa: E402
from use_cases.dashboard.callbacks.training_callbacks import register_training_callbacks  # noqa: E402
from use_cases.dashboard.callbacks.visualization_callbacks import register_visualization_callbacks  # noqa: E402
from use_cases.dashboard.callbacks.labeling_callbacks import register_labeling_callbacks  # noqa: E402


CUSTOM_CSS = """
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    padding: 0;
    overflow: hidden;
}

/* Focus rings for accessibility */
button:focus-visible,
input:focus-visible,
.form-check-input:focus-visible {
    outline: 3px solid #007AFF !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.2) !important;
}

button:hover:not(:disabled) {
    opacity: 0.85;
    transform: translateY(-1px);
}

button:active:not(:disabled) {
    transform: translateY(0);
}

button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

input:focus {
    outline: none;
    border-color: #007AFF !important;
    box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.1);
}

.form-check-input:checked {
    background-color: #007AFF;
    border-color: #007AFF;
}

/* Radio buttons focus */
.form-check-input[type="radio"]:focus-visible {
    outline: 3px solid #007AFF !important;
    outline-offset: 2px !important;
}

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #f5f5f7;
}

::-webkit-scrollbar-thumb {
    background: #c7c7cc;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #86868b;
}

/* Resize handles */
.resize-handle {
    position: relative;
    transition: background-color 0.15s;
}

#left-resize-handle::after,
#right-resize-handle::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 3px;
    height: 40px;
    background-color: #d1d1d6;
    border-radius: 2px;
    opacity: 0;
    transition: opacity 0.2s;
}

#horizontal-resize-handle::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 40px;
    height: 3px;
    background-color: #d1d1d6;
    border-radius: 2px;
    opacity: 0;
    transition: opacity 0.2s;
}

.resize-handle:hover::after {
    opacity: 1;
}

.resize-handle:hover {
    background-color: rgba(0, 122, 255, 0.1);
}

button[id*="label-button"]:hover {
    background-color: #007AFF !important;
    color: #ffffff !important;
}

#delete-label-button:hover {
    background-color: #FF3B30 !important;
    color: #ffffff !important;
}

#start-training-button:hover:not(:disabled) {
    background-color: #2FB74A !important;
    box-shadow: 0 4px 12px rgba(52, 199, 89, 0.3);
}

#modal-confirm-button:hover:not(:disabled) {
    background-color: #2FB74A !important;
}

#modal-cancel-button:hover:not(:disabled) {
    background-color: #f5f5f7 !important;
}

button, input {
    transition: all 0.2s ease;
}

.resizing {
    user-select: none !important;
}

.resizing * {
    pointer-events: none !important;
}

/* Improve modal accessibility */
.modal-content {
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12) !important;
}

.modal-header {
    border-bottom: 1px solid #e5e5e5 !important;
    padding: 20px 24px !important;
}

.modal-body {
    padding: 24px !important;
}

.modal-footer {
    border-top: 1px solid #e5e5e5 !important;
    padding: 16px 24px !important;
}
"""


def create_app() -> Dash:
    """Create and configure the Dash application."""
    initialize_model_and_data()

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=False,
    )
    app.title = "SSVAE Active Learning"
    
    app.index_string = f'''
    <!DOCTYPE html>
    <html lang="en">
        <head>
            {{%metas%}}
            <title>{{%title%}}</title>
            {{%favicon%}}
            {{%css%}}
            <style>
                {CUSTOM_CSS}
            </style>
        </head>
        <body>
            {{%app_entry%}}
            <footer>
                {{%config%}}
                {{%scripts%}}
                {{%renderer%}}
            </footer>
        </body>
    </html>
    '''
    
    app.layout = build_dashboard_layout()

    # Smart proportional resize handler
    app.clientside_callback(
        """
        function(n_intervals) {
            if (!n_intervals || window.__resizeHandlersInstalled) {
                return window.dash_clientside.no_update;
            }
            
            setTimeout(function() {
                const leftHandle = document.getElementById('left-resize-handle');
                const rightHandle = document.getElementById('right-resize-handle');
                const horizontalHandle = document.getElementById('horizontal-resize-handle');
                const leftPanel = document.getElementById('left-panel');
                const centerPanel = document.getElementById('center-panel');
                const rightPanel = document.getElementById('right-panel');
                const latentContainer = document.getElementById('latent-plot-container');
                const lossContainer = document.getElementById('loss-plot-container');
                const container = document.getElementById('main-container');
                
                if (!leftHandle || !rightHandle || !horizontalHandle || 
                    !leftPanel || !centerPanel || !rightPanel ||
                    !latentContainer || !lossContainer || !container) {
                    console.error('Resize elements not found');
                    return;
                }
                
                console.log('Installing smart resize handlers...');
                
                // State tracking
                let isResizing = false;
                let activeHandle = null;
                let startX = 0, startY = 0;
                let startWidths = {};
                let startHeights = {};
                let containerWidth = 0, containerHeight = 0;
                
                // Constraints (percentages for vertical, pixels for horizontal)
                const MIN_LEFT = 15, MAX_LEFT = 35;
                const MIN_CENTER = 40, MAX_CENTER = 70;
                const MIN_RIGHT = 15, MAX_RIGHT = 35;
                const MIN_LATENT = 300, MIN_LOSS = 120;
                
                function clamp(value, min, max) {
                    return Math.max(min, Math.min(max, value));
                }
                
                function startResize(e, handle) {
                    isResizing = true;
                    activeHandle = handle;
                    startX = e.clientX;
                    startY = e.clientY;
                    containerWidth = container.offsetWidth;
                    containerHeight = centerPanel.offsetHeight;
                    
                    // Capture current widths (in percentages)
                    startWidths = {
                        left: (leftPanel.offsetWidth / containerWidth) * 100,
                        center: (centerPanel.offsetWidth / containerWidth) * 100,
                        right: (rightPanel.offsetWidth / containerWidth) * 100,
                    };
                    
                    // Capture current heights (in pixels)
                    startHeights = {
                        latent: latentContainer.offsetHeight,
                        loss: lossContainer.offsetHeight,
                    };
                    
                    document.body.classList.add('resizing');
                    if (handle === horizontalHandle) {
                        document.body.style.cursor = 'row-resize';
                    } else {
                        document.body.style.cursor = 'col-resize';
                    }
                    e.preventDefault();
                }
                
                function doResize(e) {
                    if (!isResizing) return;
                    
                    if (activeHandle === leftHandle) {
                        // Left handle: resize left, distribute to center and right proportionally
                        const deltaX = e.clientX - startX;
                        const deltaPercent = (deltaX / containerWidth) * 100;
                        
                        let newLeftWidth = clamp(
                            startWidths.left + deltaPercent,
                            MIN_LEFT,
                            MAX_LEFT
                        );
                        
                        // Remaining space for center + right
                        const remainingWidth = 100 - newLeftWidth;
                        const currentOtherWidth = startWidths.center + startWidths.right;
                        const centerRatio = startWidths.center / currentOtherWidth;
                        const rightRatio = startWidths.right / currentOtherWidth;
                        
                        let newCenterWidth = remainingWidth * centerRatio;
                        let newRightWidth = remainingWidth * rightRatio;
                        
                        // Enforce constraints
                        if (newCenterWidth < MIN_CENTER) {
                            newCenterWidth = MIN_CENTER;
                            newRightWidth = remainingWidth - MIN_CENTER;
                            if (newRightWidth < MIN_RIGHT) {
                                newRightWidth = MIN_RIGHT;
                                newCenterWidth = remainingWidth - MIN_RIGHT;
                                newLeftWidth = 100 - newCenterWidth - newRightWidth;
                            }
                        }
                        
                        if (newRightWidth < MIN_RIGHT) {
                            newRightWidth = MIN_RIGHT;
                            newCenterWidth = remainingWidth - MIN_RIGHT;
                            if (newCenterWidth < MIN_CENTER) {
                                newCenterWidth = MIN_CENTER;
                                newLeftWidth = 100 - newCenterWidth - MIN_RIGHT;
                            }
                        }
                        
                        leftPanel.style.width = newLeftWidth + '%';
                        centerPanel.style.width = newCenterWidth + '%';
                        rightPanel.style.width = newRightWidth + '%';
                        
                    } else if (activeHandle === rightHandle) {
                        // Right handle: resize right, distribute to left and center proportionally
                        const deltaX = e.clientX - startX;
                        const deltaPercent = -(deltaX / containerWidth) * 100;
                        
                        let newRightWidth = clamp(
                            startWidths.right + deltaPercent,
                            MIN_RIGHT,
                            MAX_RIGHT
                        );
                        
                        // Remaining space for left + center
                        const remainingWidth = 100 - newRightWidth;
                        const currentOtherWidth = startWidths.left + startWidths.center;
                        const leftRatio = startWidths.left / currentOtherWidth;
                        const centerRatio = startWidths.center / currentOtherWidth;
                        
                        let newLeftWidth = remainingWidth * leftRatio;
                        let newCenterWidth = remainingWidth * centerRatio;
                        
                        // Enforce constraints
                        if (newLeftWidth < MIN_LEFT) {
                            newLeftWidth = MIN_LEFT;
                            newCenterWidth = remainingWidth - MIN_LEFT;
                            if (newCenterWidth < MIN_CENTER) {
                                newCenterWidth = MIN_CENTER;
                                newLeftWidth = remainingWidth - MIN_CENTER;
                                newRightWidth = 100 - newLeftWidth - newCenterWidth;
                            }
                        }
                        
                        if (newCenterWidth < MIN_CENTER) {
                            newCenterWidth = MIN_CENTER;
                            newLeftWidth = remainingWidth - MIN_CENTER;
                            if (newLeftWidth < MIN_LEFT) {
                                newLeftWidth = MIN_LEFT;
                                newRightWidth = 100 - MIN_LEFT - newCenterWidth;
                            }
                        }
                        
                        leftPanel.style.width = newLeftWidth + '%';
                        centerPanel.style.width = newCenterWidth + '%';
                        rightPanel.style.width = newRightWidth + '%';
                        
                    } else if (activeHandle === horizontalHandle) {
                        // Horizontal: resize latent vs loss
                        const deltaY = e.clientY - startY;
                        
                        // Get current center height (minus header and handle)
                        const availableHeight = centerPanel.offsetHeight - 70 - 5;
                        
                        let newLatentHeight = clamp(
                            startHeights.latent + deltaY,
                            MIN_LATENT,
                            availableHeight - MIN_LOSS
                        );
                        
                        let newLossHeight = availableHeight - newLatentHeight;
                        
                        latentContainer.style.height = newLatentHeight + 'px';
                        lossContainer.style.height = newLossHeight + 'px';
                    }
                    
                    // Trigger Plotly resize
                    window.dispatchEvent(new Event('resize'));
                }
                
                function stopResize() {
                    if (!isResizing) return;
                    isResizing = false;
                    activeHandle = null;
                    document.body.classList.remove('resizing');
                    document.body.style.cursor = '';
                }
                
                // Attach listeners
                leftHandle.addEventListener('mousedown', (e) => startResize(e, leftHandle));
                rightHandle.addEventListener('mousedown', (e) => startResize(e, rightHandle));
                horizontalHandle.addEventListener('mousedown', (e) => startResize(e, horizontalHandle));
                
                document.addEventListener('mousemove', doResize);
                document.addEventListener('mouseup', stopResize);
                
                window.__resizeHandlersInstalled = true;
                console.log('Smart resize handlers installed successfully');
                
            }, 100);
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('resize-setup-trigger', 'max_intervals'),
        Input('resize-setup-trigger', 'n_intervals'),
        prevent_initial_call=False,
    )

    register_training_callbacks(app)
    register_visualization_callbacks(app)
    register_labeling_callbacks(app)
    
    return app


app = create_app()


if __name__ == "__main__":
    initialize_model_and_data()
    app.run_server(debug=False, host="0.0.0.0", port=8050)