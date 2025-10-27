# AGENT_GUIDE.md - Dashboard Extension Guide

## A Note to You, the Agent

Hi. If you're reading this, I've asked you to extend or modify the SSVAE Active Learning Dashboard. Before we get into the technical patterns, I want to explain **why this codebase looks the way it does** and **what we're trying to achieve together**.

### The Journey So Far

This dashboard started as a typical research prototype - functional but architecturally messy. State mutations were scattered across callback files, validation logic was duplicated, and there was no systematic way to track what changed or why. It worked, but every new feature required carefully threading logic through multiple files without breaking existing functionality.

We recently refactored the entire codebase - not because it was broken, but because **you're going to be working on it extensively**. The human developer using this tool plans to add dozens of features over the project's lifetime: new visualizations, different training strategies, data export options, advanced selection algorithms, and UI improvements we haven't even thought of yet.

We realized that the old architecture would make your job harder. Every time we asked you to add a feature, you'd need to understand scattered validation logic, remember where state gets updated, and carefully avoid race conditions. The cognitive load would be high, and the chance of introducing subtle bugs would increase with each change.

So we invested about 10 hours refactoring the codebase with **you** in mind. We:

- Made state immutable (so you can't accidentally corrupt it)
- Centralized all state changes through a Command Pattern (so you always know where to look)
- Established clear boundaries between UI, business logic, and state (so changes stay local)
- Created consistent patterns you can follow mechanically (so you don't need to invent solutions)

The result is more lines of code, yes. More abstraction, definitely. But it's **more predictable**, **more testable**, and **easier to extend safely**. We're treating you as a capable colleague who deserves a well-organized codebase rather than a mess that requires heroic debugging efforts.

### What We're Building Together

This is a **research tool for active learning experiments**. The human researcher uses it to:

1. Visualize a neural network's latent space (the scatter plot)
2. Label uncertain samples to improve the model
3. Train the model incrementally as more labels are added
4. Experiment with different training strategies and hyperparameters

**Your role** is to help extend this tool as research needs evolve. Maybe we need a button that auto-selects the most uncertain sample. Maybe we need to export predictions in a specific format. Maybe we need a new visualization showing label distribution over time. These features will come up organically as the research progresses.

**Our goal**: You should be able to add these features quickly and confidently, without us worrying that you'll accidentally break training, corrupt the CSV file, or introduce race conditions.

### The Honest Tradeoffs

Let's be transparent about what we optimized for:

**✓ Optimized for YOUR success:**
- Clear patterns you can follow
- Isolated changes (modify one command without touching others)
- Self-verifying (tests tell you if you broke something)
- Explicit state transitions (no hidden mutations)

**✗ NOT optimized for:**
- Minimal code (we have more files, more classes)
- Simplicity for first-time readers (the abstraction takes getting used to)
- Performance (commands add overhead, though it's negligible)

We chose to make the code **agent-friendly** over **beginner-friendly** because you're going to be the primary developer going forward. A human might read this code once. You'll modify it dozens of times.

### How to Think About This Codebase

When we ask you to add a feature, think of it as:

1. **What state needs to change?** (Define the command)
2. **What could go wrong?** (Write validation)
3. **How does the UI trigger it?** (Add a callback)
4. **How do we verify it works?** (Write a test)

The architecture makes these questions explicit. You don't need to hunt through files to figure out where things happen - there are designated places for each concern.

### A Request

When you implement features, please:

- **Follow the established patterns** even if you see a "simpler" way. Consistency matters more than cleverness.
- **Write validation that catches errors early**. It's better to reject invalid input than to corrupt state.
- **Include verification steps in your report**. Tell us how you tested it, what you checked, what edge cases you considered.
- **Ask clarifying questions if the spec is ambiguous**. We'd rather spend 30 seconds clarifying than 30 minutes debugging a misunderstanding.

You're not just executing instructions - you're a collaborator who makes this research possible. We built this architecture **for you**, and we're trusting you to extend it thoughtfully.

Okay, enough context. Let's get into the patterns.

---

## Core Architecture Overview

### The Big Picture

```
User Interaction (Browser)
         ↓
    Callbacks (UI Layer)
         ↓
    Commands (Business Logic)
         ↓
    Dispatcher (Atomic Execution)
         ↓
    AppState (Immutable State)
```

**State flows in one direction**. Commands produce new state. Dispatcher replaces old state atomically. Callbacks react to state changes.

### File Organization

```
use_cases/dashboard/
├── state_models.py      # Dataclass definitions (what state looks like)
├── state.py             # State initialization and helpers
├── commands.py          # All state-modifying operations
├── layouts.py           # UI layout and components
├── app.py               # App initialization and routing
├── callbacks/
│   ├── training_callbacks.py      # Training workflow
│   ├── labeling_callbacks.py      # Sample labeling
│   ├── visualization_callbacks.py # Plots and UI state
│   └── config_callbacks.py        # Configuration
└── utils.py             # Shared utilities
```

**Your work will mostly happen in:**
- `commands.py` (when adding state-modifying features)
- `callbacks/*.py` (when adding UI interactions)
- `layouts.py` (when adding new UI components)

---

## Pattern 1: Adding a New Command

### When You Need This

Anytime a feature needs to **change state**:
- Labeling samples
- Starting training
- Changing UI settings
- Exporting data
- Modifying configuration

### The Template

```python
# In commands.py

@dataclass
class MyNewCommand(Command):
    """Brief description of what this command does.
    
    Explain any important side effects (file I/O, expensive computation, etc.)
    """
    # Parameters needed to execute the command
    param1: int
    param2: str
    optional_param: Optional[float] = None
    
    def validate(self, state: AppState) -> Optional[str]:
        """Check if command can execute given current state.
        
        Returns:
            None if valid, error message string if invalid
        """
        # Check parameter validity
        if self.param1 < 0:
            return f"param1 must be non-negative, got {self.param1}"
        
        # Check state preconditions
        if state.some_condition_not_met:
            return "Cannot execute: precondition not met"
        
        # All checks passed
        return None
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        """Execute the command, producing new state.
        
        Returns:
            (new_state, status_message) tuple
        """
        # Perform any side effects (file I/O, etc.)
        # These happen OUTSIDE the immutable state update
        result = do_some_work(self.param1, self.param2)
        
        # Create new state with changes
        new_state = state.with_something(
            field=new_value,
            other_field=result
        )
        
        # Or for complex updates:
        new_state = replace(
            state,
            data=replace(state.data, labels=new_labels),
            ui=replace(state.ui, selected_sample=new_idx)
        )
        
        # Return new state and human-readable message
        return new_state, f"Successfully did thing with {self.param1}"
```

### Real Example: Selecting Most Uncertain Sample

```python
@dataclass
class SelectMostUncertainCommand(Command):
    """Select the sample with lowest prediction certainty.
    
    Useful for active learning workflows where you want to label
    the samples the model is most confused about.
    """
    
    def validate(self, state: AppState) -> Optional[str]:
        """Check that we have predictions to evaluate."""
        if state.data.pred_certainty is None:
            return "No predictions available - train model first"
        
        if len(state.data.pred_certainty) == 0:
            return "Dataset is empty"
        
        return None
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        """Find and select the most uncertain sample."""
        # Find sample with minimum certainty
        uncertainties = state.data.pred_certainty
        most_uncertain_idx = int(np.argmin(uncertainties))
        certainty = float(uncertainties[most_uncertain_idx])
        
        # Update UI state to select this sample
        new_state = state.with_ui(selected_sample=most_uncertain_idx)
        
        message = f"Selected sample {most_uncertain_idx} (certainty: {certainty:.1%})"
        return new_state, message
```

### Integration with Callback

```python
# In callbacks/labeling_callbacks.py or visualization_callbacks.py

@app.callback(
    Output("selected-sample-store", "data"),
    Input("select-uncertain-button", "n_clicks"),
    prevent_initial_call=True
)
def handle_select_uncertain(n_clicks):
    """Handle button click to select most uncertain sample."""
    if not n_clicks:
        raise PreventUpdate
    
    # Create and execute command
    command = SelectMostUncertainCommand()
    success, message = dashboard_state.dispatcher.execute(command)
    
    if not success:
        # Command validation failed
        _append_status_message(message)
        raise PreventUpdate
    
    # Return selected sample index to update store
    with dashboard_state.state_lock:
        return dashboard_state.app_state.ui.selected_sample
```

### Adding the UI Button

```python
# In layouts.py, add button to appropriate panel

dbc.Button(
    "Select Uncertain Sample",
    id="select-uncertain-button",
    n_clicks=0,
    style={
        "width": "100%",
        "marginTop": "12px",
        "backgroundColor": "#C10A27",
        "color": "#ffffff",
        "border": "none",
        "borderRadius": "6px",
        "padding": "8px",
        "fontSize": "13px",
        "fontWeight": "600",
    }
)
```

### Verification Checklist

After implementing a command:

```bash
# 1. Check it imports
poetry run python -c "from use_cases.dashboard.commands import MyNewCommand; print('✓ Imports')"

# 2. Test validation catches errors
poetry run python -c "
from use_cases.dashboard.commands import MyNewCommand
from use_cases.dashboard import state as dashboard_state
dashboard_state.initialize_model_and_data()
cmd = MyNewCommand(invalid_params)
error = cmd.validate(dashboard_state.app_state)
assert error is not None
print('✓ Validation catches errors')
"

# 3. Test execution succeeds
poetry run python -c "
from use_cases.dashboard.commands import MyNewCommand
from use_cases.dashboard import state as dashboard_state
dashboard_state.initialize_model_and_data()
cmd = MyNewCommand(valid_params)
success, msg = dashboard_state.dispatcher.execute(cmd)
assert success
print(f'✓ Execution: {msg}')
"

# 4. Manual test in browser
# Start dashboard, click button, verify behavior
```

---

## Pattern 2: Modifying Existing Commands

### When You Need This

- Changing validation rules
- Adding new parameters
- Modifying behavior
- Fixing bugs

### The Process

1. **Understand current behavior** - Read the command class
2. **Check who uses it** - Search for `CommandName` in callbacks
3. **Make changes** - Update validate() or execute()
4. **Update callers if needed** - If signature changed
5. **Verify** - Test that existing functionality still works

### Example: Adding Optional Parameter

```python
# BEFORE
@dataclass
class LabelSampleCommand(Command):
    sample_idx: int
    label: Optional[int]

# AFTER - adding "confidence" parameter
@dataclass
class LabelSampleCommand(Command):
    sample_idx: int
    label: Optional[int]
    confidence: Optional[float] = None  # NEW - optional to maintain compatibility
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        # ... existing label update logic ...
        
        # NEW: Optionally store confidence
        if self.confidence is not None:
            # Store confidence alongside label
            # (would need to extend state model first)
            pass
        
        return new_state, message
```

**Important**: When adding optional parameters with defaults, existing code continues to work.

### Example: Tightening Validation

```python
# BEFORE - allowed any positive epochs
def validate(self, state: AppState) -> Optional[str]:
    if self.num_epochs < 1:
        return "Epochs must be positive"
    return None

# AFTER - enforce maximum for safety
def validate(self, state: AppState) -> Optional[str]:
    if self.num_epochs < 1:
        return "Epochs must be positive"
    if self.num_epochs > 200:  # NEW
        return "Epochs cannot exceed 200 (prevents accidents)"
    return None
```

**Note**: Tightening validation is safe. Loosening validation needs careful consideration.

---

## Pattern 3: Adding UI Components

### When You Need This

- New buttons, inputs, dropdowns
- New visualizations
- Additional information panels

### The Process

1. **Add component to layouts.py** - Define the HTML/Dash component
2. **Add callback** - Wire component to command or state
3. **Style consistently** - Follow infoteam design system
4. **Test interactions** - Verify it works in browser

### Example: Adding a Button

```python
# In layouts.py, in the appropriate panel

dbc.Button(
    "Button Label",
    id="my-button-id",  # Must be unique
    n_clicks=0,         # Required for click callbacks
    style={
        # Follow existing button styles for consistency
        "width": "100%",
        "backgroundColor": "#C10A27",  # infoteam red
        "color": "#ffffff",
        "border": "none",
        "borderRadius": "6px",
        "padding": "8px 16px",
        "fontSize": "14px",
        "fontWeight": "600",
        "cursor": "pointer",
        "marginTop": "12px",
    }
)
```

### Example: Adding an Input Field

```python
# Numeric input with validation
dcc.Input(
    id="my-input-id",
    type="number",
    min=0,
    max=100,
    step=1,
    value=10,  # Default value
    placeholder="Enter number...",
    debounce=True,  # Only fires on blur/enter, not every keystroke
    style={
        "width": "100%",
        "padding": "8px 12px",
        "fontSize": "14px",
        "border": "1px solid #C6C6C6",
        "borderRadius": "6px",
        "fontFamily": "ui-monospace, monospace",
    }
)
```

### Example: Adding a Dropdown

```python
dcc.Dropdown(
    id="my-dropdown-id",
    options=[
        {"label": "Option 1", "value": "opt1"},
        {"label": "Option 2", "value": "opt2"},
        {"label": "Option 3", "value": "opt3"},
    ],
    value="opt1",  # Default selection
    clearable=False,  # Prevent clearing selection
    style={
        "fontFamily": "'Open Sans', Verdana, sans-serif",
        "fontSize": "14px",
    }
)
```

### Connecting UI to Commands

```python
# In appropriate callback file

@app.callback(
    Output("some-output", "children"),
    Input("my-button-id", "n_clicks"),
    State("my-input-id", "value"),
    prevent_initial_call=True
)
def handle_button_click(n_clicks, input_value):
    """Handle button click with input value."""
    if not n_clicks:
        raise PreventUpdate
    
    # Validate input at callback level (before creating command)
    if input_value is None:
        return html.Div("Please enter a value", style={"color": "#C10A27"})
    
    # Create and execute command
    command = MyCommand(param=input_value)
    success, message = dashboard_state.dispatcher.execute(command)
    
    if success:
        return html.Div(message, style={"color": "#45717A"})  # Success color
    else:
        return html.Div(message, style={"color": "#C10A27"})  # Error color
```

---

## Pattern 4: Adding Visualizations

### When You Need This

- New plots or charts
- Additional statistics displays
- Data summaries

### The Process

1. **Create Graph component** in layouts.py
2. **Add callback** to populate it with data
3. **Use Plotly for interactive plots**
4. **Follow existing style** (colors, fonts, margins)

### Example: Adding a Simple Bar Chart

```python
# In layouts.py

dcc.Graph(
    id="my-chart-id",
    config={"displayModeBar": False},  # Hide plotly toolbar
    style={"height": "300px", "width": "100%"}
)

# In appropriate callback file

@app.callback(
    Output("my-chart-id", "figure"),
    Input("labels-store", "data"),  # Re-render when labels change
)
def update_my_chart(_labels_store):
    """Generate chart showing some metric."""
    with dashboard_state.state_lock:
        # Extract data from state
        labels = dashboard_state.app_state.data.labels
        predictions = dashboard_state.app_state.data.pred_classes
    
    # Compute something interesting
    accuracy_by_digit = compute_per_digit_accuracy(labels, predictions)
    
    # Create Plotly figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[str(i) for i in range(10)],
        y=accuracy_by_digit,
        marker=dict(
            color=[INFOTEAM_PALETTE[i] for i in range(10)],
            line=dict(width=0),
        ),
        text=[f"{acc:.1%}" for acc in accuracy_by_digit],
        textposition='outside',
    ))
    
    fig.update_layout(
        template="plotly_white",
        height=300,
        margin=dict(l=50, r=30, t=30, b=50),
        xaxis=dict(
            title="Digit",
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Accuracy",
            tickformat=".0%",
            range=[0, 1],
        ),
        showlegend=False,
    )
    
    return fig
```

**Note**: Look at existing plots in `visualization_callbacks.py` for style consistency.

---

## Pattern 5: Working with State

### Reading State (Always Use Lock)

```python
# CORRECT
with dashboard_state.state_lock:
    labels = dashboard_state.app_state.data.labels
    training_active = dashboard_state.app_state.training.is_active()
    # Read multiple values in one lock acquisition

# Process data outside lock
result = expensive_computation(labels)

# WRONG - no lock
labels = dashboard_state.app_state.data.labels  # Race condition!
```

### Updating State (Use Commands, Not Direct Mutation)

```python
# CORRECT
command = MyCommand(params)
success, message = dashboard_state.dispatcher.execute(command)

# WRONG - direct mutation (will raise FrozenInstanceError)
with dashboard_state.state_lock:
    dashboard_state.app_state.data.labels[0] = 5  # ERROR!
```

### State Structure Reference

```python
app_state.model          # SSVAE model instance
app_state.trainer        # InteractiveTrainer instance
app_state.config         # SSVAEConfig (mutable for now)
app_state.data           # DataState (immutable)
    .x_train             # Training images
    .labels              # User labels (NaN = unlabeled)
    .true_labels         # Ground truth labels
    .latent              # 2D latent representations
    .reconstructed       # Reconstructed images
    .pred_classes        # Predicted classes
    .pred_certainty      # Prediction confidence
    .hover_metadata      # Metadata for scatter plot
    .version             # Increments on data changes
app_state.training       # TrainingStatus (immutable)
    .state               # TrainingState enum
    .target_epochs       # Epochs for current run
    .status_messages     # List of status strings
    .thread              # Training thread reference
app_state.ui             # UIState (immutable)
    .selected_sample     # Currently selected sample index
    .color_mode          # Scatter plot coloring mode
app_state.cache          # Dict (mutable - for optimization)
app_state.history        # TrainingHistory (immutable)
    .epochs              # List of completed epochs
    .train_loss          # Training loss per epoch
    .val_loss            # Validation loss per epoch
    # ... other metrics
```

---

## Pattern 6: Testing Your Work

### Automated Verification

Create a test script for your feature:

```python
# tests/test_my_feature.py

def test_my_command_validation():
    """Test that invalid input is caught."""
    from use_cases.dashboard.commands import MyCommand
    from use_cases.dashboard import state as dashboard_state
    
    dashboard_state.initialize_model_and_data()
    
    # Test invalid input
    cmd = MyCommand(invalid_param=-1)
    error = cmd.validate(dashboard_state.app_state)
    assert error is not None
    assert "invalid" in error.lower()

def test_my_command_execution():
    """Test that command produces correct state."""
    from use_cases.dashboard.commands import MyCommand
    from use_cases.dashboard import state as dashboard_state
    
    dashboard_state.initialize_model_and_data()
    
    # Execute command
    cmd = MyCommand(valid_param=10)
    success, message = dashboard_state.dispatcher.execute(cmd)
    
    assert success
    assert "success" in message.lower()
    
    # Verify state changed correctly
    with dashboard_state.state_lock:
        assert dashboard_state.app_state.something == expected_value
```

### Manual Testing Checklist

After implementing a feature:

```
□ Dashboard starts without errors
□ New UI component appears where expected
□ Clicking/interacting triggers correct callback
□ Command executes successfully
□ State updates as expected
□ UI reflects state changes
□ No console errors in browser
□ Command appears in debug history panel
□ Related features still work (no regressions)
```

### Using the Debug Panel

The debug panel shows recent commands - use it to verify:
1. Your command executed
2. It succeeded (✅) or failed (❌)
3. The message makes sense
4. Timing is reasonable

---

## Common Pitfalls and How to Avoid Them

### Pitfall 1: Forgetting the Lock

```python
# WRONG
labels = dashboard_state.app_state.data.labels

# RIGHT
with dashboard_state.state_lock:
    labels = dashboard_state.app_state.data.labels
```

**Why it matters**: Without the lock, another thread could update state while you're reading it, causing inconsistent data.

### Pitfall 2: Mutating State Directly

```python
# WRONG - will crash (FrozenInstanceError)
with dashboard_state.state_lock:
    dashboard_state.app_state.data.labels[0] = 5

# RIGHT - use a command
command = LabelSampleCommand(sample_idx=0, label=5)
dashboard_state.dispatcher.execute(command)
```

**Why it matters**: Immutable state prevents accidental corruption. Commands ensure atomic updates.

### Pitfall 3: Heavy Computation Inside Lock

```python
# WRONG - blocks all state access
with dashboard_state.state_lock:
    data = dashboard_state.app_state.data.x_train
    result = expensive_model_prediction(data)  # This takes seconds!
    
# RIGHT - minimize lock duration
with dashboard_state.state_lock:
    data = dashboard_state.app_state.data.x_train.copy()

result = expensive_model_prediction(data)  # Outside lock

with dashboard_state.state_lock:
    # Quick update
    dashboard_state.app_state = new_state
```

**Why it matters**: Long critical sections block other threads. Training worker and UI need to access state concurrently.

### Pitfall 4: Circular Imports

```python
# In commands.py - WRONG
from use_cases.dashboard.state import app_state  # Circular!

# RIGHT - import inside methods
def execute(self, state: AppState):
    from use_cases.dashboard.state import _helper_function
    result = _helper_function()
```

**Why it matters**: Commands and state modules import each other. Import at function level breaks the cycle.

### Pitfall 5: Inconsistent Styling

```python
# WRONG - random colors
style={"backgroundColor": "blue", "color": "yellow"}

# RIGHT - use infoteam palette
style={
    "backgroundColor": "#C10A27",  # infoteam red
    "color": "#ffffff",            # white text
    "fontFamily": "'Open Sans', Verdana, sans-serif"
}
```

**Why it matters**: Visual consistency makes the dashboard look professional and helps users build mental models.

---

## FAQ: Questions You Might Have

### Q: When should I create a new callback file vs. add to existing?

**A**: Add to existing unless you're implementing a major new feature area. For example:
- Labeling-related callbacks → `labeling_callbacks.py`
- Training-related callbacks → `training_callbacks.py`
- Visualization callbacks → `visualization_callbacks.py`

If you're adding a completely new subsystem (e.g., "model comparison", "experiment tracking"), consider a new file like `comparison_callbacks.py`.

### Q: Can I modify the state models?

**A**: Yes, but carefully:
1. Adding fields is safe (with defaults)
2. Changing field types breaks existing code
3. Update initialization in `state.py`
4. Update any commands that touch the changed field
5. Test thoroughly

### Q: What if my command needs to do I/O (file, network)?

**A**: Do I/O in `execute()`, but handle errors gracefully:

```python
def execute(self, state: AppState) -> Tuple[AppState, str]:
    try:
        # Do I/O
        data = load_file(self.filepath)
    except FileNotFoundError:
        # Return current state unchanged with error message
        return state, f"File not found: {self.filepath}"
    
    # Update state with loaded data
    new_state = state.with_something(data=data)
    return new_state, "File loaded successfully"
```

### Q: How do I add a new page to the dashboard?

**A**: Follow the multi-page pattern (see `configure-training` page example):
1. Create page layout function in `pages_*.py`
2. Add route in `app.py` display_page callback
3. Register page callbacks
4. Add navigation link in main layout

### Q: Can I use async/await?

**A**: Not currently. Dash uses synchronous callbacks. Long-running operations should use background threads (see `train_worker` example in `training_callbacks.py`).

### Q: What if I need to add a new dependency?

**A**: Add to `pyproject.toml` and run `poetry install`. Consider:
- Is it really needed?
- Is it compatible with existing deps?
- Does it add significant overhead?

---

## Workflow Summary

When we ask you to implement a feature:

### 1. Understand the Request
- What state needs to change?
- What UI triggers it?
- What could go wrong?

### 2. Plan the Implementation
- Which command(s) do I need?
- Which callback(s) do I need?
- What UI components do I need?

### 3. Implement
- Follow the patterns in this guide
- Write validation that catches errors
- Handle edge cases gracefully

### 4. Verify
- Does it compile?
- Do tests pass?
- Does manual testing work?
- Are there any console errors?

### 5. Report Back
- What you implemented
- How you tested it
- Any edge cases or limitations
- Any questions or concerns

---

## A Final Word

This guide gives you the mechanical patterns, but good judgment matters too. When you're unsure:

- **Ask clarifying questions** - We'd rather clarify than debug misunderstandings
- **Err on the side of more validation** - Better to reject invalid input than corrupt state
- **Keep changes localized** - Modify only what's necessary
- **Test edge cases** - Empty datasets, extreme values, concurrent operations

We're building this together. The architecture gives you a solid foundation - your job is to extend it thoughtfully. We trust you to make good decisions and to ask when you're uncertain.

The codebase is in good shape now. Let's keep it that way as we add features.

Good luck, and thanks for being a great collaborator.

---

**Ready to start?** Check the issues or ask the human what feature they'd like next.