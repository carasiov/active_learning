# Command Pattern Architecture

## Overview

The SSVAE Active Learning Dashboard implements the Command Pattern for all state-modifying operations. This architectural decision centralizes business logic, provides complete audit trails, and eliminates scattered validation code across the codebase. Every user action—from labeling a sample to starting a training run—is encapsulated as a command object that validates, executes, and logs atomically under a single lock.

The migration from direct state mutations to command-based execution occurred after the initial immutable dataclass migration. While immutable state prevented accidental mutations, validation logic remained scattered across four callback files with no systematic way to track what changed, when, or why. The Command Pattern addresses these gaps by making every state transition explicit, testable, and auditable.

## Core Components

### Command Base Class

Every command inherits from the abstract `Command` class, which defines two core responsibilities: validation and execution. The `validate()` method checks whether the command can execute given the current state, returning an error message if invalid or `None` if valid. The `execute()` method performs the state transition, returning both the new state and a human-readable status message.

Commands are immutable dataclasses that capture all parameters needed for the operation. This design makes commands serializable and enables future features like command replay, undo/redo functionality, and event sourcing.

### Command Dispatcher

The `CommandDispatcher` serves as the single entry point for all state modifications. When a command is dispatched, the dispatcher acquires the state lock, validates the command against current state, executes it if valid, updates the global state atomically, and logs the result to command history. This ensures thread safety without requiring callbacks to manage locks manually.

The dispatcher maintains a separate history lock to prevent contention between state access and history logging. Command history is bounded at 1,000 entries with automatic pruning to 500 when the limit is reached, providing a debugging window without unbounded memory growth.

### Command History

Every command execution generates a `CommandHistoryEntry` containing the command instance, timestamp, success status, and result message. This audit trail enables debugging ("what commands led to this state?"), performance analysis ("which commands are slowest?"), and future features like command replay or telemetry.

History entries are stored in memory and accessible via `dispatcher.get_history(limit)`. The history can be cleared programmatically for testing or long-running sessions.

## Implemented Commands

### LabelSampleCommand

Assigns or removes a label for a specific sample. Validates that the sample index is within bounds and that label values (if provided) are integers between 0 and 9. When executed, the command updates the labels array, persists changes to the CSV file, regenerates hover metadata for the affected sample, and increments the data version counter to trigger UI reactivity.

The command reuses existing CSV persistence helpers (`_load_labels_dataframe` and `_persist_labels_dataframe`) to maintain compatibility with CLI tools that share the same labels file. This ensures the dashboard and command-line workflows remain interoperable without format migrations.

### StartTrainingCommand

Queues a training run with specified hyperparameters. Validation checks that no training is currently active, that epoch count is between 1 and 200, that at least one labeled sample exists, and that hyperparameters fall within valid ranges (non-negative reconstruction weight, KL weight ≤ 10, learning rate ≤ 0.1).

Execution updates the mutable config objects (configs remain mutable as an implementation detail), synchronizes settings across the model, trainer, and dashboard config instances, and transitions the training state machine to QUEUED. The command does not start the background thread—that responsibility remains in the callback layer to maintain separation between business logic and execution orchestration.

### CompleteTrainingCommand

Marks training as complete and updates the model's predictions. The background training worker calls this command when training finishes, passing the new latent representations, reconstructions, predicted classes, certainties, and updated hover metadata.

Validation ensures array shapes match the expected dataset size, preventing shape mismatches from corrupting state. Execution delegates to the existing `AppState.with_training_complete()` method, which atomically updates data, increments the version counter, and transitions training state to COMPLETE.

### SelectSampleCommand

Updates the currently selected sample in the UI. Validation checks that the sample index is within dataset bounds. Execution updates the UI state via `AppState.with_ui(selected_sample=idx)`, triggering reactivity in callbacks that depend on the selected sample (image display, label buttons, prediction info).

### ChangeColorModeCommand

Switches the latent space scatter plot coloring mode. Validation restricts color modes to the predefined set: `user_labels`, `pred_class`, `true_class`, and `certainty`. Execution updates UI state, triggering the visualization callback to recompute colors and redraw the scatter plot.

## Integration with Callbacks

Callbacks serve as thin adapters between Dash component events and command execution. A typical callback pattern extracts parameters from component inputs, constructs the appropriate command, dispatches it via `dashboard_state.dispatcher.execute(command)`, and returns the result to update component outputs.

This separation enables testing business logic without Dash infrastructure. Commands can be validated and executed in pure Python tests, while callback integration tests verify that user interactions trigger the correct commands.

### Example: Labeling Callback

```python
def handle_label_actions(label_clicks, delete_clicks, selected_idx):
    if selected_idx is None:
        raise PreventUpdate
    
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    triggered_id = ctx.triggered_id
    
    # Determine command based on trigger
    if triggered_id == "delete-label-button":
        command = LabelSampleCommand(sample_idx=int(selected_idx), label=None)
    elif isinstance(triggered_id, dict) and "label" in triggered_id:
        label_value = int(triggered_id["label"])
        command = LabelSampleCommand(sample_idx=int(selected_idx), label=label_value)
    else:
        raise PreventUpdate
    
    # Execute command
    success, message = dashboard_state.dispatcher.execute(command)
    
    # Return updated version
    with dashboard_state.state_lock:
        version_payload = {"version": dashboard_state.app_state.data.version}
    
    return version_payload, message
```

The callback contains no validation logic (delegated to the command), no state mutation logic (handled by the command), and no CSV persistence logic (encapsulated in the command). It simply maps UI events to commands and returns results to Dash.

## Thread Safety Guarantees

The Command Pattern integrates seamlessly with the existing immutable state architecture. The dispatcher acquires `state_lock` before validating or executing any command, ensuring that validation sees consistent state and that state updates are atomic. Callbacks that execute commands do not need to manage locks manually—the dispatcher handles synchronization automatically.

The separate `_history_lock` prevents command logging from blocking state access. A callback can dispatch a command and return while the dispatcher logs the result to history in parallel, reducing lock contention on the critical path.

Background threads (like the training worker) dispatch commands the same way callbacks do. The `CompleteTrainingCommand` is dispatched from the background training thread, and the dispatcher ensures atomic state updates without race conditions.

## Validation Philosophy

Validation occurs inside commands rather than in callbacks for several reasons. First, centralizing validation in one place eliminates duplication—multiple callbacks that label samples all reuse `LabelSampleCommand.validate()` logic. Second, validation becomes testable without UI dependencies—tests can instantiate commands and verify validation behavior directly. Third, validation sees consistent state because it executes under the state lock, preventing TOCTOU (time-of-check-time-of-use) race conditions.

Commands validate parameters (type, range, format) and state preconditions (sample exists, training not active, has labeled samples). Validation returns `None` on success or a human-readable error message on failure. The dispatcher short-circuits execution if validation fails, logging the failure without modifying state.

## Performance Characteristics

Command dispatch overhead is negligible compared to the operations being performed (CSV I/O, model inference, UI updates). Profiling shows command validation and execution typically complete in microseconds, while CSV persistence takes milliseconds and training takes minutes.

The dispatcher's sequential execution model (one command at a time under lock) is appropriate for this single-user localhost application. Lock contention is minimal because critical sections are short—commands update state and return quickly. Long-running operations (training, model prediction) occur outside the lock, with only the final state update protected.

Command history grows linearly with user actions. At 1,000 entries per session and typical usage patterns (dozens of labels, few training runs), memory overhead remains negligible (~100KB). Automatic pruning to 500 entries ensures bounded growth for long-running sessions.

## Future Extensions

The Command Pattern provides a foundation for several advanced features:

**Undo/Redo**: Commands already capture all information needed to reverse operations. A `LabelSampleCommand` could provide an `undo()` method that restores the previous label. The dispatcher could maintain an undo stack and redo stack, enabling users to revert accidental changes.

**Event Sourcing**: Command history is effectively an event log. Persisting this log to disk would enable replaying dashboard sessions, reproducing bugs, or analyzing user behavior. The entire application state could be reconstructed by replaying commands from an empty initial state.

**Optimistic Locking**: Commands could include version numbers to detect concurrent modifications. If two users (in a hypothetical multi-user deployment) try to label the same sample, the second command would fail validation due to a version mismatch.

**Remote Execution**: Commands are serializable dataclasses. A client-server architecture could serialize commands to JSON, transmit them over the network, deserialize on the server, and execute remotely. Responses would flow back the same way.

**Telemetry and Analytics**: Command history provides rich data about how users interact with the dashboard. Metrics like "most common commands," "average time between labels," or "training success rate" could be computed from logged commands.

## Migration Impact

The Command Pattern migration touched four callback files and eliminated approximately 120 lines of duplicated validation and state mutation logic. Callback functions became significantly shorter (70% reduction in some cases) and easier to understand—each callback is now a thin mapping from UI events to commands.

The `_update_label()` helper function (47 lines) was deleted entirely, with its logic absorbed into `LabelSampleCommand`. This eliminated a layer of indirection and made the labeling workflow easier to trace. CSV persistence helpers remain as standalone functions that commands call, maintaining code reuse without tight coupling.

No changes were required to existing state models, training infrastructure, or UI layouts. The Command Pattern is purely additive—it adds structure to how state changes occur without altering the state schema itself.

## Testing Strategy

Commands are designed for testability. Unit tests can instantiate commands with various parameters, call `validate()` with different state fixtures, and verify that validation catches errors correctly. Tests can call `execute()` and assert on the returned state without any Dash infrastructure.

Example test structure:

```python
def test_label_sample_validates_bounds():
    state = create_test_state(num_samples=100)
    cmd = LabelSampleCommand(sample_idx=150, label=5)
    error = cmd.validate(state)
    assert "Invalid sample index" in error

def test_label_sample_validates_range():
    state = create_test_state(num_samples=100)
    cmd = LabelSampleCommand(sample_idx=50, label=15)
    error = cmd.validate(state)
    assert "Label must be 0-9" in error

def test_label_sample_executes():
    state = create_test_state(num_samples=100)
    cmd = LabelSampleCommand(sample_idx=50, label=7)
    new_state, message = cmd.execute(state)
    assert new_state.data.labels[50] == 7.0
    assert "Labeled sample 50 as 7" in message
```

Integration tests verify that dispatching commands from callbacks produces the expected UI updates, leveraging Dash's testing utilities to simulate button clicks and keyboard input.

## Conclusion

The Command Pattern transforms scattered imperative state mutations into a structured, auditable, and testable architecture. Every action in the dashboard is now explicit (a command class), validated (centralized validation logic), atomic (executed under lock), and logged (command history). This foundation supports the current single-user localhost deployment while enabling future enhancements like undo/redo, event sourcing, and telemetry.

The pattern's primary tradeoff is additional abstraction—instead of mutating state directly, developers create command classes. This upfront cost pays dividends in maintainability, debuggability, and extensibility. As the dashboard evolves, adding new state-modifying actions becomes a matter of implementing a new command rather than threading validation and mutation logic through multiple callback functions.
