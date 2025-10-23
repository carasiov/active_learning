# **SSVAE Dashboard - Product Requirements**

**Version:** 2.0  
**Date:** January 2025  
**Status:** In Development  
**Project:** Interactive Dashboard for Active Learning with SSVAE

---

## 1. What We're Building

We're building a web-based dashboard that makes active learning with our SSVAE model actually enjoyable to use. Right now, the workflow involves running separate scripts, manually tracking labels in a CSV, switching between a matplotlib viewer and the command line, and generally feeling disconnected. 

The goal is simple: one cohesive interface where you can see your data, label uncertain samples, tweak model settings, kick off training, and watch it all happen in real-time. No more context switching, no more terminal gymnastics.

### Who Is This For?

This is for ML researchers and practitioners (like us) doing iterative experiments. We're comfortable with Python and ML concepts, we value speed and efficiency over hand-holding, and we work alone on localhost for hours at a time. We need to label hundreds of samples and train dozens of times per session. This isn't for production deployment, multiple users, or public internet access - it's a tool for focused research work.

### Success Looks Like

We'll know this works when we can complete a full label-train-evaluate cycle without touching the terminal. When labeling 50 samples feels fast and fluid, not tedious. When training happens in the background while we keep exploring. When we actually choose to use this instead of the old CLI workflow. When the dashboard feels responsive and natural, not laggy or frustrating.

---

## 2. Current State

### What's Working

We've completed the initial implementation (Phase 1 & 2), and the core functionality is there. We have a web dashboard that loads and visualizes the latent space with all 60,000 MNIST samples. You can click any point to see the original and reconstructed images. You can label samples with buttons (0-9), and it immediately saves to CSV. You can configure training parameters like loss weights, learning rate, and number of epochs. You can start training in the background and see live progress updates. When training completes, the latent space automatically refreshes.

All the pieces are in place. The workflow exists end-to-end.

### The Problems

But there are real issues making it frustrating to use. The performance is rough - clicking around feels laggy, there's lots of unnecessary refreshing happening in the background, and the whole experience feels heavier than it should. The code has become unwieldy too, with everything living in one giant file over 1000 lines long, making it hard to find specific logic or make changes confidently. And we're missing the visual feedback that makes training feel tangible - no loss curves, no clear sense of progress beyond text updates.

These aren't fundamental design problems. The architecture is sound. We just need to optimize what we have, organize it better, and add the finishing touches that make it feel complete.

### What We're Keeping

We're intentionally not touching the core SSVAE implementation. The model code, training loops, loss functions, and callback system all stay as they are. The existing CLI scripts (`train.py`, `infer.py`, `view_latent.py`) must continue to work - we're adding an interface, not replacing the infrastructure. The labels CSV format and checkpoint format need to stay compatible so the dashboard and CLI can work together interchangeably.

---

## 3. The User Experience We Want

### The Core Workflow

Active learning is inherently iterative. You explore the latent space, looking for clusters, outliers, or uncertain regions. You label interesting points by clicking them and assigning labels. You might adjust loss weights or learning rate based on what you're seeing, though often the defaults work fine. You kick off training, watch the progress, and wait for completion. Then you evaluate the updated latent space, checking if clusters improved or if the model learned what you hoped. And you repeat this cycle, each time becoming more strategic about which samples to label.

A typical session might look like this: open the dashboard and see the current model state, spend 10-15 minutes labeling 20-50 samples, train for 5-10 minutes on 10-50 epochs, spend 2-3 minutes evaluating the results, and then repeat this cycle 3-5 times over the course of an hour or two.

### Key User Stories

When we open the dashboard, we want to immediately see our latent space without having to run inference scripts manually. The dashboard should load our existing checkpoint if we have one, automatically run inference on all the samples, and present the scatter plot ready to explore. This should take around half a minute from launching the command to having an interactive visualization.

As we explore, we need the 60,000-point scatter plot to render smoothly without lag. We should be able to zoom, pan, and hover around naturally. When we click any point, it should be selected immediately, and we should see the original and reconstructed images for that sample right away. We want to be able to switch between different color schemes - sometimes we want to see our manual labels, sometimes the model's predictions, sometimes the prediction certainty, sometimes the ground truth for reference.

For labeling, speed and fluidity matter enormously. The interaction should be: click a point, click a digit button, done. No page reload, no confirmation dialog, no waiting. If we're in "user labels" color mode, we should immediately see that point change color. The CSV should update behind the scenes without us thinking about it. If we label rapidly, clicking through dozens of samples, it shouldn't get confused or corrupt the data. And if we make a mistake, we should be able to delete a label just as easily.

Training is where background execution becomes critical. We want to click "Start Training" and have training begin immediately while the dashboard stays completely responsive. We should be able to browse other points, even label more samples, while training runs. We want to see live updates - what epoch we're on, what the current losses are - but with a latency of a second or two, which is totally fine since epochs themselves take many seconds. When training completes, the latent space should refresh automatically, and we should be ready to immediately start another training session if we want.

We'd like to be able to adjust hyperparameters between training sessions. The sliders and inputs for reconstruction weight, KL weight, and learning rate should be there when we want them, but they shouldn't get in the way. Changes should apply to the next training session, and the optimizer state should be preserved so we can train for 10 epochs, evaluate, then train 20 more as a continuation rather than starting from scratch.

Finally, we want to understand what's happening during training. Live loss curves showing how total loss and component losses are evolving. Clear visibility into training and validation metrics. Dataset statistics showing how many samples we've labeled and what percentage coverage we have. And if training fails for some reason, we want to see an error message that explains what went wrong, not have the whole dashboard crash silently.

---

## 4. What "Good" Feels Like

### Performance and Responsiveness

The dashboard should feel immediate in response to direct user actions. When we click a point, we expect images to update instantly - within the time it takes to move our eyes to the image panel. When we click a label button, we expect the CSV to save immediately without any perceived delay. When we switch color modes, the scatter plot should redraw smoothly in about a second. When we start training, status updates should appear within a second or two.

These aren't arbitrary numbers. They're about the natural rhythm of interaction. If actions feel laggy - if there's a noticeable pause between clicking and seeing the result - the tool becomes tedious to use. We'll start working around it, using it less, going back to the CLI. But if it responds at human speed, it disappears into the workflow.

The dashboard should also feel stable over time. Labels should never get corrupted or lost, even if we're clicking rapidly. Training errors shouldn't crash the whole interface. We should be able to work for two or three hours straight without the dashboard slowing down or acting weird. Race conditions shouldn't be a thing we worry about.

### The Feeling of Cohesion

Everything should happen in one interface. We shouldn't need to switch to the terminal to check something or restart a process. Visual feedback should be everywhere - button states that show when we've clicked them, progress indicators during slow operations, clear status messages about what's happening. When training is active, it should be obvious. When an error occurs, we should see it without having to check the console.

The interface should feel intuitive enough that we don't need to reference documentation during normal use. Not because it's dumbed down, but because it maps naturally to what we're trying to do. Click the thing, label the thing, train the thing. The complexity should be in the decisions we make, not in operating the tool.

### What "Good Enough" Actually Means

This is a prototype for research use. It doesn't need fancy animations or pixel-perfect polish. It's fine if training history is kept in memory and lost when we restart - we care about the current session, not archiving every experiment. Basic error messages are fine as long as they tell us what went wrong. The code needs to be "good enough to extend" but doesn't need to be production-grade.

What actually matters is that it's fast enough not to frustrate us, stable enough that we trust it with our data, organized enough that we can maintain and extend it without fear, and complete enough that it becomes our primary workflow instead of something we use occasionally when the CLI feels too tedious.

---

## 5. Technical Landscape

### Current Architecture

The dashboard is built on Dash and Plotly because they're Python-native and designed for exactly this kind of ML visualization work. Training runs in a background thread so it doesn't block the UI, communicating with the main thread through a queue. Shared state is protected with a simple lock. UI updates happen through polling - the dashboard checks for new metrics every second or so.

These are straightforward choices that work well for a single-user, localhost tool. They're not particularly clever, but they don't need to be. The architecture is simple enough to understand and modify, which is exactly what we want for a prototype.

### Scale and Constraints

We're working with 60,000 MNIST samples, which is completely manageable in memory. The images are tiny (28×28), we have maybe a few hundred labels growing slowly over time, and the latent space is just 2D. This means we can take simple approaches: load everything on startup, keep it all in memory, use a CSV for labels, no database needed, no lazy loading or virtualization required.

The entire system is built around MNIST specifically. The 28×28 dimensions are hardcoded in places, and that's fine - if we want to support other datasets later, we can refactor then. Right now we're focused on making the MNIST experience excellent.

### Integration Philosophy

The dashboard integrates with existing code rather than replacing it. We reuse the SSVAE model, the InteractiveTrainer, the callback system. We don't modify the core training loops or loss functions. The CLI scripts continue to work independently. The labels CSV and checkpoint formats are shared, so you can train in the CLI and visualize in the dashboard, or vice versa.

This constraint is actually helpful - it keeps the scope focused. We're building a layer on top of proven infrastructure, not reimplementing everything.

---

## 6. Open Questions and Design Considerations

### The Polling vs Push Question

Right now we're using polling - the UI checks for updates every second or so during training. It's simple, it works, and with some optimization (polling faster during training, slower when idle, only refreshing components that actually changed), it should be plenty responsive. Training epochs take tens of seconds each, and human perception operates on hundreds of milliseconds, so a one or two second latency on updates is completely acceptable.

But there's always the temptation to think about WebSockets or Server-Sent Events for true push-based updates. The question is whether it's worth the complexity. WebSockets would give us sub-100ms latency on updates and slightly lower CPU usage when idle. But they'd also add a new dependency, require connection management and reconnection logic, and introduce a new paradigm to reason about.

The honest assessment is that for a single-user localhost tool where training epochs are measured in tens of seconds, optimized polling is probably sufficient. We'd get 90% of the benefits for 10% of the complexity. But it's something to keep in mind - if after optimizing polling we still find the experience feels sluggish, push-based updates would be the next thing to try. We just shouldn't reach for it prematurely when simpler fixes might be enough.

### State Management Complexity

Currently we're using a simple dictionary protected by a threading lock. It works fine - one thread owns the state, the background training thread just reads from it and pushes results to a queue. It's not fancy, but it's easy to understand and debug.

We could imagine more sophisticated patterns - a state manager object, immutable state with copy-on-write, message-passing architectures. These would be more elegant in some ways, but they'd also be more complex to reason about. For a prototype with a single background operation and straightforward state needs, the simple dictionary might be exactly right.

The question is whether we'll hit limits as we add features. If we find ourselves fighting race conditions or lock contention becomes a problem, that would be a signal to invest in better state management. But if the simple approach continues to work well, there's no reason to over-engineer it.

### Visualization Tradeoffs

We're rendering 60,000 points in a scatter plot, which Plotly handles well with WebGL acceleration. But there are always tradeoffs. Do we want hover tooltips on every point, or would that slow things down? Should we support selecting multiple points at once for batch labeling, or keep it simple with single selection? Do we want animation when the latent space updates after training, or just a quick swap?

These questions don't need answers right now. They're about polish and refinement that comes after the core experience is solid. But they're worth thinking about because they affect how we structure the visualization code. If we think we might want batch operations later, we'd structure things differently than if we're committed to single-selection only.

### Metrics and Observability

We want loss curves and dataset statistics, but what's the right level of detail? Do we show every component loss in the same plot, or separate plots? Do we persist training history across sessions, or just keep it in memory for the current session? Do we add accuracy metrics beyond just loss values?

The challenge is balancing information density with clarity. Too much data and the plots become cluttered and hard to read. Too little and we miss important signals. We probably want to start minimal - total loss, component losses, basic dataset stats - and add more only if we find ourselves wishing it was there during actual use.

### Code Organization Philosophy

We know the current 1000-line file needs to be split up, but there are different philosophies we could follow. We could organize by technical role (all callbacks in one file, all UI components in another, all state in a third), or by feature (everything related to labeling together, everything related to training together), or by layer (data access, business logic, presentation).

For a dashboard this size, organizing by technical role probably makes the most sense - state management in one place, UI layout in another, callback logic in a third. It makes it easy to find "all the callbacks" or "all the state initialization" without jumping between files. But we want to avoid creating a rigid structure that becomes painful to modify. The organization should serve us, not constrain us.

---

## 7. Success Metrics and Validation

### How We'll Know It's Working

The ultimate test is whether we actually use it. Not whether we can use it, or whether it technically works, but whether we naturally reach for it when doing active learning experiments. If we find ourselves opening the dashboard by default rather than thinking "should I use the CLI or the dashboard?", that's success.

More concretely, success means we can complete active learning experiments faster than before. A full label-train-evaluate cycle that used to require terminal commands, script execution, and manual file checking now happens in one continuous flow. We're not fighting the tool or working around its limitations - it's enabling the workflow we want.

Success also means we trust it. We don't double-check the CSV after labeling to make sure it saved correctly. We don't worry about race conditions when clicking quickly. We don't keep the CLI as a backup "just in case." We just use it confidently.

### What Good Feels Like in Practice

When we sit down to do active learning, the dashboard should disappear into the background of our thinking. We should be thinking about which samples to label, whether the model is learning the right patterns, whether we need to adjust the loss balance - not thinking about the tool itself. If we're thinking "I wish this was faster" or "why did that just happen?" then the tool is getting in the way.

The experience should feel fluid. Moving from exploration to labeling to training should be seamless, without context switches or interruptions. The feedback should be clear enough that we always know what's happening without it being noisy or overwhelming. Errors should be rare, but when they happen they should be handleable without panic.

### The Bar for "Done"

We're done when the dashboard is our preferred tool for active learning work. When we'd be comfortable showing it to a colleague. When we can extend it to add new features without anxiety about breaking things. When we stop thinking about what's missing and start just using what's there. It means the code is organized well enough that we can maintain and evolve it.

Essentially, we're done when we stop treating it as a "project" and start treating it as a "tool."

