# Native async acceptance handoff

The Hub local HTTP admission may finish before the native MSST async worker
enters its execution scope. Declare background acceptance before starting the
thread, preserve the existing copied request context, and report completion when
the native owner exits. This uses the shared SDK hooks and does not replace the
native queue, cancellation behavior, UI, model selection, or inference code.

Validation: 12 hub_tasks unittest cases passed with SDK 0.2.0, including receipt
ownership, thread context, cancellation, native failure and start/finish hooks.
Deployment refresh overlays only hub_runtime/tasks.py on the current fixed
managed image, through Hub's fenced publication and real async separation check.
