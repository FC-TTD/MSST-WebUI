# MSST managed adoption

Preparation on the `hub` branch; not yet a completed production handoff.
Native source: `41d03345d63f61d9605a68997cf4312e3c7b8718`. Existing image:
`registry.ttd/msst-webui/msst@sha256:b1113e1b2273806a7246b066f95e6b275e60d338f371883b9430b60aadc88470`.
Original CPU/GPU split, PyTorch2.7.1+cu128, Gradio4.38.1, FastAPI0.111.0,
Pydantic2.10.6 and CPython3.11.13 are preserved. No upstream algorithm files change.

## Boundaries

Only GPU-using services enter this pool; independent CPU-only services are excluded.
This GPU application's native CPU helpers and force_cpu options remain available.
Copy original UI controls/layouts; change GPU callback bindings only. CPU audio
conversion, file ensemble and SDR helpers do not create pool activities.

API `/api/v1/tasks/msst-batch` retains sync/SSE/async, task IDs, status/result/cancel
and model discovery. Original OpenAPI matches the new factory exactly in both
local dependency fixtures and the production base image. Sync handler's existing
blocking execution remains; adoption does not silently make it concurrent.

Native API semaphore(2) is retained. It is not a global limit for UI callbacks:
the original Gradio queues and independent UI events remain. MSST creates a model
in a native process per task and releases it when that process exits; Hub does
not introduce a new per-request unload, nor claim these native weights are cached
across tasks. A persistent leased supervisor owns the native execution tree.

## Execution ownership

Async HTTP receipts are not completion. A persistent background owner copies
request context and consumes native SSE under one Runtime execution scope until
native finally and process exit. SSE uses an owner thread as well: ASGI's separate
next() calls must not carry a ContextVar token across copied contexts.

Cancellation reaches the active engine's native task/PID registry, without a new
load or activity. Cleanup must finish before an interrupted native task becomes
canceled; unknown execution retains uncertainty. Native failed sync results stay
failed in the pool while retaining the original API response shape.

UI coverage includes MSST, VR, preset, ensemble, SOME, training and validation.
Training returns its original startup message but a background activity keeps
consuming the engine stream until the actual child/grandchild execution ends.
MSST explicitly sets the SDK's business execution timeout to `None`, preserving
native long-running training/folder calls; startup, completion, stream close and
shutdown controls remain bounded. Transport failure still preserves uncertainty.
UI GPU selection represents lease-local cuda:0; ordinal/numeric-string/current
UUID validation rejects other GPUs instead of native parsers silently falling
back. Existing CPU/GPU precision and CPU modes are not rewritten.

The supervisor instruments only each imported native module's multiprocessing
handle, recording the original Process objects/descendant identities; it does not
change inference targets or impose a global lock. Unproven child exit becomes
sticky uncertainty and the SDK completion hook preserves `unknown`.

CPython's spawn resource tracker otherwise outlives the engine and prevents empty
process-group proof. The release hook, after all native work is confirmed done,
validates this engine's helper identity/parent/starttime/group, finalizes only its
semaphores and closes/reaps that helper. This narrow private-API compatibility
hook is covered by real SDK process tests; replace it when public Python/SDK
helper teardown becomes available. It is not a general SDK policy or per-request
inference change.

## Data and deployment

Worker-only destination; edge remains outside pool scheduling. Original domain
`http://msst` is transferred only after candidate real API/UI acceptance. Candidate
validation uses `http://msst-pool-validation`, not a new permanent business URL.

Retain `/TTD-Data/msst/data` (including task SQLite), original pretrain mounts and
`/TTD` path semantics. Copy original container-owned configs/results/input/cache
and its dedicated root cache to explicit shared MSST-owned directories; preserve
prior files. Native boot can replace `data` on a version mismatch and auto-delete
cache. Managed boot instead checks the existing config version, creates missing
files only, and requires an explicit migration on mismatch. It never resets the
task DB or deletes archived outputs as a startup side effect.

The CPU-only image check constructed the full copied UI (569 components,
121 dependencies), matched API schema and assembled the SDK app with CUDA hidden.
This is not GPU deployment acceptance.

The first managed async API single/double workloads passed. The first real UI
attempt failed in the SDK while testing the truth value of a Gradio4 Progress
object; native inference had already started and later produced files. That
attempt remains a failed UI delivery, not an unexecuted request or acceptance.
The fix treats progress as a callable (`is not None` in SDK, explicit forwarding
lambda in the model UI bridge). Replacement uses a fresh actor after confirmed
drain/retirement; no in-place package patch or revival of the old actor.

## Native GPU evidence collected before handoff

2026-09-16, original edge service, 20.143s input. Default vocals model single task
peaked at4852MiB; two native tasks overlapped at9704MiB and both produced complete
44.1kHz stereo stems. Original other-process GPU memory (2832MiB) was attributed
separately and not charged as MSST memory.

All nine Gateway checkpoint profiles produced real files. Other single-task peaks:
vocals Becruily3090MiB, BigBeta4424, instrumental Becruily3090, male/female2100,
dereverb3090, Bandit1638, HTDemucs1356, Apollo12004. The default-model double test
does not prove every pair of large checkpoints fits; do not equate two API slots
to a whole-service memory guarantee or silently reduce native concurrency.

Native evidence remains edge `/TTD-Data/msst/data/hub-adoption-baseline-20260916/`
and `hub-adoption-profiles-20260916/`. Actual managed API/UI, asynchronous cancellation,
NVML process attribution, unload/reload, original-domain handoff and Gateway task
ownership/Range download/usage are still required before declaring adoption done.
