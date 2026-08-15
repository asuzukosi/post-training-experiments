Smoke overrides: two steps, batch 1, fp32, no saving, no W&B.

Deliberately does NOT override `max_length` or any other data-shape setting — smokes use each
stage's production value. A 256-token cap once made a smoke pass while training nothing: real rows
are longer than that, truncation removed every assistant token, and loss stayed at 0.0 while the
test still confirmed a checkpoint existed. Keep smokes short via `max_steps`, never by reshaping
the data.
