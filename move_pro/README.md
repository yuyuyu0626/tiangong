# move_pro

`move_pro` uses the complete robot implementation from `move` and replaces
only the fixed placement decision with an online PCT decision.

## Source ownership

- Robot URDF handling, IK, route planning, Isaac Gym actors, timelines,
  attach/release behavior, and the final task scripts are copied unchanged
  from `move`.
- `move_pro/pct_core/space.py`, `PctTools.py`, and `convex_hull.py` are copied
  unchanged from `Online-3D-BPP-PCT/pct_envs/PctDiscrete0`.
- `bpp_decider.py` is the adapter that calls the PCT space and converts a PCT
  placement to the pallet world coordinate system.
- `integrator.py` only converts an incoming box stream into online placement
  records. It deliberately does not reimplement IK or simulation.

The large URDF mesh assets remain shared through `move/assets`; duplicating
them would not change behavior and would make the repository much larger.

## Offline decision check

```bash
python -m move_pro.run --mode plan --method LSAH --num-boxes 30
python -m unittest discover -s move_pro/tests -v
```

The default allows only the two upright orientations `(x, y, z)` and
`(y, x, z)`. The current robot flow does not physically support arbitrary
box tipping, so enabling all six PCT orientations would produce plans that
the motion layer cannot execute faithfully.

## Move task baseline

All original task entry points are available under `move_pro.tasks`, for
example:

```bash
python -m move_pro.tasks.task1_2 --fast --viewer-render-every 8
python -m move_pro.tasks.task2_2 --fast-viewer --viewer-render-every 8
```

They intentionally retain imports from `move`; this keeps their runtime
behavior byte-for-byte aligned with the known working project.
