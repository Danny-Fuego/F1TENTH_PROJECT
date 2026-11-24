# F1TENTH_PROJECT

Autonomous racing on the F1Tenth platform. This is undergraduate research at MTSU (Middle Tennessee State University). The stack runs in the F1Tenth simulator and on a real 1/10-scale car with an NVIDIA Jetson Orin, a Hokuyo LiDAR and a VESC.

The car follows a precomputed racing line with **Pure Pursuit**. The goal is for it to hand control to **Follow the Gap** only when something new is in its path, with a separate **AEB** node braking if a collision is close. See [Status](#status) for where each part stands.

Demo: [real-car run on LinkedIn](https://www.linkedin.com/posts/danielbaadom_f1tenth-autonomousvehicles-robotics-ugcPost-7529326200922517505-88RE/)

---

## Status

| Part | State | Results so far |
|---|---|---|
| Pure Pursuit | Working on the real car and in the sim | Real car, 2nd-floor raceline at 50% speed: about 10 cm average tracking error, 0.27 m worst |
| AEB | Working | Sim, PP at 70% speed: 18 laps with no false brakes |
| Follow the Gap | Working in the sim | Laps the Levine and 2nd-floor maps; gets through the 7-box obstacle map. Stops at very tight gaps (0.65 m) instead of squeezing through |
| Switcher (PP ↔ FTG) | **Work in progress** | Still hands control to FTG at random on a clean track, from a few times per 10 laps to more often |

**What the switcher problem turned out to be.** The switcher compares each LiDAR point with the clean map to tell known walls from new objects. Small errors in the car's estimated position, which grow with speed and lag, put real wall points up to about 0.5 m off the map's walls, so they get flagged as "new objects" and FTG takes over. A stricter map check helped but didn't remove it. A second, rarer trigger comes from the wall check predicting the path from the current steering angle on corner entry.

**Next steps**
- Line each scan up with the map before judging points (the same idea a particle filter uses), keeping the correction small and fast enough for the Orin
- Rebuild the wall check on the AEB's two-prediction rule, so FTG acts a little before AEB brakes
- A smooth handback from FTG to Pure Pursuit; right now PP oscillates when it rejoins the line at an angle
- Later: a local planner so the car can squeeze through tight gaps and rejoin the racing line cleanly

---

## How it fits together

```
                 /pp_drive                      /drive
Pure Pursuit  ───────────────►                ─────────►  ackermann_mux ──► VESC
                               Switcher node
Follow the Gap ──────────────►  (PP / FTG / STOP)
                 /gap_drive                                    ▲ /safety (priority 200)
                                                               │
                                               AEB ────────────┘
```

- **Pure Pursuit** (`pure_pursuit/`) follows a racing line CSV (`x, y, v`). It uses a speed-scaled lookahead and a corner lookahead that limits corner cutting. It also has a "black box" that saves the last 3 s of driving when something goes wrong.
- **Follow the Gap** (`ftg/`) is purely reactive: one scan in, one drive command out. It uses a disparity extender plus a tight-gap "squeeze" mode, a stop ring around the car body, and speed scaled by steering.
- **Switcher** (`switcher/`) runs Pure Pursuit by default. On every scan it checks the arc PP is actually driving:
  - Points that are *not on the map* (people, boxes) trigger FTG early, using `ttc_new_object`.
  - Points on *known walls* trigger it only when a crash is close, using `ttc_wall`.
  - Hysteresis on entering and leaving prevents flicker.
  - If the active controller goes quiet, it publishes a stop.
- **AEB** (`aeb/`, ROS 2 package) brakes on time to collision, independent of everything else. It latches until a controller button or the `/aeb_reset` service resets it.
- **Stanley avoidance** (`stanley_avoidance/`) is an earlier Stanley-controller experiment with obstacle avoidance on the Spielberg raceline.

---

## Repo layout

```
ftg/                ftg_node.py + sim params (ftg_sim.yaml) and switcher params (ftg_switcher.yaml)
pure_pursuit/       pure_pursuit_node.py, waypoint_visualizer_node.py
  pp_real.yaml        real-car params (particle filter pose, slow raceline)
  config/config.yaml  sim params (Spielberg)
  racelines/          second_floor_raceline(.csv / _slow.csv), Spielberg_raceline.csv
switcher/           switcher_node.py, switcher_sim.yaml, pp_switcher.yaml
aeb/                ROS 2 package (colcon) + aeb_real.yaml
stanley_avoidance/  stanley_avoidance_node.py
tools/
  map_to_raceline.py  SLAM map -> centerline -> min-curvature racing line + speed profile
  pf_start.sh         start the particle filter and send the start pose
  pf_monitor.py       warns when the particle filter freezes or jumps
  car_model_node.py   3D car marker for Foxglove/RViz
  sim_levine.sh       switch the sim to the Levine map
lidar_test/         Hokuyo setup script + live polar-plot visualizer
rviz/ftg.rviz       RViz config for Follow the Gap
maps/sim/           2nd-floor maps for the sim: clean track, obstacles, test scenarios
maps/real/          2nd-floor SLAM maps used by the particle filter
real_car/config/    my tuned configs for the third-party real-car stack (see below)
```

---

## Dependencies

- ROS 2 Humble (on the Jetson Orin, Ubuntu 22.04)
- [f1tenth_gym_ros](https://github.com/f1tenth/f1tenth_gym_ros) for the simulator
- Real car: [f1tenth_system](https://github.com/f1tenth/f1tenth_system) (VESC driver, ackermann_mux, teleop), [particle_filter](https://github.com/f1tenth/particle_filter) + [range_libc](https://github.com/kctess5/range_libc), and `urg_node2` for the Hokuyo
- Python: `numpy`, `scipy`. `tools/map_to_raceline.py` also needs `opencv-python pillow pyyaml trajectory-planning-helpers quadprog`.

The nodes in `ftg/`, `pure_pursuit/` and `switcher/` run directly with `python3` and a params file. The configs use absolute paths under `/home/john/...`, so edit those if your checkout is somewhere else.

### Real-car configs

`real_car/config/` holds the files I changed in the third-party packages. Copy them over the originals in your workspace:

| File | Goes to |
|---|---|
| `vesc.yaml`, `joy_teleop.yaml`, `mux.yaml`, `sensors.yaml`, `f1tenth_online_async.yaml` | `f1tenth_ws/src/f1tenth_system/f1tenth_stack/config/` |
| `particle_filter_localize.yaml` | `f1tenth_ws/src/particle_filter/config/localize.yaml` |
| `maps/real/*` | `f1tenth_ws/src/particle_filter/maps/` |
| `maps/sim/*` | `f1tenth_gym_ros/maps/` |

`mux.yaml` gives the AEB its own `safety` input at priority 200, above joystick (100) and navigation (10).

---

## Running in the simulator

```bash
# Terminal 1: simulator (sim.yaml map_path -> maps/second_floor_track or Spielberg_map)
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

**Pure Pursuit alone (Spielberg):**
```bash
python3 ~/F1TENTH_PROJECT/pure_pursuit/pure_pursuit_node.py --ros-args --params-file ~/F1TENTH_PROJECT/pure_pursuit/config/config.yaml
```

**Follow the Gap alone:**
```bash
python3 ~/F1TENTH_PROJECT/ftg/ftg_node.py --ros-args --params-file ~/F1TENTH_PROJECT/ftg/ftg_sim.yaml
```

**Full hybrid stack (PP + FTG + switcher):** one terminal each.
```bash
python3 ~/F1TENTH_PROJECT/pure_pursuit/pure_pursuit_node.py --ros-args --params-file ~/F1TENTH_PROJECT/switcher/pp_switcher.yaml
python3 ~/F1TENTH_PROJECT/ftg/ftg_node.py --ros-args --params-file ~/F1TENTH_PROJECT/ftg/ftg_switcher.yaml
python3 ~/F1TENTH_PROJECT/switcher/switcher_node.py --ros-args --params-file ~/F1TENTH_PROJECT/switcher/switcher_sim.yaml
```
The switcher publishes its current mode on `/switcher_mode` (`PP` / `FTG` / `STOP`).

---

## Running on the real car

```bash
source ~/f1tenth_ws/install/setup.bash
ros2 launch f1tenth_stack bringup_launch.py                     # VESC, LiDAR, mux, joystick
~/F1TENTH_PROJECT/tools/pf_start.sh                             # particle filter + start pose
ros2 run aeb aeb_node --ros-args --params-file ~/F1TENTH_PROJECT/aeb/aeb_real.yaml
python3 ~/F1TENTH_PROJECT/pure_pursuit/pure_pursuit_node.py --ros-args --params-file ~/F1TENTH_PROJECT/pure_pursuit/pp_real.yaml
python3 ~/F1TENTH_PROJECT/tools/pf_monitor.py                   # optional: localization health
```

`pp_real.yaml` uses the **slow** raceline and `velocity_percentage: 0.5` for first runs. To build the AEB package, link or copy `aeb/` into `~/f1tenth_ws/src/` and run `colcon build --packages-select aeb`.

---

## Making a racing line from a new map

```bash
python3 tools/map_to_raceline.py ~/maps/second_floor.yaml --out pure_pursuit/racelines
```
This writes `NAME_centerline.csv`, a cleaned track map for the sim, `NAME_raceline.csv` / `NAME_raceline_slow.csv` (`x,y,v`, no header), and a preview PNG. It works for a single loop around a single central block, like the 2nd floor.

---

## Author

Daniel Baadom: Mechatronics Engineering, MTSU

## License

MIT. See [LICENSE](LICENSE).
