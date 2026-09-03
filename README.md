# F1TENTH_PROJECT

Autonomous racing research project implementing and comparing path following and obstacle avoidance algorithms on the F1Tenth platform. Developed as part of undergraduate research at MTSU (Middle Tennessee State University).

---

## Overview

This project implements a **hybrid autonomous racing stack** combining:
- **Pure Pursuit** — primary racing algorithm following an optimal raceline
- **Follow the Gap** — reactive obstacle avoidance
- **Switcher Node** — intelligent switching between the two based on obstacle detection

The system runs in the F1Tenth ROS2 simulator (Docker) on the Spielberg F1 circuit using a pre-computed optimal raceline.

---

## Project Structure

```
F1TENTH_PROJECT/
  pure_pursuit/
    pure_pursuit_node.py          # Pure Pursuit path following algorithm
    waypoint_visualizer_node.py   # RViz visualization of raceline
    racelines/
      Spielberg_raceline.csv      # Optimal raceline with velocity profile
    config/
      config.yaml                 # Parameters for Pure Pursuit
  scripts/
    reactive_node.py              # Follow the Gap obstacle avoidance
    switcher_node.py              # Hybrid switching logic
  README.md
```

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [F1Tenth Gym ROS](https://github.com/f1tenth/f1tenth_gym_ros) cloned and built
- Python 3.8+
- ROS2 Foxy (inside Docker container)

---

## Simulator Setup

**Clone and set up the F1Tenth simulator:**
```bash
git clone https://github.com/f1tenth/f1tenth_gym_ros
cd f1tenth_gym_ros
```

**Set map to Spielberg in `config/sim.yaml`:**
```yaml
map_path: '/sim_ws/src/f1tenth_gym_ros/maps/Spielberg_map'
```

**Set car starting position to match raceline start:**
```yaml
sx: -0.0440806
sy: -0.8491629
stheta: 0.0
```

---

## Running the System

### Terminal 1 — Start Docker
```bash
cd f1tenth_gym_ros
docker-compose up
```
Press **d** to detach.

### Terminal 2 — Launch Simulator
```bash
docker exec -it f1tenth_gym_ros-sim-1 /bin/bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

### Terminal 3 — Copy Project into Container
```bash
docker cp ~/F1TENTH_PROJECT f1tenth_gym_ros-sim-1:/sim_ws/src/
```

### Terminal 4 — Run Pure Pursuit
```bash
docker exec -it f1tenth_gym_ros-sim-1 /bin/bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
python3 /sim_ws/src/F1TENTH_PROJECT/pure_pursuit/pure_pursuit_node.py
```

### Terminal 5 — Run Waypoint Visualizer
```bash
docker exec -it f1tenth_gym_ros-sim-1 /bin/bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
python3 /sim_ws/src/F1TENTH_PROJECT/pure_pursuit/waypoint_visualizer_node.py
```

### Terminal 6 — Run Follow the Gap
```bash
docker exec -it f1tenth_gym_ros-sim-1 /bin/bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
python3 /sim_ws/src/F1TENTH_PROJECT/scripts/reactive_node.py
```

### Terminal 7 — Run Switcher
```bash
docker exec -it f1tenth_gym_ros-sim-1 /bin/bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
python3 /sim_ws/src/F1TENTH_PROJECT/scripts/switcher_node.py
```

### Browser
Go to `http://localhost:8080/vnc.html` and click Connect.

In RViz add a **MarkerArray** display and set topic to `/waypoints` to visualize the raceline.

---

## ROS2 Topics

| Topic | Publisher | Subscriber | Description |
|---|---|---|---|
| `/scan` | Simulator | Reactive Node, Switcher | LIDAR scan data |
| `/ego_racecar/odom` | Simulator | Pure Pursuit | Car position and velocity |
| `/pp_drive` | Pure Pursuit | Switcher | Pure Pursuit steering commands |
| `/gap_drive` | Follow the Gap | Switcher | Follow the Gap steering commands |
| `/drive` | Switcher | Simulator | Final drive commands to car |
| `/lookahead_angle` | Pure Pursuit | Switcher | Current racing line direction |
| `/current_speed` | Pure Pursuit | Follow the Gap | Current car speed |
| `/waypoints` | Waypoint Visualizer | RViz | Raceline visualization |
| `/current_waypoint` | Pure Pursuit | RViz | Closest waypoint marker |
| `/lookahead_waypoint` | Pure Pursuit | RViz | Lookahead waypoint marker |

---

## Parameters

### Pure Pursuit (`config/config.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `K_p` | 0.47 | Proportional steering gain |
| `K_d` | 0.11 | Derivative gain (dampens oscillation) |
| `min_lookahead` | 0.5 | Minimum lookahead distance (m) |
| `max_lookahead` | 2.8 | Maximum lookahead distance (m) |
| `lookahead_ratio` | 8.0 | Speed-to-lookahead scaling |
| `steering_limit` | 25.0 | Maximum steering angle (degrees) |
| `velocity_percentage` | 0.7 | Fraction of optimal raceline speed |

**Live tuning while running:**
```bash
docker exec -it f1tenth_gym_ros-sim-1 /bin/bash
source /opt/ros/foxy/setup.bash
ros2 param set /pure_pursuit_node velocity_percentage 0.5
ros2 param set /pure_pursuit_node K_p 0.4
```

### Switcher Node

| Parameter | Default | Description |
|---|---|---|
| `obstacle_threshold` | 3.0 | Distance (m) to trigger obstacle avoidance |
| `CLEAR_THRESHOLD` | 15 | Consecutive clear scans before returning to Pure Pursuit |

---

## Algorithm Details

### Pure Pursuit
Follows the pre-computed Spielberg optimal raceline. Steering angle calculated as:
```
γ = K_p × (2y / L²) + K_d × (y - prev_y)
```
Where `y` is lateral distance to lookahead waypoint and `L` is variable lookahead distance scaled with speed.

Raceline sourced from [f1tenth/f1tenth_racetracks](https://github.com/f1tenth/f1tenth_racetracks). Implementation reference: [Steven Gong's f1tenth_ws](https://github.com/CL2-UWaterloo/f1tenth_ws).

### Follow the Gap
Reactive obstacle avoidance running every LIDAR scan:
1. Preprocess LIDAR — clip to 3m, smooth with moving average, limit to ±70° viewport
2. Find closest obstacle — draw physical safety bubble of 0.6m radius
3. Find largest gap in remaining free space
4. Steer toward middle of gap
5. Speed scaled based on steering angle (30-100% of current speed)

### Switcher
- **Entry to Follow the Gap:** Obstacle detected in single LIDAR beam matching Pure Pursuit lookahead direction
- **Exit to Pure Pursuit:** Wide zone (±20 beams around lookahead) clear for 15 consecutive scans

---

## Tuning Results

Stable lap completion achieved at 70% of optimal speed (~5.6 m/s):

| Parameter | Value |
|---|---|
| `K_p` | 0.47 |
| `K_d` | 0.11 |
| `max_lookahead` | 2.8m |
| `velocity_percentage` | 0.7 |

At 100% speed (8 m/s) the PD Pure Pursuit becomes unstable — consistent with known Pure Pursuit limitations at high speeds. This motivates future work with the Stanley controller.

---

## Known Limitations

- Pure Pursuit unstable above ~70% of optimal speed
- Follow the Gap has no spatial awareness — treats walls and obstacles identically
- Hybrid switcher works well for isolated obstacles but struggles with diagonal/extended obstacles
- These limitations motivate transition to Stanley Avoidance as next research phase

---

## Next Steps

- Stanley Avoidance implementation (occupancy grid based)
- Multi-agent overtaking research
- Sim-to-real transfer on physical F1Tenth hardware

---

## References

- [F1Tenth Gym ROS](https://github.com/f1tenth/f1tenth_gym_ros)
- [F1Tenth Racetracks](https://github.com/f1tenth/f1tenth_racetracks)
- [Steven Gong's F1Tenth Stack](https://github.com/CL2-UWaterloo/f1tenth_ws)
- [F1Tenth 2020 Course — Penn Engineering](https://www.youtube.com/playlist?list=PL7rtKJAz_mPdFDJtufKmqfWRNu55s_LMc)

## CAD Files
F1Tenth assembly CAD files (SolidWorks):
[View on Google Drive](https://drive.google.com/drive/folders/1RC6OWucZGz4a2GUob5C4aIZ_LDUboesL?usp=sharing)
