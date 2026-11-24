#!/bin/bash
# Switch the sim to the Levine map and set up a saveable RViz config for Follow the Gap.
set -e
CFG=~/f1tenth_gym_ros/config/sim.yaml
MAPS=~/f1tenth_gym_ros/maps
[ -f "$MAPS/levine.yaml" ] || { echo "Levine map not found. Maps you have:"; ls "$MAPS"; exit 1; }

# 1) back up the 2nd-floor config once, then point the sim at Levine
[ -f ~/f1tenth_gym_ros/config/sim_second_floor.yaml ] || cp "$CFG" ~/f1tenth_gym_ros/config/sim_second_floor.yaml
sed -i -E "s|^(\s*map_path:).*|\1 '$MAPS/levine'|; s|^(\s*sx:).*|\1 0.0|; s|^(\s*sy:).*|\1 0.0|; s|^(\s*stheta:).*|\1 0.0|" "$CFG"
echo "--- sim.yaml now:"; grep -E "map_path|^\s*s(x|y|theta):" "$CFG"

# 2) your own RViz config, copied from the sim's default, in a folder rebuilds never touch
mkdir -p ~/F1TENTH_PROJECT/rviz
RV=$(find ~/f1tenth_gym_ros -name "gym_bridge.rviz" | head -1)
if [ -f ~/F1TENTH_PROJECT/rviz/ftg.rviz ]; then echo "ftg.rviz already exists, left it alone"
elif [ -n "$RV" ]; then cp "$RV" ~/F1TENTH_PROJECT/rviz/ftg.rviz; echo "--- made ~/F1TENTH_PROJECT/rviz/ftg.rviz"
else echo "!! couldn't find the sim's gym_bridge.rviz, skip this part"; fi

# 3) rebuild the sim so the new sim.yaml is used
INST=$(find ~ -maxdepth 4 -type d -name f1tenth_gym_ros -path "*/install/*" | head -1)
[ -n "$INST" ] || { echo "!! couldn't find the sim workspace, tell Claude"; exit 1; }
WS=$(dirname "$(dirname "$INST")")
echo "--- rebuilding in $WS"
cd "$WS" && colcon build --packages-select f1tenth_gym_ros
echo "DONE. Run: source $WS/install/setup.bash   then launch the sim."
