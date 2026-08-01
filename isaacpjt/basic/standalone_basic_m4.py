from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import time
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

world = World(stage_units_in_meters=1.0)    

step_count = 0

while simulation_app.is_running():
    world.step(render=True)
    time.sleep(0.01)
    step_count += 1

    if step_count % 100 == 0:
        print(f"Step count: {step_count}")
