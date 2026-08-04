from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})     # 1. Application

import numpy as np
import time
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

world = World(stage_units_in_meters=1.0)                # 2. World
stage = omni.usd.get_context().get_stage()              # 3. Stage

# cube_prim1 = DynamicCuboid(                              # 4. Prim
#     prim_path="/World/BlueCube",
#     name="blue_cube",
#     position=np.array([0.0, 0.0, 10.0]),
#     scale=np.array([0.1, 0.1, 0.1]),
#     color=np.array([0.0, 0.0, 1.0]),
# )


cube_prim2 = DynamicCuboid(                              # 4. Prim
    prim_path="/World/RedCube",
    name="red_cube",
    position=np.array([0.0, 0.0, 0.5]),
    scale=np.array([0.3, 0.3, 0.3]),
    color=np.array([1.0, 0.0, 0.0]),
    mass=100
)

world.scene.add_default_ground_plane()                  # 5. Scene
world.scene.add(cube_prim2)
step_count = 0
world.reset()


while simulation_app.is_running():                      # 6. Simulation
    world.step(render=True)
    time.sleep(0.01)
    step_count += 1
    n = step_count
    if step_count % 100 == 0:
        print(f"큐브가 생성되었습니다: {step_count}")
        cube_prim = DynamicCuboid(                              # 4. Prim
            prim_path=f"/World/RedCube_{n}",
            name=f"red_cube_{n}",
            position=np.array([0.0, 0.0, 5.0]),
            scale=np.array([0.1, 0.1, 0.1]),
            color=np.array([0.0, 0.0, 1.0]),
            mass=10 - (step_count * 0.5)
        )
        
        world.scene.add(cube_prim)
        


world.reset()
simulation_app.close()
