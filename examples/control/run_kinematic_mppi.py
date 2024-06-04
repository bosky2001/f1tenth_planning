# MIT License

# Copyright (c) Hongrui Zheng, Johannes Betz

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



import numpy as np
import gymnasium as gym
from f1tenth_gym.envs import F110Env
import time


from f1tenth_planning.control.kinematic_mppi.kinematic_mppi import MPPIPlanner
# from f1tenth_planning.control.kinematic_mppi.kinematic_mppi_wip import MPPIPlanner

import matplotlib.pyplot as plt

def main():
    """
    STMPPI example. This example uses fixed waypoints throughout the 2 laps.
    For an example using dynamic waypoints, see the lane switcher example.
    """

    # create environment
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "control_input": "accl",
            "observation_config": {"type": "dynamic_state"},
        },
        render_mode="human",
    )
    # create planner

    # print(env.params)
    planner = MPPIPlanner(track=env.track, debug=False, env_config=env.params)
    # planner = MPPIPlanner(track=env.track, debug=False)

    # env.unwrapped.add_render_callback(planner.render_waypoints)

    env.unwrapped.add_render_callback(planner.render_waypoints)
    env.unwrapped.add_render_callback(planner.render_local_plan)
    env.unwrapped.add_render_callback(planner.render_mppi_sol)
    # env.unwrapped.add_render_callback(planner.render_mppi_samples)

    
    # reset environment
    poses = np.array(
        [
            [
                env.track.raceline.xs[0],
                env.track.raceline.ys[0],
                env.track.raceline.yaws[0],
                0.0
            ]
        ]
    )
    obs, info = env.reset(options={"poses": poses})
    done = False
    env.render()

    # waypoints = [
    #         env.track.raceline.ss,
    #         env.track.raceline.xs,
    #         env.track.raceline.ys,
    #         env.track.raceline.yaws,
    #         env.track.raceline.ks,
    #         env.track.raceline.vxs,
    #         env.track.raceline.axs
    # ]

    # waypoints = np.array(waypoints)
    # print("wypoints shape is " , waypoints.shape)
    laptime = 0.0
    start = time.time()
    steerv_inputs = []
    while not done:
        plan_time = time.time()
        steerv, accl = planner.plan(obs["agent_0"])
        steerv_inputs.append(steerv)

        print(1/(time.time() - plan_time))
            
        obs, timestep, terminated, truncated, infos = env.step(
            np.array([[steerv, accl]])
        )
        done = terminated or truncated
        laptime += timestep
        env.render()

        print(
            "speed: {}, steer vel: {}, accl: {}".format(
                obs["agent_0"]["linear_vel_x"], steerv, accl
            )
        )
        if laptime > 60:
            done = True
        
    plt.plot(planner.tracking_error)
    # plt.plot(steerv_inputs)
    plt.show()
    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)



if __name__ == "__main__":
    main()
