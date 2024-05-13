#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import LaserScan




# TODO CHECK: include needed ROS msg type headers and libraries
import math
from .mppi_env import MPPIEnv
from .jax_mpc.mppi import MPPI


from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import time
import matplotlib.pyplot as plt
class oneLineJaxRNG:
    def __init__(self, init_num=0) -> None:
        self.rng = jax.random.PRNGKey(init_num)
        
    def new_key(self):
        self.rng, key = jax.random.split(self.rng)
        return key

jRNG = oneLineJaxRNG(1337)


class MPPIPlanner():
    

    def __init__(
        self,
        waypoints=None,

        debug=False,
    ):
        self.waypoints = waypoints
        self.debug = debug
        self.n_steps = 12
        self.n_samples = 128
        self.jRNG = jRNG
        self.DT = 0.1

        self.mppi_env = MPPIEnv(self.waypoints, self.n_steps, mode = 'ks', DT= self.DT)
        self.mppi = MPPI(n_iterations = 1, n_steps = self.n_steps,
                         n_samples = self.n_samples, a_noise = 1.0, scan = False)
        
        self.a_opt = None
        self.a_cov = None
        self.mppi_state = None
        
        self.target_vel = 3.0
        self.norm_param = np.array([0.45, 3.5])
        self.init_state()
        self.laptime = 0

    
    def init_state(self):
        self.mppi_state =  self.mppi.init_state(self.mppi_env.a_shape, self.jRNG.new_key() )
        self.a_opt = self.mppi_state[0]
    
    
    def plan(self, obs):


        self.a_opt = jnp.concatenate([self.a_opt[1:, :],
                    jnp.expand_dims(jnp.zeros((2,)),
                                    axis=0)])  # [n_steps, dim_a]
        
        # da = jax.random.normal(
        #     self.jRNG.new_key(),
        #     shape=(self.n_samples, self.n_steps, self.mppi_env.a_shape)
        # ) 
        a_opt = self.a_opt.copy()
        da = jax.random.truncated_normal(
            self.jRNG.new_key(),
            -jnp.ones_like(a_opt) - a_opt,
            jnp.ones_like(a_opt) - a_opt,
            shape=(self.n_samples, self.n_steps, 2)
        )

        # vehicle_state = State(
        #     x=states[0],
        #     y=states[1],
        #     delta=states[2],
        #     v=states[3],
        #     yaw=states[4],
        #     yawrate=states[5],
        #     beta=states[6],
        # )
        state_x = obs[0]
        state_y = obs[1]
        delta = obs[2]
        yaw = obs[4]
        v = obs[3]

        state = np.array([state_x, state_y, delta, v, yaw])

        ref_traj,_ = self.mppi_env.get_refernece_traj(state, target_speed = self.target_vel,  vind = 5, speed_factor= 1)
        # print(ref_traj.shape) #[n_steps + 1, 7]

        self.mppi_state, sampled_traj, s_opt, _, _,_ = self.mppi.update(self.mppi_state, self.mppi_env, state.copy(), self.jRNG.new_key(), da)

        # print(sampled_traj.shape)

        if self.debug:
            # plots sampled trajectories every 40 steps
            if(self.laptime %40 == 0):
                for i in range(128):
                    plt.plot(sampled_traj[0][i][:, 0], sampled_traj[0][i][:, 1])
                plt.show()

        # self.laptime+=1
        a_opt = self.mppi_state[0]
        control = a_opt[0]
        scaled_control = np.multiply(self.norm_param, control)
        # print(sampled_traj[0].shape) [n_samples, n_steps, 5]
        # print(control)
        # print(scaled_control)
        # TODO: check the mppi outputs( its in steerv, accl), convert to vel and steering angle control ig and check mpc node what they do
        
        steerv = scaled_control[0]
        accl = scaled_control[1]

        steer_angle = delta + steerv*self.DT
        speed = v + accl*self.DT
        return steer_angle, speed

def main(args=None):

    rclpy.init(args=args)
    print("MPC Initialized")
    mpc_node = MPPIPlanner()
    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()