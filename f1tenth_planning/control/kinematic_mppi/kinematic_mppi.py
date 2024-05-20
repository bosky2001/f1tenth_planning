import jax
import jax.numpy as jnp
import numpy as np
import math
# from .mppi_env import MPPIEnv
# from .jax_mpc.mppi import MPPI
import time
import matplotlib.pyplot as plt
# from f1tenth_planning.utils.utils import nearest_point
from numba import njit

from functools import partial

import yaml
from pathlib import Path


class ConfigYAML():
    """
    Config class for yaml file
    Able to load and save yaml file to and from python object
    """
    def __init__(self) -> None:
        pass
    
    def load_file(self, filename):
        d = yaml.safe_load(Path(filename).read_text())
        for key in d: 
            setattr(self, key, d[key]) 
    
    

config = ConfigYAML()
config.load_file("/home/bosky2001/Downloads/ESE_6150_FINAL_PROJECT/f1tenth_gym/f1tenth_planning/f1tenth_planning/control/kinematic_mppi/config.yaml")

@njit(cache=True)
def nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.
    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
            NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world
    Returns:
        nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
        i (int): index of nearest point in the array of trajectory waypoints
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    dist_from_segment_start = np.linalg.norm(diffs[min_dist_segment] * t[min_dist_segment])
    return projections[min_dist_segment], dist_from_segment_start, dists[min_dist_segment], t[
        min_dist_segment], min_dist_segment


class MPPI():
    """An MPPI based planner."""
    def __init__(self, n_iterations=5, n_steps=16, n_samples=16, temperature=0.01,
                damping=0.001, a_noise=0.5, scan=False,
                adaptive_covariance=False, mode='st'):
        self.n_iterations = n_iterations
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.temperature = temperature
        self.damping = damping
        self.a_std = a_noise
        self.scan = scan  # whether to use jax.lax.scan instead of python loop
        self.adaptive_covariance = adaptive_covariance
        self.mode = mode


    def init_state(self, a_shape, rng):
        # uses random as a hack to support vmap
        # we should find a non-hack approach to initializing the state
        dim_a = jnp.prod(a_shape)  # np.int32
        a_opt = 0.0*jax.random.uniform(rng, shape=(self.n_steps,
                                                dim_a))  # [n_steps, dim_a]
        # a_cov: [n_steps, dim_a, dim_a]
        if self.adaptive_covariance:
            # note: should probably store factorized cov,
            # e.g. cholesky, for faster sampling
            a_cov = (self.a_std**2)*jnp.tile(jnp.eye(dim_a), (self.n_steps, 1, 1))
        else:
            a_cov = None
        return (a_opt, a_cov)

    def update(self, mpc_state, env, env_state, rng):
        # mpc_state: ([n_steps, dim_a], [n_steps, dim_a, dim_a])
        # env: {.step(s, a), .reward(s)}
        # env_state: [env_shape] np.float32
        # rng: rng key for mpc sampling

        dim_a = jnp.prod(env.a_shape)  # np.int32
        a_opt, a_cov = mpc_state
        a_opt = jnp.concatenate([a_opt[1:, :],
                                jnp.expand_dims(jnp.zeros((dim_a,)),
                                                axis=0)])  # [n_steps, dim_a]

        
        def iteration_step(input_, _):
            a_opt, a_cov, rng = input_
            rng_da, rng = jax.random.split(rng)

            
            da = jax.random.truncated_normal(
                rng_da,
                -jnp.ones_like(a_opt) - a_opt,
                jnp.ones_like(a_opt) - a_opt,
                shape=(self.n_samples, self.n_steps, 2)
            )
            a = jnp.clip(jnp.expand_dims(a_opt, axis=0) + da, -1.0, 1.0)
            # print('up', env_state.shape)
            r_sample, s, ret_dyna = jax.vmap(self.rollout, in_axes=(0, None, None))(
                a, env, env_state
            )

            r = jax.vmap(env.reward_fn, in_axes=(0, None))(
                s, env.reference
            ) # [n_samples, n_steps]
            
            # if self.mode == 'nf':
            #     r = r + (r_sample * 4) ** 4 * jnp.sign(r_sample)
            R = jax.vmap(self.returns)(r) # [n_samples, n_steps], pylint: disable=invalid-name
            w = jax.vmap(self.weights, 1, 1)(R)  # [n_samples, n_steps]
            da_opt = jax.vmap(jnp.average, (1, None, 1))(da, 0, w)  # [n_steps, dim_a]
            a_opt = jnp.clip(a_opt + da_opt, -1.0, 1.0)  # [n_steps, dim_a]\
            
            _, s_opt, _ = self.rollout(a_opt, env, env_state)
            
            return (a_opt, a_cov, s, s_opt, rng, r_sample, a, ret_dyna), None
        
        predicted_states = []
        if not self.scan:
            for _ in range(self.n_iterations):
                (a_opt, a_cov, s, s_opt, rng, r_sample, a, ret_dyna), _ = iteration_step((a_opt, a_cov, rng), None)
                predicted_states.append(s)
        else:
            (a_opt, a_cov, rng), _ = jax.lax.scan(
                iteration_step, (a_opt, a_cov, rng), None, length=self.n_iterations
            )
        return (a_opt, a_cov), jnp.stack(predicted_states), s_opt, r_sample, a, ret_dyna

    def get_action(self, mpc_state, a_shape):
        a_opt, _ = mpc_state
        return jnp.reshape(a_opt[0, :], a_shape)

    @partial(jax.jit, static_argnums=(0))
    def returns(self, r):
        # r: [n_steps]
        return jnp.dot(jnp.triu(jnp.ones((self.n_steps, self.n_steps))),
                    r)  # R: [n_steps]

    @partial(jax.jit, static_argnums=(0))
    def weights(self, R):  # pylint: disable=invalid-name
        # R: [n_samples]
        # R_stdzd = (R - jnp.min(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)
        # R_stdzd = R - jnp.max(R) # [n_samples] np.float32
        R_stdzd = (R - jnp.max(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)  # pylint: disable=invalid-name
        w = jnp.exp(R_stdzd / self.temperature)  # [n_samples] np.float32
        w = w/jnp.sum(w)  # [n_samples] np.float32
        return w
    
    @partial(jax.jit, static_argnums=(0,2))
    def rollout(self, actions, env, env_state):
        # actions: [n_steps, dim_a]
        # env: {.step(s, a), .reward(s)}
        # env_state: np.float32

    
        def rollout_step(env_state, a):
            a = jnp.reshape(a, env.a_shape)
            (env_state, env_var, ret_dyna) = env.step(env_state, a)
            env_var = jax.device_get(env_var)            
            r = -env_var
            return env_state, (env_state, r, ret_dyna)
        if not self.scan:
            # python equivalent of lax.scan
            scan_output = []
            for t in range(self.n_steps):
                env_state, output = rollout_step(env_state, actions[t, :])
                scan_output.append(output)
            s, r, ret_dyna = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *scan_output)
        else:
            _, (s, r) = jax.lax.scan(rollout_step, env_state, actions)
            
        return r, s, ret_dyna
    
class oneLineJaxRNG:
    def __init__(self, init_num=0) -> None:
        self.rng = jax.random.PRNGKey(init_num)
        
    def new_key(self):
        self.rng, key = jax.random.split(self.rng)
        return key





param1 = {
        # vehicle body dimensions
        'length': 4.298,  # vehicle length [m]
        'width': 1.674,  # vehicle width [m]

        # steering constraints
        's_min': -0.910,  # minimum steering angle [rad]
        's_max': 0.910,  # maximum steering angle [rad]
        'sv_min': -0.4,  # minimum steering velocity [rad/s]
        'sv_max': 0.4,  # maximum steering velocity [rad/s]

        # longitudinal constraints
        'v_min': -13.9,  # minimum velocity [m/s]
        'v_max': 45.8,  # minimum velocity [m/s]
        'v_switch': 3.0,  # switching velocity [m/s]
        'a_max': 3.5,  # maximum absolute acceleration [m/s^2]

        # masses
        'm': 1225.887,  # vehicle mass [kg]  MASS
        'm_s': 1094.542,  # sprung mass [kg]  SMASS
        'm_uf': 65.672,  # unsprung mass front [kg]  UMASSF
        'm_ur': 65.672,  # unsprung mass rear [kg]  UMASSR

        # axes distances
        'lf': 0.88392,  # distance from spring mass center of gravity to front axle [m]  LENA
        'lr': 1.50876,  # distance from spring mass center of gravity to rear axle [m]  LENB
    }

# TODO: put inside mppi_env
params = jnp.array(list(param1.values()))


class MPPIEnv():
    def __init__(self, track, n_steps, mode='st', DT=0.1) -> None:
        self.a_shape = 2

        waypoints = [
            track.raceline.ss,
            track.raceline.xs,
            track.raceline.ys,
            track.raceline.yaws,
            track.raceline.ks,
            track.raceline.vxs,
            track.raceline.axs
        ]
        self.waypoints = np.array(waypoints).T
        # self.frenet_coord = FrenetCoord(jnp.array(waypoints))
        # self.diff = self.waypoints[1:, 1:3] - self.waypoints[:-1, 1:3]
        self.waypoints_distances = np.linalg.norm(self.waypoints[1:, (1, 2)] - self.waypoints[:-1, (1, 2)], axis=1)
        # print(self.waypoints_distances)
        self.n_steps = n_steps
        self.reference = None
        self.DT = DT
        self.dlk = self.waypoints[1,0] - self.waypoints[0, 0]

        # config.load_file(config.savedir + 'config.json')
        config_norm_params = jnp.array(config.normalization_param[7:9])

        self.normalization_param = config_norm_params[:, 0]/2
        self.mode = mode
        # self.mb_dyna_pre = None
        if mode == 'ks':
            def update_fn(x, u):
                x1 = x.copy()
                Ddt = 0.05
                def step_fn(i, x0):
                    # # Forward euler
                    # return x0 + vehicle_dynamics_st_trap([x0, u]) * Ddt

                    # RK45
                    k1 = self.vehicle_dynamics_ks(x0, u)
                    k2 = self.vehicle_dynamics_ks(x0 + k1 * 0.5 * Ddt, u)
                    k3 = self.vehicle_dynamics_ks(x0 + k2 * 0.5 * Ddt, u)
                    k4 = self.vehicle_dynamics_ks(x0 + k3 * Ddt, u)
                    return x0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * Ddt
                    
                x1 = jax.lax.fori_loop(0, int(self.DT/Ddt), step_fn, x1)
                return (x1, 0, x1-x)
            self.update_fn = update_fn

        if mode == 'st':
            def update_fn(x, u):
                x1 = x.copy()
                Ddt = 0.05
                def step_fn(i, x0):
                    # # Forward euler
                    # return x0 + vehicle_dynamics_st_trap([x0, u]) * Ddt

                    # RK45
                    k1 = self.vehicle_dynamics_st(x0, u)
                    k2 = self.vehicle_dynamics_st(x0 + k1 * 0.5 * Ddt, u)
                    k3 = self.vehicle_dynamics_st(x0 + k2 * 0.5 * Ddt, u)
                    k4 = self.vehicle_dynamics_st(x0 + k3 * Ddt, u)
                    return x0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * Ddt
                    
                x1 = jax.lax.fori_loop(0, int(self.DT/Ddt), step_fn, x1)
                return (x1, 0, x1-x)
            self.update_fn = update_fn
    
    ## Constraints handling
    def accl_constraints(self, vel, accl, v_switch, a_max, v_min, v_max):
        """
        Acceleration constraints, adjusts the acceleration based on constraints

            Args:
                vel (float): current velocity of the vehicle
                accl (float): unconstraint desired acceleration
                v_switch (float): switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
                a_max (float): maximum allowed acceleration
                v_min (float): minimum allowed velocity
                v_max (float): maximum allowed velocity

            Returns:
                accl (float): adjusted acceleration
        """

        # positive accl limit
        # if vel > v_switch:
        #     pos_limit = a_max*v_switch/vel
        # else:
        #     pos_limit = a_max
        pos_limit = jax.lax.select(vel > v_switch, a_max*v_switch/vel, a_max)

        # accl limit reached?
        # accl = jax.lax.select(vel <= v_min and accl <= 0, 0., accl)
        # accl = jax.lax.select(vel >= v_max and accl >= 0, 0., accl)
        accl = jax.lax.select(jnp.all(jnp.asarray([vel <= v_min, accl <= 0])), 0., accl)
        accl = jax.lax.select(jnp.all(jnp.asarray([vel >= v_max, accl >= 0])), 0., accl)
        
        accl = jax.lax.select(accl <= -a_max, -a_max, accl)
        accl = jax.lax.select(accl >= pos_limit, pos_limit, accl)

        return accl
    
    def steering_constraint(self, steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
        """
        Steering constraints, adjusts the steering velocity based on constraints

            Args:
                steering_angle (float): current steering_angle of the vehicle
                steering_velocity (float): unconstraint desired steering_velocity
                s_min (float): minimum steering angle
                s_max (float): maximum steering angle
                sv_min (float): minimum steering velocity
                sv_max (float): maximum steering velocity

            Returns:
                steering_velocity (float): adjusted steering velocity
        """

        # constraint steering velocity
        steering_velocity = jax.lax.select(jnp.all(jnp.asarray([steering_angle <= s_min, steering_velocity <= 0])), 0., steering_velocity)
        steering_velocity = jax.lax.select(jnp.all(jnp.asarray([steering_angle >= s_max, steering_velocity >= 0])), 0., steering_velocity)
        # steering_velocity = jax.lax.select(steering_angle >= s_max and steering_velocity >= 0, 0., steering_velocity)
        steering_velocity = jax.lax.select(steering_velocity <= sv_min, sv_min, steering_velocity)
        steering_velocity = jax.lax.select(steering_velocity >= sv_max, sv_max, steering_velocity)
        # if (steering_angle <= s_min and steering_velocity <= 0) or (steering_angle >= s_max and steering_velocity >= 0):
        #     steering_velocity = 0.
        # elif steering_velocity <= sv_min:
        #     steering_velocity = sv_min
        # elif steering_velocity >= sv_max:
        #     steering_velocity = sv_max

        return steering_velocity
    ##Vehicle Dynamics models
    def vehicle_dynamics_ks(self, x, u_init, lf=0.15875, lr=0.17145):
        """
        Single Track Kinematic Vehicle Dynamics.

            Args:
                x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5)
                    x1: x position in global coordinates
                    x2: y position in global coordinates
                    x3: steering angle of front wheels
                    x4: velocity in x direction
                    x5: yaw angle
                u (numpy.ndarray (2, )): control input vector (u1, u2)
                    u1: steering angle velocity of front wheels
                    u2: longitudinal acceleration

            Returns:
                f (numpy.ndarray): right hand side of differential equations
        """
        # wheelbase
        lwb = lf + lr

        # constraints
        s_min = params[2]  # minimum steering angle [rad]
        s_max = params[3]  # maximum steering angle [rad]
        # longitudinal constraints
        v_min = params[6]  # minimum velocity [m/s]
        v_max = params[7] # minimum velocity [m/s]
        sv_min = params[4] # minimum steering velocity [rad/s]
        sv_max = params[5] # maximum steering velocity [rad/s]
        v_switch = params[8]  # switching velocity [m/s]
        a_max = params[9] # maximum absolute acceleration [m/s^2]

        # constraints
        u = jnp.array([self.steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), self.accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

        # system dynamics
        f = jnp.array([x[3]*jnp.cos(x[4]),
            x[3]*jnp.sin(x[4]), 
            u[0],
            u[1],
            x[3]/lwb*jnp.tan(x[2])])
        return f
    
    def vehicle_dynamics_st(self, x, u_init, mu=1.0489, C_Sf=4.718, C_Sr=5.4562, 
                        lf=0.15875, lr=0.17145, h=0.074, m=3.74, I=0.04712):
        """
        Single Track Dynamic Vehicle Dynamics.

            Args:
                x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5, x6, x7)
                    x1: x position in global coordinates
                    x2: y position in global coordinates
                    x3: steering angle of front wheels
                    x4: velocity in x direction
                    x5: yaw angle
                    x6: yaw rate
                    x7: slip angle at vehicle center
                u (numpy.ndarray (2, )): control input vector (u1, u2)
                    u1: steering angle velocity of front wheels
                    u2: longitudinal acceleration

            Returns:
                f (numpy.ndarray): right hand side of differential equations
        """
        # gravity constant m/s^2
        g = 9.81
        params = jnp.array(list(param1.values()))
        
        # steering constraints
        s_min = params[2]  # minimum steering angle [rad]
        s_max = params[3]  # maximum steering angle [rad]
        # longitudinal constraints
        v_min = params[6]  # minimum velocity [m/s]
        v_max = params[7] # minimum velocity [m/s]
        sv_min = params[4] # minimum steering velocity [rad/s]
        sv_max = params[5] # maximum steering velocity [rad/s]
        v_switch = params[8]  # switching velocity [m/s]
        a_max = params[9] # maximum absolute acceleration [m/s^2]

        # constraints
        u = jnp.array([self.steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), self.accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

        # system dynamics
        f = jnp.array([x[3]*jnp.cos(x[6] + x[4]),
            x[3]*jnp.sin(x[6] + x[4]),
            u[0],
            u[1],
            x[5],
            -mu*m/(x[3]*I*(lr+lf))*(lf**2*C_Sf*(g*lr-u[1]*h) + lr**2*C_Sr*(g*lf + u[1]*h))*x[5] \
                +mu*m/(I*(lr+lf))*(lr*C_Sr*(g*lf + u[1]*h) - lf*C_Sf*(g*lr - u[1]*h))*x[6] \
                +mu*m/(I*(lr+lf))*lf*C_Sf*(g*lr - u[1]*h)*x[2],
            (mu/(x[3]**2*(lr+lf))*(C_Sr*(g*lf + u[1]*h)*lr - C_Sf*(g*lr - u[1]*h)*lf)-1)*x[5] \
                -mu/(x[3]*(lr+lf))*(C_Sr*(g*lf + u[1]*h) + C_Sf*(g*lr-u[1]*h))*x[6] \
                +mu/(x[3]*(lr+lf))*(C_Sf*(g*lr-u[1]*h))*x[2]])

        return f
                
    # @partial(jax.jit, static_argnums=(0,))
    def step(self, x, u):
        return self.update_fn(x, u * self.normalization_param)
        # return self.update_fn(x, u)
    
    
    @partial(jax.jit, static_argnums=(0,))
    def reward_fn(self, s, reference):
        

        xy_cost = -jnp.linalg.norm(reference[1:, :2] - s[:, :2], ord=1, axis=1)
        # vel_cost = -jnp.linalg.norm(reference[1:, 5] - s[:, 3])
        yaw_cost = -jnp.abs(jnp.sin(reference[1:, 3]) - jnp.sin(s[:, 4])) - \
            jnp.abs(jnp.cos(reference[1:, 3]) - jnp.cos(s[:, 4]))
        
        # adding terminal cost( useful sometimes)
        terminal_cost = -jnp.linalg.norm(reference[-1, :2] - s[-1, :2])
        return 15*xy_cost + 20*yaw_cost  
            
    
    # @partial(jax.jit, static_argnums=(0,))
    def reward(self, x):
        return 0
    
    def get_refernece_traj(self, state, target_speed=None, vind=5, speed_factor=1.0):
        _, dist, _, _, ind = nearest_point(np.array([state[0], state[1]]), 
                                           self.waypoints[:, (1, 2)].copy())
        
        if target_speed is None:
            # speed = self.waypoints[ind, vind] * speed_factor
            speed = np.minimum(self.waypoints[ind, vind] * speed_factor, 20.)
            # speed = state[3]
        else:
            speed = target_speed
        
        # if ind < self.waypoints.shape[0] - self.n_steps:
        #     speeds = self.waypoints[ind:ind+self.n_steps, vind]
        # else:
        speeds = np.ones(self.n_steps) * speed
        
        reference = self.get_reference_trajectory(speeds, dist, ind, 
                                            self.waypoints.copy(), int(self.n_steps),
                                            self.waypoints_distances.copy(), DT=self.DT)
        orientation = state[4]
        angle_thres = 5.0
        reference[3, :][reference[3, :] - orientation > angle_thres] = np.abs(
            reference[3, :][reference[3, :] - orientation > angle_thres] - (2 * np.pi))
        reference[3, :][reference[3, :] - orientation < -angle_thres] = np.abs(
            reference[3, :][reference[3, :] - orientation < -angle_thres] + (2 * np.pi))
        
        # reference[2] = np.where(reference[2] - speed > 5.0, speed + 5.0, reference[2])
        self.reference = reference.T
        return reference.T, ind
    
    def get_reference_trajectory(self, predicted_speeds, dist_from_segment_start, idx, 
                                waypoints, n_steps, waypoints_distances, DT):
        s_relative = np.zeros((n_steps + 1,))
        s_relative[0] = dist_from_segment_start
        s_relative[1:] = predicted_speeds * DT
        s_relative = np.cumsum(s_relative)

        waypoints_distances_relative = np.cumsum(np.roll(waypoints_distances, -idx))

        index_relative = np.int_(np.ones((n_steps + 1,)))
        for i in range(n_steps + 1):
            index_relative[i] = (waypoints_distances_relative <= s_relative[i]).sum()
        index_absolute = np.mod(idx + index_relative, waypoints.shape[0] - 1)

        segment_part = s_relative - (
                waypoints_distances_relative[index_relative] - waypoints_distances[index_absolute])

        t = (segment_part / waypoints_distances[index_absolute])
        # print(np.all(np.logical_and((t < 1.0), (t > 0.0))))

        position_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, (1, 2)] -
                            waypoints[index_absolute][:, (1, 2)])
        orientation_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 3] -
                                waypoints[index_absolute][:, 3])
        speed_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 5] -
                        waypoints[index_absolute][:, 5])

        interpolated_positions = waypoints[index_absolute][:, (1, 2)] + (t * position_diffs.T).T
        interpolated_orientations = waypoints[index_absolute][:, 3] + (t * orientation_diffs)
        interpolated_orientations = (interpolated_orientations + np.pi) % (2 * np.pi) - np.pi
        interpolated_speeds = waypoints[index_absolute][:, 5] + (t * speed_diffs)
        
        reference = np.array([
            # Sort reference trajectory so the order of reference match the order of the states
            interpolated_positions[:, 0],
            interpolated_positions[:, 1],
            interpolated_speeds,
            interpolated_orientations,
            # Fill zeros to the rest so number of references mathc number of states (x[k] - ref[k])
            np.zeros(len(interpolated_speeds)),
            np.zeros(len(interpolated_speeds)),
            np.zeros(len(interpolated_speeds))
        ])
        return reference




class MPPIPlanner():
    

    def __init__(
        self,
        track=None,

        debug=False,
    ):
        self.track = track
        self.debug = debug
        self.n_steps = 8
        self.n_samples = 128
        self.jRNG = oneLineJaxRNG(1337)
        self.DT = 0.1
        self.mppi_env_ks = MPPIEnv(self.track, self.n_steps, mode = 'ks', DT= self.DT)
        self.mppi_env_st = MPPIEnv(self.track, self.n_steps, mode = 'st', DT= self.DT)
        self.mppi = MPPI(n_iterations = 5, n_steps = self.n_steps,
                         n_samples = self.n_samples, a_noise = 1.0, scan = False)
        
        self.a_opt = None
        self.a_cov = None
        self.mppi_state = None
        
        self.mppi_path = None
        self.target_vel = 5.0
        config_norm_params = jnp.array(config.normalization_param[7:9])

        self.norm_param = config_norm_params[:, 0]/2
        self.init_state()
        self.ref_path = None
        self.laptime = 0

        self.init_mppi_compile()
        
    
    def init_state(self):
        self.mppi_state =  self.mppi.init_state(self.mppi_env_ks.a_shape, self.jRNG.new_key() )
        self.a_opt = self.mppi_state[0]
    

    def init_mppi_compile(self):


        init_x = self.track.raceline.xs[0]
        init_y = self.track.raceline.ys[0]
        init_yaw = self.track.raceline.yaws[0]

        state_ks = np.array([init_x, init_y, 0, 0, init_yaw], dtype=np.float32)

        _,_ = self.mppi_env_ks.get_refernece_traj(state_ks, target_speed = self.target_vel,  vind = 5, speed_factor= 1)
        _, _, _, _, _,_ = self.mppi.update(self.mppi_state, self.mppi_env_ks, state_ks.copy(), self.jRNG.new_key())
        
        
        state_st = np.array([init_x, init_y, 0, 0, init_yaw, 0, 0], dtype=np.float32)

        _,_ = self.mppi_env_st.get_refernece_traj(state_st, target_speed = self.target_vel,  vind = 5, speed_factor= 1)
        _, _, _, _, _,_ = self.mppi.update(self.mppi_state, self.mppi_env_st, state_st.copy(), self.jRNG.new_key())

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        waypoints = [
            self.track.raceline.ss,
            self.track.raceline.xs,
            self.track.raceline.ys,
            self.track.raceline.yaws,
            self.track.raceline.ks,
            self.track.raceline.vxs,
            self.track.raceline.axs
        ]
        waypoints = np.array(waypoints)
        points = np.array(waypoints).T[:, 1:3]
        e.render_closed_lines(points, color=(128, 0, 0), size=1)

    def render_local_plan(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        if self.ref_path is not None:
            points = self.ref_path[:, :2]
            e.render_lines(points, color=(0, 128, 0), size=2)

    def render_mppi_sol(self, e):
        """
        Callback to render the mppi sol.
        """
        if self.mppi_path is not None:
            opt_path = np.array(self.mppi_path)
            e.render_lines(opt_path[:, :2], color=(0, 0, 128), size=2)

    def plan(self, obs):
        """
        Planner plan function for MPPI, returns acutation based on current state

        Args:
            pose_x (float): current vehicle x position
            pose_y (float): current vehicle y position
            delta (float): current vehicle steering angle
            linear_vel_x (float): current vehicle velocity
            pose_theta (float): current vehicle heading angle
            ang_vel_z (float): current vehicle yawrate
            beta (float): current vehicle sideslip angle
            

        Returns:
            steering_rate (float):  commanded vehicle steering rate
            accl (float): commanded vehicle acceleration
        """
         
        state_x = obs['pose_x']
        state_y = obs['pose_y']
        delta = obs['delta']
        yaw = obs['pose_theta']
        v = obs['linear_vel_x']
        yawrate = obs['ang_vel_z']
        beta = obs['beta']

        if v < params[8]:
            state = np.array([state_x, state_y, delta, v, yaw])

            ref_traj,_ = self.mppi_env_ks.get_refernece_traj(state, target_speed = self.target_vel,  vind = 5, speed_factor= 1)
            self.ref_path = ref_traj
            self.mppi_state, sampled_traj, s_opt, _, _,_ = self.mppi.update(self.mppi_state, self.mppi_env_ks, state.copy(), self.jRNG.new_key())
        
        else:
            state = np.array([state_x, state_y, delta, v, yaw, yawrate, beta])

            ref_traj,_ = self.mppi_env_st.get_refernece_traj(state, target_speed = self.target_vel,  vind = 5, speed_factor= 1)
            self.ref_path = ref_traj
            # s_opt = None
            self.mppi_state, sampled_traj, s_opt, _, _,_ = self.mppi.update(self.mppi_state, self.mppi_env_st, state.copy(), self.jRNG.new_key())


        self.mppi_path = s_opt
        # print(s_opt.shape

        if self.debug:
            # plots sampled trajectories every 40 steps
            if(self.laptime %400 == 0):
                for i in range(128):
                    plt.plot(sampled_traj[0][i][:, 0], sampled_traj[0][i][:, 1])
                plt.show()

        self.laptime+=1
        a_opt = self.mppi_state[0]
        control = a_opt[0]
        scaled_control = np.multiply(self.norm_param, control)

        steerv = scaled_control[0]
        accl = scaled_control[1]

        if obs["linear_vel_x"] < 0.1:
            steer, accl = 0.0, 3.0
            return steer, accl
        # return sampled_traj
        # steer_angle = delta + steerv*self.DT
        # speed = v + accl*self.DT
        return steerv, accl
