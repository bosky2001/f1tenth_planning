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
    
class Config(ConfigYAML):
    exp_name = 'mppi_tracking'
    segment_length = 1
    sim_time_step = 0.1
    render = 1
    kmonitor_enable = 1

    max_lap = 300
    random_seed = None    

    n_steps = 10
    # n_samples = 1024
    n_samples = 128
    n_iterations = 1
    control_dim = 2
    control_sample_noise = [1.0, 1.0]
    state_predictor = 'ks'
    half_width = 4
    
    adaptive_covariance = False
    # init_noise = [5e-3, 5e-3, 5e-3] # control_vel, control_steering, state 
    init_noise = [0, 0, 0] # control_vel, control_steering, state


class MPPI():
    """An MPPI based planner."""
    def __init__(self, config, env, jrng, 
                 temperature=0.01, damping=0.001):
        self.config = config
        self.n_iterations = config.n_iterations
        self.n_steps = config.n_steps
        self.n_samples = config.n_samples
        self.temperature = temperature
        self.damping = damping
        self.a_std = jnp.array(config.control_sample_noise)
        self.scan = False  # whether to use jax.lax.scan instead of python loop
        self.adaptive_covariance = config.adaptive_covariance
        self.a_shape = config.control_dim
        self.env = env
        self.jrng = jrng
        self.init_state(self.env, self.a_shape)
        self.accum_matrix = jnp.triu(jnp.ones((self.n_steps, self.n_steps)))

    def init_state(self, env, a_shape):
        # uses random as a hack to support vmap
        # we should find a non-hack approach to initializing the state
        dim_a = jnp.prod(a_shape)  # np.int32
        self.env = env
        self.a_opt = 0.0*jax.random.uniform(self.jrng.new_key(), shape=(self.n_steps,
                                                dim_a))  # [n_steps, dim_a]
        # a_cov: [n_steps, dim_a, dim_a]
        if self.adaptive_covariance:
            # note: should probably store factorized cov,
            # e.g. cholesky, for faster sampling
            self.a_cov = (self.a_std**2)*jnp.tile(jnp.eye(dim_a), (self.n_steps, 1, 1))
            self.a_cov_init = self.a_cov.copy()
        else:
            self.a_cov = None
            self.a_cov_init = self.a_cov
    
    @partial(jax.jit, static_argnums=(0))
    def iteration_step(self, a_opt, a_cov, rng_da, env_state, reference_traj):
        self.a_opt = jnp.concatenate([self.a_opt[1:, :],
                                jnp.expand_dims(jnp.zeros((self.a_shape,)),
                                                axis=0)])  # [n_steps, a_shape]
        if self.adaptive_covariance:
            a_cov = jnp.concatenate([a_cov[1:, :],
                                    jnp.expand_dims((self.a_std**2)*jnp.eye(self.a_shape),
                                                    axis=0)])
        
        da = jax.random.truncated_normal(
            rng_da,
            -jnp.ones_like(a_opt) * self.a_std - a_opt,
            jnp.ones_like(a_opt) * self.a_std - a_opt,
            shape=(self.n_samples, self.n_steps, self.a_shape)
        )  # [n_samples, n_steps, dim_a]

        actions = jnp.clip(jnp.expand_dims(a_opt, axis=0) + da, -1.0, 1.0)
        _, states = jax.vmap(self.rollout, in_axes=(0, None))(
            actions, env_state
        )
        reward = jax.vmap(self.env.reward_fn, in_axes=(0, None))(
            states, reference_traj
        ) # [n_samples, n_steps]
        
        R = jax.vmap(self.returns)(reward) # [n_samples, n_steps], pylint: disable=invalid-name
        w = jax.vmap(self.weights, 1, 1)(R)  # [n_samples, n_steps]
        da_opt = jax.vmap(jnp.average, (1, None, 1))(da, 0, w)  # [n_steps, dim_a]
        a_opt = jnp.clip(a_opt + da_opt, -1.0, 1.0)  # [n_steps, dim_a]
        return a_opt, a_cov, states


    def update(self, env_state, reference_traj):
        for _ in range(self.n_iterations):
            self.a_opt, self.a_cov, self.states = self.iteration_step(self.a_opt, self.a_cov, self.jrng.new_key(), env_state, reference_traj)
            if self.config.render:
                _, self.s_opt = self.rollout(self.a_opt, env_state)
            else:
                self.s_opt = self.states[0]
        self.sampled_states = self.states


    @partial(jax.jit, static_argnums=(0))
    def returns(self, r):
        # r: [n_steps]
        return jnp.dot(self.accum_matrix, r)  # R: [n_steps]

    @partial(jax.jit, static_argnums=(0))
    def weights(self, R):  # pylint: disable=invalid-name
        # R: [n_samples]
        # R_stdzd = (R - jnp.min(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)
        # R_stdzd = R - jnp.max(R) # [n_samples] np.float32
        R_stdzd = (R - jnp.max(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)  # pylint: disable=invalid-name
        w = jnp.exp(R_stdzd / self.temperature)  # [n_samples] np.float32
        w = w/jnp.sum(w)  # [n_samples] np.float32
        return w
    
    @partial(jax.jit, static_argnums=0)
    def rollout(self, actions, env_state):
        """
        # actions: [n_steps, a_shape]
        # env: {.step(states, actions), .reward(states)}
        # env_state: np.float32
        # actions: # a_0, ..., a_{n_steps}. [n_steps, a_shape]
        # states: # s_1, ..., s_{n_steps+1}. [n_steps, env_state_shape]
        """
    
        def rollout_step(env_state, actions):
            actions = jnp.reshape(actions, self.env.a_shape)
            (env_state, env_var, mb_dyna) = self.env.step(env_state, actions)
            reward = self.env.reward(env_state)
            return env_state, (env_state, reward)
        # if not self.scan:
        #     # python equivalent of lax.scan (faster without scan)
        scan_output = []
        for t in range(self.n_steps):
            env_state, output = rollout_step(env_state, actions[t, :])
            scan_output.append(output)
        states, reward = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *scan_output)
        # else:
        #     _, (s, r) = jax.lax.scan(rollout_step, env_state, actions)
            
        return reward, states
    
class oneLineJaxRNG:
    def __init__(self, init_num=0) -> None:
        self.rng = jax.random.PRNGKey(init_num)
        
    def new_key(self):
        self.rng, key = jax.random.split(self.rng)
        return key


class MPPIEnv():
    def __init__(self, track, n_steps, normalization_param, mode='st', DT=0.1, env_config=None,
                 dyna_model=None, jrng=None) -> None:
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

        self.diff = self.waypoints[1:, 1:3] - self.waypoints[:-1, 1:3]
        self.waypoints_distances = np.linalg.norm(self.waypoints[1:, (1, 2)] - self.waypoints[:-1, (1, 2)], axis=1)
        self.n_steps = n_steps
        self.reference = None
        self.DT = DT
        
        self.env_config = env_config
        self.normalization_param = normalization_param
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
    def vehicle_dynamics_ks(self, x, u_init):
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
        

        params = self.env_config


        # wheelbase
        lwb = params["lf"] + params["lr"]
        # constraints
        s_min = params["s_min"]  # minimum steering angle [rad]
        s_max = params["s_max"]  # maximum steering angle [rad]
        # longitudinal constraints
        v_min = params["v_min"]  # minimum velocity [m/s]
        v_max = params["v_max"] # minimum velocity [m/s]
        sv_min = params["sv_min"] # minimum steering velocity [rad/s]
        sv_max = params["sv_max"] # maximum steering velocity [rad/s]
        v_switch = params["v_switch"]  # switching velocity [m/s]
        a_max = params["a_max"] # maximum absolute acceleration [m/s^2]

        # constraints
        u = jnp.array([self.steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), self.accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

        # system dynamics
        f = jnp.array([x[3]*jnp.cos(x[4]),
            x[3]*jnp.sin(x[4]), 
            u[0],
            u[1],
            x[3]/lwb*jnp.tan(x[2])])
        return f
    
    def vehicle_dynamics_st(self, x, u_init):
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
        params = self.env_config

        lf = params["lf"]
        lr = params["lr"]
        m = params["m"]
        mu = params["mu"]
        I = params["I"]
        h = params["h"]
        C_Sf = params["C_Sf"]
        C_Sr = params["C_Sr"]
        # constraints
        s_min = params["s_min"]  # minimum steering angle [rad]
        s_max = params["s_max"]  # maximum steering angle [rad]
        # longitudinal constraints
        v_min = params["v_min"]  # minimum velocity [m/s]
        v_max = params["v_max"] # minimum velocity [m/s]
        sv_min = params["sv_min"] # minimum steering velocity [rad/s]
        sv_max = params["sv_max"] # maximum steering velocity [rad/s]
        v_switch = params["v_switch"]  # switching velocity [m/s]
        a_max = params["a_max"] # maximum absolute acceleration [m/s^2]

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
        return self.update_fn(x, u * self.normalization_param[0, 7:9]/2)
        # return self.update_fn(x, u)
    
    
    @partial(jax.jit, static_argnums=(0,))
    def reward_fn(self, s, reference):
        """
        reward function for the state s with respect to the reference trajectory
        """
        # gamma = 0.8
        # gamma_vec = jnp.array([gamma ** i for i in range(reference.shape[0] - 1)])
        xy_cost = -jnp.linalg.norm(reference[1:, :2] - s[:, :2], ord=1, axis=1)
        vel_cost = -jnp.linalg.norm(reference[1:, 3] - s[:, 3])
        yaw_cost = -jnp.abs(jnp.sin(reference[1:, 4]) - jnp.sin(s[:, 4])) - \
            jnp.abs(jnp.cos(reference[1:, 4]) - jnp.cos(s[:, 4]))
        

        # terminal cost
        terminal_cost = -jnp.linalg.norm(reference[-1, :2] - s[-1, :2])
        return xy_cost
        return 15*xy_cost  + 50*terminal_cost+ 25*yaw_cost
    
    # @partial(jax.jit, static_argnums=(0,))
    def reward(self, x):
        return 0
    
    def get_refernece_traj_sequential(self, state, steps=10, waypoints=None, DT=0.05):
        if waypoints is None:
            waypoints = self.waypoints
        _, dist, _, _, ind = nearest_point(np.array([state[0], state[1]]), 
                                           waypoints[:, (0, 1)].copy())
        
        
        interval = 5
        inds = np.arange(ind, ind + interval * steps + 1, interval)
        inds[np.where(inds >= waypoints.shape[0])] -= waypoints.shape[0]
        reference = waypoints[inds.astype(np.int16)]
        self.reference = reference
        return reference, inds
    
    def get_refernece_traj_v(self, state, steps=10, waypoints=None, DT=0.05):
        if waypoints is None:
            waypoints = self.waypoints
        _, dist, _, _, ind = nearest_point(np.array([state[0], state[1]]), 
                                           waypoints[:, (0, 1)].copy())
        travel = state[3] * DT * steps
        _, dist, _, _, ind_future = nearest_point(np.array([state[0] + travel * np.cos(state[4]), 
                                                     state[1] + travel * np.sin(state[4])]), 
                                           waypoints[:, (0, 1)].copy())
        if ind_future < ind:
            ind_future += waypoints.shape[0]
        interval = (ind_future - ind) // 10
        # print('ind_future', ind, ind_future, interval)
        inds = np.arange(ind, ind + interval * steps + 1, interval)
        inds[np.where(inds >= waypoints.shape[0])] -= waypoints.shape[0]
        reference = waypoints[inds.astype(np.int16)]
        self.reference = reference
        return reference, inds
    
    def get_refernece_traj(self, state, target_speed=None, vind=5, speed_factor=1.0):
        _, dist, _, _, ind = nearest_point(np.array([state[0], state[1]]), 
                                           self.waypoints[:, (1, 2)].copy())
        
        if target_speed is None:
            # speed = self.waypoints[ind, vind] * speed_factor
            # speed = np.minimum(self.waypoints[ind, vind] * speed_factor, 20.)
            speed = state[3]
        else:
            speed = target_speed
        
        # if ind < self.waypoints.shape[0] - self.n_steps:
        #     speeds = self.waypoints[ind:ind+self.n_steps, vind]
        # else:
        print('speed', speed)
        speeds = np.ones(self.n_steps) * speed
        
        reference = get_reference_trajectory(speeds, dist, ind, 
                                            self.waypoints.copy(), int(self.n_steps),
                                            self.waypoints_distances.copy(), DT=self.DT)
        orientation = state[4]
        reference[3, :][reference[3, :] - orientation > 5] = np.abs(
            reference[3, :][reference[3, :] - orientation > 5] - (2 * np.pi))
        reference[3, :][reference[3, :] - orientation < -5] = np.abs(
            reference[3, :][reference[3, :] - orientation < -5] + (2 * np.pi))
        
        # reference[2] = np.where(reference[2] - speed > 5.0, speed + 5.0, reference[2])
        self.reference = reference.T
        return reference.T, ind

class MPPIPlanner():
    

    def __init__(
        self,
        track=None,

        debug=False,
        env_config = None
    ):
        self.track = track
        self.debug = debug
        self.n_steps = 8
        self.n_samples = 128
        self.jRNG = oneLineJaxRNG(1337)
        self.DT = 0.1
        
        self.config = Config()
        self.config.load_file("../../f1tenth_planning/control/kinematic_mppi/config.yaml")

        self.normalization_param = np.array(self.config.normalization_param).T

        # print(self.config.normalization_param.shape)
        self.mppi_env_ks = MPPIEnv(self.track, self.config.n_steps, self.normalization_param, mode='ks', DT=self.config.sim_time_step, env_config=env_config)

        self.mppi_env_st = MPPIEnv(self.track, self.config.n_steps, self.normalization_param, mode='st', DT=self.config.sim_time_step, env_config=env_config)
        
        # self.mppi_env_st = MPPIEnv(self.track, self.n_steps, mode = 'st', DT= self.DT)
        self.mppi_ks = MPPI(self.config, self.mppi_env_ks, self.jRNG) 
        self.mppi_st = MPPI(self.config, self.mppi_env_st, self.jRNG) 

        
        self.mppi_path = None
        self.mppi_samples = None
        self.target_vel = 3
        config_norm_params = jnp.array(self.config.normalization_param[7:9])

        self.norm_param = config_norm_params[:, 0]/2
        # self.init_state()
        self.ref_path = None
        self.laptime = 0

        # self.init_mppi_compile()

    
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
    
    def render_mppi_samples(self, e):

        if self.mppi_samples is not None:
            for i in range(self.n_samples):
                sampled_path = np.array(self.mppi_samples[i])
                e.render_lines(sampled_path[:, :2], color=(128, 128, 0), size=2)

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

        
        # print("hi2")
        if v < 4.0:

            state = np.array([state_x, state_y, delta, v, yaw])

            ref_traj,_ = self.mppi_env_ks.get_refernece_traj(state, target_speed = self.target_vel,  vind = 5, speed_factor= 1)
            self.ref_path = ref_traj
            # print("hi")
            self.mppi_ks.update(jnp.asarray(state), self.ref_path.copy())

            self.mppi_path = self.mppi_ks.s_opt
            self.mppi_samples = self.mppi_ks.sampled_states
            a_opt = self.mppi_ks.a_opt[0]

            
        
        else:
            state = np.array([state_x, state_y, delta, v, yaw, yawrate, beta])

            ref_traj,_ = self.mppi_env_st.get_refernece_traj(state, target_speed = self.target_vel,  vind = 5, speed_factor= 1)
            self.ref_path = ref_traj
            # print("hi")
            self.mppi_st.update(jnp.asarray(state), self.ref_path.copy())

            self.mppi_path = self.mppi_st.s_opt
            self.mppi_samples = self.mppi_st.sampled_states
            a_opt = self.mppi_st.a_opt[0]
              
        # print(sampled_traj.shape)
        # if(self.laptime %400 == 0):
        #     for i in range(128):
        #         plt.plot(sampled_traj[i][:, 0], sampled_traj[i][:, 1])
        #     plt.show()

        self.laptime+=1
        
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
        print(steerv, accl)
        return steerv, accl
        # return 0.0, 1.0

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


# @njit(cache=True)
def get_reference_trajectory(predicted_speeds, dist_from_segment_start, idx, 
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