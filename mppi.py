import jax
import jax.numpy as jnp
from f1tenth_planning.utils.utils import nearest_point
from dataclasses import dataclass, field
import numpy as np
import math
from functools import partial
import matplotlib.pyplot as plt
import matplotlib
@dataclass
class mppi_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering speed, acceleration]
    TK: int = 8  # finite time horizon length kinematic
    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 100.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 100.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, delta, v, yaw, yaw-rate, beta]
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0])
    )  # final state error matrix, penalty  for the final state constraints: [x, y, delta, v, yaw, yaw-rate, beta]
    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 0.03  # dist step [m] kinematic
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 6.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]

@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0



class MPPIPlanner:
    def __init__(
        self,
        track,
        config=mppi_config(),
        params=np.array(
            [3.74, 0.15875, 0.17145, 0.074, 4.718, 5.4562, 0.04712, 1.0489]
        )
    ):  
        self.waypoints = [
            track.raceline.xs,
            track.raceline.ys,
            track.raceline.yaws,
            track.raceline.vxs,
        ]
        self.vehicle_params = params
        self.config = config
        self.ref_path = None
        self.samples = 1000
        self.u_prev = np.zeros((self.config.TK, 2))
        self.drawn_waypoints = []

        # MPPI weights stuff
        self.n_iterations = 5
        self.n_samples = 100
        self.n_steps = 8
        self.dim_a = jnp.prod(2)
        self.adaptive_covariance = False
        self.initial_state = None
        self.a_std = 0.5
        self.damping = 0.001
        self.temperature = 0.01

        self.sampled_trajs = np.zeros((self.samples, self.config.TK, 4))
        self.key = jax.random.PRNGKey(1337)
        self.rollouts = None
        self.dt = self.config.DTK
        self.u_prev = jnp.zeros((8,2))
        self.target_state = jnp.array([-5, -5, 3, 2.5])
        self.Q = jnp.diag(jnp.array([10,10, 1, 1]))
        self.mppi_rollout = None
        self.a_opt, self.a_cov = self.init_state(2, self.key)
    
    def init_state(self, a_shape, rng):
        # uses random as a hack to support vmap
        # we should find a non-hack approach to initializing the state
        dim_a = jnp.prod(a_shape)  # np.int32
        a_opt = jnp.array([0, 3.0]) * jnp.ones((8,2))  # [n_steps, dim_a]
        # a_cov: [n_steps, dim_a, dim_a]
        if self.adaptive_covariance:
            # note: should probably store factorized cov,
            # e.g. cholesky, for faster sampling
            a_cov = (self.a_std**2)*jnp.tile(jnp.eye(dim_a), (self.n_steps, 1, 1))
        else:
            a_cov = None
        return (a_opt, a_cov)
    
    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        points = np.array(self.waypoints).T[:, :2]
        e.render_closed_lines(points, color=(128, 0, 0), size=1)

    def render_local_plan(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        if self.ref_path is not None:
            # points = self.ref_path[:2 ,:4].T
            points = self.ref_path[:2].T
            e.render_lines(points, color=(0, 128, 0), size=2)
        else:
            print("balls")
    
    def render_sampled_trajs(self, e):
        """
        Drawing the sampled trajectory from MPPI
        """
        # if self.rollouts is not None:
        #     for i in range(self.n_samples):
        #         points = np.asarray(self.rollouts[i][:, :2])
        #         e.render_lines(points, color=(128, 128, 0), size=2)

        if self.mppi_rollout is not None:
            points = np.asarray(self.mppi_rollout[:, :2])
            e.render_lines(points, color=(0,128,0), size=2)
            
            
    def calc_ref_trajectory_kinematic(self, state, cx, cy, cyaw, sp):
        """
        calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        _, _, _, ind = nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        travel = abs(state.v) * self.config.DTK
        dind = travel / self.config.dlk
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]
        cyaw[cyaw - state.yaw > 4.5] = np.abs(
            cyaw[cyaw - state.yaw > 4.5] - (2 * np.pi)
        )
        cyaw[cyaw - state.yaw < -4.5] = np.abs(
            cyaw[cyaw - state.yaw < -4.5] + (2 * np.pi)
        )
        ref_traj[3, :] = cyaw[ind_list]

        return ref_traj
    
    def plan(self, states, waypoints=None):
        """
        Planner plan function overload for Pure Pursuit, returns acutation based on current state

        Args:
            pose_x (float): current vehicle x position
            pose_y (float): current vehicle y position
            pose_theta (float): current vehicle heading angle
            lookahead_distance (float): lookahead distance to find next waypoint to track
            waypoints (numpy.ndarray [N x 4], optional): list of dynamic waypoints to track, columns are [x, y, velocity, heading]

        Returns:
            speed (float): commanded vehicle longitudinal velocity
            steering_angle (float):  commanded vehicle steering angle
        """

        if states["linear_vel_x"] < 0.1:
            steer, accl = 0.0, self.config.MAX_ACCEL
            return steer, accl

        if waypoints is not None:
            if waypoints.shape[1] < 3 or len(waypoints.shape) != 2:
                raise ValueError("Waypoints needs to be a (Nxm), m >= 3, numpy array!")
            self.waypoints = waypoints
        else:
            if self.waypoints is None:
                raise ValueError(
                    "Please set waypoints to track during planner instantiation or when calling plan()"
                )
        vehicle_state = State(
            x=states["pose_x"],
            y=states["pose_y"],
            delta=states["delta"],
            v=states["linear_vel_x"],
            yaw=states["pose_theta"],
            yawrate=states["ang_vel_z"],
            beta=states["beta"],
        )

        # setting the initial state
        self.initial_state = jnp.array([states["pose_x"], states["pose_y"], states["linear_vel_x"], states["pose_theta"]])


        cx = self.waypoints[0]  # Trajectory x-Position
        cy = self.waypoints[1]  # Trajectory y-Position
        cyaw = self.waypoints[2]  # Trajectory Heading angle
        sp = self.waypoints[3]  # Trajectory Velocity

        # Calculate the next reference trajectory for the next T steps:: [x, y, v, yaw]
        self.ref_path = self.calc_ref_trajectory_kinematic(
            vehicle_state, cx, cy, cyaw, sp
        )

        # print(self.ref_path.shape)    4 states and 9 time steps
        # print(self.ref_path[:2].T)               #waypoints checked
        # ref path works as expected
        # self.ref_path = self.ref_path.T
        # self.ref_path = self.ref_path[:-1]

        # sampling
        # checking the clipping
        self.a_opt = jnp.concatenate([self.a_opt[1:, :],
                             jnp.expand_dims(jnp.zeros((self.dim_a,)),
                                             axis=0)])  # [n_steps, dim_a]
        rng_da, self.key = jax.random.split(self.key)
        da = jax.random.multivariate_normal(rng_da, jnp.zeros(2), jnp.diag(jnp.array([0.1, 1])), (32,8))

        da = jnp.clip(da, jnp.array([-0.4, 0]), jnp.array([0.4, 6]))  #[n_samples, n_steps, dim_a]
        print(da)
        # print(self.a_opt)
        # bad makes it v slow and changes nothin ig or I need to be more patient n maybe it works?
        # (self.a_opt, _, self.key), rollout = jax.lax.scan(self.iteration_step,init=(self.a_opt, self.a_cov, self.key), xs=None, length = self.n_iterations)
        

        # print(rollout.shape)

        # da = jax.random.truncated_normal(
        #         rng_da,
        #         -jnp.ones_like(self.a_opt) * self.a_std - self.a_opt,
        #         jnp.ones_like(self.a_opt) * self.a_std - self.a_opt,
        #         shape=(self.n_samples, self.n_steps, 2)
        #     )

        # a = jnp.expand_dims(self.a_opt, axis=0) + da

        # a = jnp.clip(a, jnp.array([-0.4, 0]), jnp.array([0.4, 6]))  #[n_samples, n_steps, dim_a]

        # rollout_iter, cost = jax.vmap(self.iteration_vmap)(a)
        # print(rollout_iter.shape)

        def close_event():
            plt.close() #timer calls this function after 3 seconds and closes the window 

        # fig = plt.figure()
        # timer = fig.canvas.new_timer(interval = 1000) #creating a timer object and setting an interval of 3000 milliseconds
        # timer.add_callback(close_event)

        # timer.start()
        # plt.scatter(-1, -1)
        # for i in range(self.n_samples):
        #     plt.plot(rollout_iter[i][:,1], rollout_iter[i][:,0])
        # plt.show()

        # R = jax.vmap(self.returns)(cost)

        # w = jax.vmap(self.weights, 1, 1)(R)
        # da_opt = jax.vmap(jnp.average, (1, None, 1))(da, 0, w)

        # self.a_opt  = jnp.clip(self.a_opt + da_opt, jnp.array([-0.4, 0]), jnp.array([0.4, 6]))


        # _, (mppi_rollout,_) = jax.lax.scan(self.dynamics,init=self.initial_state, xs=self.a_opt)
        # timer.start()
        # plt.scatter(-1, -1)
        # plt.plot(mppi_rollout[:,1], mppi_rollout[:,0])
            
        # plt.show()
        steerv, accel = self.a_opt[0]
        return  steerv, accel

        
    @partial(jax.jit, static_argnums=0)
    def iteration_step(self, input, _):
        a_opt, a_cov, key = input
        dim_a = jnp.prod(2)

        key_da, key = jax.random.split(key)

        da = jax.random.truncated_normal(
                key_da,
                -jnp.ones_like(a_opt) * self.a_std - a_opt,
                jnp.ones_like(a_opt) * self.a_std - a_opt,
                shape=(self.n_samples, self.n_steps, 2)
            )

        a = jnp.expand_dims(a_opt, axis=0) + da

        a = jnp.clip(a, jnp.array([-0.4, 0]), jnp.array([0.4, 6]))  #[n_samples, n_steps, dim_a]

        rollout_iter, cost = jax.vmap(self.iteration_vmap)(a)
        
        # ref_state_copy = jnp.repeat(self.ref_path[None], self.n_samples, axis = 0)
        # print(cost.shape)
        # cost = jax.vmap(self.stage_cost)((rollout_iter, ref_state_copy))

        W = jax.vmap(self.weights, 1, 1)(cost)

        da_opt = jax.vmap(jnp.average, (1, None, 1))(da, 0, W)  # [n_steps, dim_a]

        a_opt  = jnp.clip(a_opt + da_opt, jnp.array([-0.4, 0]), jnp.array([0.4, 6]))
        
        return (a_opt, a_cov, key), rollout_iter



    @partial(jax.jit, static_argnums=0)
    def dynamics(self, state, control):
        v, yaw = state[2:]
        delta, a = control
        # set wheelbase
        L = self.config.WB

        # x-state
        state = state.at[0].add(v*jnp.cos(yaw)*self.dt)
        #y-state
        state = state.at[1].add(v*jnp.sin(yaw)*self.dt)

        state = state.at[2].add(a*self.dt)
        state = state.at[3].add((v/L)*jnp.tan(delta)*self.dt)
        

        target_state = jnp.array([-1, -1, 0.8, 3])
        delta_x = (state - target_state).reshape((4,1)) # 1x4
        Q = jnp.diag(jnp.array([ 0, 0, 10, 100]))
        
        # cost = jnp.dot
        cost = jnp.dot(delta_x.T, jnp.dot(Q, delta_x))[0][0]
        # print(cost)
        return state, (state, cost)
    
    @partial(jax.jit, static_argnums=0)
    def iteration_vmap(self, sampled_input):
        x_init = self.initial_state
        x_init, (rollout, stage_cost) = jax.lax.scan(self.dynamics,init=x_init, xs=sampled_input)

        return rollout, stage_cost
    
    @partial(jax.jit, static_argnums=0)
    def stage_cost(self, rollout_delta):
        rollout_state, ref_state = rollout_delta

        # print(jnp.sum(jnp.square(rollout_state - ref_state), axis = -1))

        return jnp.sum(jnp.square(rollout_state - ref_state), axis = -1)
    
    @partial(jax.jit, static_argnums=0)
    def returns(self, r):
    # r: [n_steps]
        return jnp.dot(jnp.triu(jnp.ones((self.n_steps, self.n_steps))),
                   r)  # R: [n_steps]
    
    
    @partial(jax.jit, static_argnums=0)
    def weights(self, R):  # pylint: disable=invalid-name
        # R: [n_samples]
        # R_stdzd = (R - jnp.min(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)
        # R_stdzd = R - jnp.max(R) # [n_samples] np.float32
        R_stdzd = (R - jnp.max(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)  # pylint: disable=invalid-name
        w = jnp.exp(R_stdzd / self.temperature)  # [n_samples] np.float32
        w = w/jnp.sum(w)  # [n_samples] np.float32
        return w
    
