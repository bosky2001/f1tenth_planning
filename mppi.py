import jax
import jax.numpy as jnp
from f1tenth_planning.utils.utils import nearest_point
from dataclasses import dataclass, field
import numpy as np
import math

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
        self.damping = 0.01
        self.temperature = 0.01
        self.sigma = np.diag([0.1, 1.5])  # steering, acceleration
        self.mean = np.array([0.0, 2.0])
        self.sampled_trajs = np.zeros((self.config.TK, 4))
    
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
            points = self.ref_path[:2].T
            e.render_lines(points, color=(0, 128, 0), size=2)
        else:
            print("balls")
    
    def render_sampled_plan(self, e):
        """
        Drawing the sampled trajectory from MPPI
        """
        if self.sampled_trajs is not None:
            points = self.sampled_trajs[:2].T
            e.render_lines(points, color=(0, 128, 0), size=2)
            
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

        # self.MPC_Control_kinematic(vehicle_state, self.waypoints)

        cx = self.waypoints[0]  # Trajectory x-Position
        cy = self.waypoints[1]  # Trajectory y-Position
        cyaw = self.waypoints[2]  # Trajectory Heading angle
        sp = self.waypoints[3]  # Trajectory Velocity

        # Calculate the next reference trajectory for the next T steps:: [x, y, v, yaw]
        self.ref_path = self.calc_ref_trajectory_kinematic(
            vehicle_state, cx, cy, cyaw, sp
        )
        # print(self.ref_path.shape)    4 states and 9 time steps


        x_0 = vehicle_state


        sampled_inputs = np.random.multivariate_normal(self.mean, self.sigma, (1000, 8))
        traj_rollout_costs = np.zeros(self.samples)

        traj_rollout_control = np.zeros((self.samples, self.config.TK, 2))
        # print(traj_rollout_control.shape)
        
        # sampling trajectories/ rollout
        for k in range(self.samples):
            curr_state = x_0

            cost = 0.0
            for dt in range(1, self.config.TK+1):

                sampled_input = sampled_inputs[k,dt-1]
                # do rollout with input
                curr_state = self.update_state_kinematic(curr_state, sampled_input)
                cost += self.stage_cost(curr_state, self.ref_path[:, dt])
                traj_rollout_control[k, dt-1] = sampled_input
            traj_rollout_costs[k] = cost

        w = self.weights(traj_rollout_costs)
        # print(w.shape)


        mppi_control = np.zeros((self.config.TK, 2))

        for dt in range(self.config.TK):
            for k in range(self.samples):
                mppi_control[dt] += w[k]* traj_rollout_control[k, dt]
        
        # print(mppi_control.shape)
        curr_state = x_0
        self.sampled_trajs[0] = np.array([x_0.x, x_0.y, x_0.x, x_0.yaw])
        for i in range(1, self.config.TK):
            curr_state = self.update_state_kinematic(curr_state, mppi_control[i-1])
            self.sampled_trajs[i] = np.array([curr_state.x, curr_state.y, curr_state.v, curr_state.yaw])
            
        print(self.sampled_trajs.shape)
        self.u_prev[:-1] = mppi_control[1:]
        self.u_prev[-1] = mppi_control[-1]
        return mppi_control[0]

    def update_state_kinematic(self, state, control):

        a = control[0]
        delta = control[1]
        # input check
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= -self.config.MAX_STEER:
            delta = -self.config.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        )
        state.v = state.v + a * self.config.DTK

        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED

        return state
    
    
        


    def stage_cost(self, state, ref_state):
        Q = np.diag([100, 100, 100, 10])

        cost = 0.0

        xt = np.array([state.x, state.y, state.v, state.yaw])
        cost = (xt-ref_state) @ Q @ (xt-ref_state).T
        # print(xt - ref_state)
        # for i in range(ref_state.shape[1]):
        #     cost += (state-ref_state[i]) @ Q @ (state - ref_state[i])

        return cost
    
    def terminal_cost(self, state):
        pass
    
    def weights(self, R):  # pylint: disable=invalid-name
        # R: [n_samples]
        # R_stdzd = (R - jnp.min(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)
        # R_stdzd = R - jnp.max(R) # [n_samples] np.float32
        R_stdzd = (R - np.max(R)) / ((np.max(R) - np.min(R)) + self.damping)  # pylint: disable=invalid-name
        w = np.exp(R_stdzd / self.temperature)  # [n_samples] np.float32
        w = w/np.sum(w)  # [n_samples] np.float32
        return w
    
