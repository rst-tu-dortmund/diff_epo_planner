# ------------------------------------------------------------------------------
# Copyright:    Technical University Dortmund
# Project:      KISSaF
# Created by:   Institute of Control Theory and System Engineering
# ------------------------------------------------------------------------------
import torch as torch
import torch.nn as nn
import theseus as th
import data.constants as c
import numpy as np
from theseus.core.cost_function import AutogradMode
from theseus.optimizer.nonlinear import BackwardMode
import einops

class MultiAgentGameObjective(nn.Module):
    def __init__(self, model, configs, fps):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available else "cpu"

        # Config Parameters
        self.configs = configs
        self.total_time = configs["hyperparameters"]["prediction_length"]
        self.batch_size = configs["hyperparameters"]["batch_size"]
        self.num_of_modes = configs["hyperparameters"]["num_modes"]
        self.cost_token = configs["differentiable_optimization"]["cost_token"]

        self.fps = fps
        self.dt = 1 / self.fps
        self.planning_horizon = int(configs["hyperparameters"]["prediction_length"] * self.fps)
        self.collision_mode = configs["differentiable_optimization"]["circle_approximation"]


        self.set_theseus_autograd_mode(configs)
        self.set_theseus_backward_mode(configs)

        # Model Parameters
        self.model = model
        self.num_of_states = model.num_of_states
        self.num_of_inputs = model.num_of_controls

        self.number_player = configs["differentiable_optimization"]["number_player"]
        self.car_lenghts = None
        self.car_widths = None
        self.centerline = None
        self.sampled_centerline = None

        self.optim_state_dof = self.number_player * self.planning_horizon * self.num_of_states
        self.optim_input_dof = self.number_player * self.planning_horizon * self.num_of_inputs
        self.start_pose_dim = self.number_player * 1 * self.num_of_states

        # Create Optimization Variables
        # single Shooting only needs u_opt
        self.cntrl_dict = {}
        self.cntrl_dict[f"opt_cntrl"] = th.Vector(dof=self.optim_input_dof, name=f"opt_cntrl")
        self.num_of_distances = int(self.number_player * (self.number_player - 1) / 2) # num of possible distances between agents (n*(n-1)/2)
        self.distances = np.zeros((self.num_of_modes, self.num_of_distances, self.planning_horizon))
        self.lane_subsampling_factor = self.configs["differentiable_optimization"]["lane_subsampling_factor"]

        self.collision_avoidance_subsampling = self.configs["differentiable_optimization"]["collision_avoidance_subsampling_factor"]

        self.create_cost_weights()
        self.create_start_goal_states()
        self.create_objective()

    def set_theseus_autograd_mode(self, configs):
        if configs["differentiable_optimization"]["autograd_mode"] == "DENSE":
            self.autograd_mode = AutogradMode.DENSE
        elif configs["differentiable_optimization"]["autograd_mode"] == "VMAP":
            self.autograd_mode = AutogradMode.VMAP
        elif configs["differentiable_optimization"]["autograd_mode"] == "LOOP_BATCH":
            self.autograd_mode = AutogradMode.LOOP_BATCH
        else:
            raise NotImplementedError

    def set_theseus_backward_mode(self, configs):
        if configs["differentiable_optimization"]["backward_mode"] == "UNROLL":
            self.bw_mode = BackwardMode.UNROLL
        elif configs["differentiable_optimization"]["backward_mode"] == "IFT":
            self.bw_mode = BackwardMode.IMPLICIT
        else:
            raise NotImplementedError


    def set_dataset_specific_FPS(self, configs):
        if self.configs["general"]["dataset"] == c.WAYMO:
            self.fps = c.FPS_WAYMO
        else:
            raise NotImplementedError

    def set_lanes(self, mini_batch):
        self.centerline = mini_batch["agent_centerlines"].repeat_interleave(self.num_of_modes, dim=0).to(self.device).type(torch.float32)
        self.sampled_centerline = self.centerline # currently these are not used in wayomo


    def create_cost_weights(self):
        """
        Create cost weights for the different cost functions.
        """
        # Set the degree of freedom for each cost function
        if self.collision_mode == "single":
            int_dof = (self.batch_size * self.num_of_modes, self.num_of_distances * self.planning_horizon // self.collision_avoidance_subsampling)
        else:
            raise NotImplementedError

        own_dof = (self.batch_size * self.num_of_modes, self.number_player * self.planning_horizon)
        bound_dof = (self.batch_size * self.num_of_modes, self.number_player * 2)
        change_dof = (self.batch_size * self.num_of_modes, self.number_player * (self.planning_horizon - 1))
        lane_dof = (self.batch_size * self.num_of_modes, self.number_player * self.planning_horizon * 2 // self.lane_subsampling_factor)


        cost_tensor_collision = th.Vector(tensor=torch.zeros(int_dof), name="CostTensorCollision")
        cost_tensor_input_omega = th.Vector(tensor=torch.zeros(own_dof), name="CostTensorOmega")
        cost_tensor_input_omega_change = th.Vector(tensor=torch.zeros(change_dof), name="CostTensorOmegaChange")
        cost_tensor_input_omega_box = th.Vector(tensor=torch.zeros(own_dof), name="CostTensorOmegaBox")
        cost_tensor_input_acc = th.Vector(tensor=torch.zeros(own_dof), name="CostTensorAcc")
        cost_tensor_input_acc_change = th.Vector(tensor=torch.zeros(change_dof), name="CostTensorAccChange")
        cost_tensor_input_states = th.Vector(tensor=torch.zeros(own_dof), name="CostTensorStates")
        cost_tensor_input_states_box = th.Vector(tensor=torch.zeros(own_dof), name="CostTensorStatesBox")
        cost_tensor_lane = th.Vector(tensor=torch.zeros(lane_dof), name="CostTensorLane")
        cost_tensor_boundary = th.Vector(tensor=torch.zeros(bound_dof), name="CostTensorBoundary")
        cost_tensor_ref_vel = th.Vector(tensor=torch.zeros(own_dof), name="CostTensorReferenceVelocity")


        # Interactive cost weights
        self.cost_weights_collision = th.DiagonalCostWeight(diagonal=cost_tensor_collision, name="CostWeightCollision")

        # Non-interactive cost weights
        self.cost_weights_input_omega = th.DiagonalCostWeight(diagonal=cost_tensor_input_omega, name="CostWeightInputOmega")
        self.cost_weights_input_omega_change = th.DiagonalCostWeight(diagonal=cost_tensor_input_omega_change, name="CostWeightInputOmegaChange")
        self.cost_weights_input_omega_box = th.DiagonalCostWeight(diagonal=cost_tensor_input_omega_box, name="CostWeightInputBox")
        self.cost_weights_input_acc = th.DiagonalCostWeight(diagonal=cost_tensor_input_acc, name="CostWeightInputAcc")
        self.cost_weights_input_acc_change = th.DiagonalCostWeight(diagonal=cost_tensor_input_acc_change, name="CostWeightInputAccChange")
        self.cost_weights_states = th.DiagonalCostWeight(diagonal=cost_tensor_input_states, name="CostWeightStates")
        self.cost_weights_states_box = th.DiagonalCostWeight(diagonal=cost_tensor_input_states_box, name="CostWeightStatesBox")
        self.cost_weights_lane = th.DiagonalCostWeight(diagonal=cost_tensor_lane, name="CostWeightLane")
        self.cost_weights_boundary = th.DiagonalCostWeight(diagonal=cost_tensor_boundary, name="CostWeightBoundary")
        self.cost_weights_ref_vel = th.DiagonalCostWeight(diagonal=cost_tensor_ref_vel, name="CostWeightRefVel")


    def create_start_goal_states(self):
        """
        Create the start and goal states for the optimization problem. These are used as auxiliary variables in the cost functions.
        """
        self.start_states = th.Variable(torch.empty(1, self.start_pose_dim), name='start_pose')
        self.goal_states = th.Variable(torch.empty(1, self.start_pose_dim), name='goal_pose')



    def create_objective(self):
        """
        Create the objective function for the optimization problem. The objective function is dynamicly created based on
        the self.costtoken string, which is passed with the configs.
        For instance, "BCS" generates boundary, control, and state costs for the optimization problem.
        """
        self.obj_func = th.Objective()

        # Boundary Cost w.r.t goal state
        if "B" in self.cost_token:
            self.create_boundary_objectives()

        # Control Cost
        if "C" in self.cost_token:
            self.create_control_cost()

        # State Cost
        if "S" in self.cost_token:
            self.create_state_vel_cost()

        # Lane Keeping Cost
        if "L" in self.cost_token:
            self.create_lane_keeping_cost()

        # Reference Velocity Cost
        if "V" in self.cost_token:
            self.create_reference_velocity_cost()

        # Interaction Costs (care: must be the last for correct indexing)
        if "I" in self.cost_token:
            if self.number_player > 1:
                self.create_collision_avoidance_cost()


    # *  #############################################################################  #*
    # *  ############################## COST FUNCTIONS ###############################  #*
    # *  #############################################################################  #*

    def update_weights(self, weights_own, weights_int):
        """
        Update the cost weights for the different cost functions based on neural network outputs-
        """

        if not self.configs["differentiable_optimization"]["multimodal_weights"]:
            # Case: Unimodal weight prediction
            weights_own = einops.rearrange(weights_own, "b (a w) -> b w a", a=self.number_player)
            weights_int = weights_int.repeat_interleave(self.num_of_modes, dim=0)
            weights_own = weights_own.repeat_interleave(self.num_of_modes, dim=0)

        else:
            # Case: Unimodal weight prediction (different weights for each mode)
            weights_own = einops.rearrange(weights_own, "b (m a w) -> (b m) w a", b=self.batch_size, m=self.num_of_modes, a=self.number_player)
            weights_int = einops.rearrange(weights_int, "b (m d) -> (b m) d", b=self.batch_size, m=self.num_of_modes)

       # repeat interaction weights (for collision avoidance) for all time steps (time-invariant)
        weights_int = weights_int.repeat_interleave(self.planning_horizon // self.collision_avoidance_subsampling, dim=1)


        for i, cost_name in enumerate(self.obj_func.cost_functions):

            if cost_name == "Boundary_Cost":
                self.obj_func.cost_functions[cost_name].weight.diagonal.update(weights_own[:, i].repeat_interleave(2, dim=1))
            elif cost_name == "QuadrInpAccChange":
                self.obj_func.cost_functions[cost_name].weight.diagonal.update(weights_own[:, i].repeat_interleave(self.planning_horizon - 1, dim=1))
            elif cost_name == "QuadrInpOmegaChange":
                self.obj_func.cost_functions[cost_name].weight.diagonal.update(weights_own[:, i].repeat_interleave(self.planning_horizon - 1, dim=1))
            elif cost_name == "LaneKeeping":
                self.obj_func.cost_functions[cost_name].weight.diagonal.update(
                    weights_own[:, i].repeat_interleave((self.planning_horizon // self.lane_subsampling_factor) * 2, dim=1))
            elif cost_name == "Col_Cost":
                self.cost_weights_collision.diagonal.update(weights_int)
            else:
                self.obj_func.cost_functions[cost_name].weight.diagonal.update(weights_own[:, i].repeat_interleave(self.planning_horizon, dim=1))

    def update_controls(self, control_dict):
        self.obj_func.update(control_dict)


    def update_boundary_points(self, gamex0, gamexN):
        """
        updates the start and goal states for the optimization problem (auxiliary variables in the cost functions)
        :param gamex0: current joint state for all agents
        :param gamexN: predicted goal positions for all agents
        :return:
        """
        self.start_states.update(gamex0.type(torch.float32))
        gamexN = einops.rearrange(gamexN[:, :, :, 0], 'b m a s -> (b m) (a s)',
                                      m=self.num_of_modes,
                                      a=self.number_player,
                                      s=self.num_of_states)

        self.goal_states.update(gamexN.type(torch.float32))

    # *  #############################################################################  #*
    # *  ############################## CONTROL COST   ###############################  #*
    # *  #############################################################################  #*

    def create_control_cost(self):
        input_cost_omega = th.AutoDiffCostFunction(
            optim_vars=[self.cntrl_dict[f"opt_cntrl"]],
            aux_vars=[self.start_states, self.goal_states],
            err_fn=self.control_cost_omega_err_fn,
            dim=self.number_player * self.planning_horizon,
            cost_weight=self.cost_weights_input_omega,
            name=f"QuadrInpOmega")
        self.obj_func.add(input_cost_omega)

        input_cost_omega_change = th.AutoDiffCostFunction(
            optim_vars=[self.cntrl_dict[f"opt_cntrl"]],
            aux_vars=[self.start_states, self.goal_states],
            err_fn=self.control_cost_omega_change_err_fn,
            dim=self.number_player * (self.planning_horizon - 1),
            cost_weight=self.cost_weights_input_omega_change,
            name=f"QuadrInpOmegaChange")
        self.obj_func.add(input_cost_omega_change)

        input_cost_acc_change = th.AutoDiffCostFunction(
            optim_vars=[self.cntrl_dict[f"opt_cntrl"]],
            aux_vars=[self.start_states, self.goal_states],
            err_fn=self.control_cost_acc_change_err_fn,
            dim=self.number_player * (self.planning_horizon - 1),
            cost_weight=self.cost_weights_input_acc_change,
            name=f"QuadrInpAccChange")
        self.obj_func.add(input_cost_acc_change) # jerk


        input_cost_acc = th.AutoDiffCostFunction(
            optim_vars=[self.cntrl_dict[f"opt_cntrl"]],
            aux_vars=[self.start_states, self.goal_states],
            err_fn=self.control_cost_acc_err_fn,
            dim=self.number_player * self.planning_horizon,
            cost_weight=self.cost_weights_input_acc,
            name=f"QuadrInpAcc")
        self.obj_func.add(input_cost_acc)


    def control_cost_omega_err_fn(self, optim_vars, aux_vars):
        optim_control, current_state, goal_state, trajectory = self.get_state(optim_vars, aux_vars)
        omega = optim_control[..., c.OMEGA_IND]
        # if torch.isnan(omega).any():
        #    print("omega nan")
        return einops.rearrange(omega, 'b p t -> b (p t)')

    def control_cost_omega_change_err_fn(self, optim_vars, aux_vars):
        optim_control, current_state, goal_state, trajectory = self.get_state(optim_vars, aux_vars)
        omega = optim_control[..., c.OMEGA_IND]
        diff_omega = torch.diff(omega) / self.dt
        return einops.rearrange(diff_omega, 'b p t -> b (p t)')

    def control_cost_box_omega_err_fn(self, optim_vars, aux_vars):
        optim_control, current_state, goal_state, trajectory = self.get_state(optim_vars, aux_vars)
        omega = optim_control[..., c.OMEGA_IND]

        omega_lb_soft = torch.where((omega < self.inp_lb_omega),
                                    (self.inp_lb_omega - omega),
                                    torch.zeros(1, device=self.device))

        omega_ub_soft = torch.where((omega > self.inp_ub_omega),
                                    (omega - self.inp_ub_omega),
                                    torch.zeros(1, device=self.device))

        omega_soft = (omega_lb_soft + omega_ub_soft)

        return einops.rearrange(omega_soft, 'b p t -> b (p t)')

    def control_cost_acc_err_fn(self, optim_vars, aux_vars):
        optim_control, current_state, goal_state, trajectory = self.get_state(optim_vars, aux_vars)
        acc = optim_control[..., c.ACC_IND]
        return einops.rearrange(acc, 'b p t -> b (p t)')

    def control_cost_acc_change_err_fn(self, optim_vars, aux_vars):
        optim_control, current_state, goal_state, trajectory = self.get_state(optim_vars, aux_vars)
        acc = optim_control[..., c.ACC_IND]
        diff_acc = torch.diff(acc) / self.dt
        return einops.rearrange(diff_acc, 'b p t -> b (p t)')

    def control_cost_box_acc_err_fn(self, optim_vars, aux_vars):
        optim_control = optim_vars[0].tensor.view(-1, self.number_player, self.planning_horizon, self.num_of_inputs).to(self.device)
        curr_a = optim_control.view(-1, self.number_player, self.planning_horizon, self.num_of_inputs)[:, :, :, c.ACC_IND]
        curr_a = curr_a.view(-1, self.number_player * self.planning_horizon)
        a_lb_soft = torch.where((curr_a < self.inp_lb_a),
                                (curr_a - self.inp_lb_a),
                                torch.zeros(1, device=self.device))
        a_ub_soft = torch.where((curr_a > self.inp_ub_a),
                                (curr_a - self.inp_ub_a),
                                torch.zeros(1, device=self.device))
        a_soft = a_lb_soft + a_ub_soft
        return a_soft

    # *  #############################################################################  #!
    # *  ############################## STATE COST ###################################  #!
    # *  #############################################################################  #!

    def create_state_vel_cost(self):
        state_cost = th.AutoDiffCostFunction(
            optim_vars=[self.cntrl_dict["opt_cntrl"]],
            aux_vars=[self.start_states, self.goal_states],
            err_fn=self.state_vel_cost_err_fn,
            dim=self.number_player * self.planning_horizon,
            cost_weight=self.cost_weights_states,
            name="QuadrState")
        self.obj_func.add(state_cost)


    def state_vel_cost_err_fn(self, optim_vars, aux_vars):
        optim_control, current_state, goal_state, trajectory = self.get_state(optim_vars, aux_vars)
        velocity = trajectory[..., c.V_IND]
        return einops.rearrange(velocity, 'b p t -> b (p t)')

    def state_vel_cost_box_err_fn(self, optim_vars, aux_vars):
        optim_control, current_state, goal_state, trajectory = self.get_state(optim_vars, aux_vars)

        velocity = trajectory[..., c.V_IND]

        vel_lb_soft = torch.where((velocity < self.vel_lb),
                                  (self.vel_lb - velocity),
                                  torch.zeros(1, device=self.device))

        vel_ub_soft = torch.where((velocity > self.vel_ub),
                                  (velocity - self.vel_ub),
                                  torch.zeros(1, device=self.device))

        vel_soft = (vel_lb_soft + vel_ub_soft)
        return einops.rearrange(vel_soft, 'b p t -> b (p t)')

    # *  #############################################################################  #!
    # *  ############################## BOUNDARY COST ################################  #!
    # *  #############################################################################  #!

    def get_state(self, optim_vars, aux_vars):
        optim_control = einops.rearrange(optim_vars[0].tensor, 'b (p t c) -> b p t c', p=self.number_player, t=self.planning_horizon, c=self.num_of_inputs)
        current_state = einops.rearrange(aux_vars[0].tensor, 'b (p t s) -> b p t s', p=self.number_player, s=self.num_of_states, t=1)
        goal_state = einops.rearrange(aux_vars[1].tensor, 'b (p s) -> b p s', p=self.number_player, s=self.num_of_states)
        trajectory = self.model.state_transition(optim_control, current_state, self.dt)
        return optim_control, current_state, goal_state, trajectory

    def boundary_err_fn(self, optim_vars, aux_vars):
        optim_control, current_state, goal_state, trajectory = self.get_state(optim_vars, aux_vars)
        goal_opt_state = trajectory[:, :, -1, :]

        # * new version (x-y differences)
        diff_goal = (goal_opt_state[..., 0:2] - goal_state[..., 0:2])

        return einops.rearrange(diff_goal, 'b p s -> b (p s)')

    def create_boundary_objectives(self):
        Boundary_cost = th.AutoDiffCostFunction(
            optim_vars=[self.cntrl_dict[f"opt_cntrl"]],
            aux_vars=[self.start_states, self.goal_states],
            err_fn=self.boundary_err_fn,
            dim=self.number_player * 2,  # 2 for (x,y) states
            autograd_mode=self.autograd_mode,
            cost_weight=self.cost_weights_boundary,
            name=f"Boundary_Cost")
        self.obj_func.add(Boundary_cost)

    # *  #############################################################################  #!
    # *  ############################## LANE KEEPING #################################  #!
    # *  #############################################################################  #!

    def create_lane_keeping_cost(self):
        lane_keeping_cost = th.AutoDiffCostFunction(
            optim_vars=[self.cntrl_dict["opt_cntrl"]],
            aux_vars=[self.start_states, self.goal_states],
            err_fn=self.lane_keeping_cost_err_fn,
            dim=self.number_player * self.planning_horizon * 2 // self.lane_subsampling_factor,
            cost_weight=self.cost_weights_lane,
            name="LaneKeeping")
        self.obj_func.add(lane_keeping_cost)

    def create_reference_velocity_cost(self):
        ref_vel_cost = th.AutoDiffCostFunction(
            optim_vars=[self.cntrl_dict["opt_cntrl"]],
            aux_vars=[self.start_states, self.goal_states],
            err_fn=self.ref_vel_cost_err_fn,
            dim=self.number_player * self.planning_horizon,
            cost_weight=self.cost_weights_ref_vel,
            name="RefVel")
        self.obj_func.add(ref_vel_cost)



    def lane_keeping_cost_err_fn(self, optim_vars, aux_vars):
        optim_control, current_state, goal_state, trajectory = self.get_state(optim_vars, aux_vars)
        distance_to_ref = torch.cdist(trajectory[..., 0::self.lane_subsampling_factor, 0:2].double(),
                                      self.sampled_centerline[..., 0::self.lane_subsampling_factor, 0:2].double())

        indices = einops.repeat(torch.argmin(distance_to_ref, dim=-1), 'b p t -> b p t s', s=3)[..., 0:2]

        ref_points = torch.gather(self.sampled_centerline, 2, indices)
        lane_error = torch.cat([trajectory[:, :, 0::self.lane_subsampling_factor, 0] - ref_points[:, :, :, 0],
                                trajectory[:, :, 0::self.lane_subsampling_factor, 1] - ref_points[:, :, :, 1]], dim=-1)
        return einops.rearrange(lane_error, 'b p s -> b (p s)')

    def ref_vel_cost_err_fn(self, optim_vars, aux_vars):
        optim_control, current_state, goal_state, trajectory = self.get_state(
            optim_vars, aux_vars)
        vel_trajectory = trajectory[:, :, :, c.V_IND]
        if self.configs["differentiable_optimization"]["map_based_ref_velocity"]:
            # tensor
            vel_diff = vel_trajectory - self.reference_velocity
        else:
            # scalar
            vel_diff = vel_trajectory - einops.repeat(self.reference_velocity, '()  -> b p t', b=self.batch_size * self.num_of_modes, p=self.number_player,
                                                      t=self.planning_horizon)
        return einops.rearrange(vel_diff, 'b p s -> b (p s)')

    # *  #############################################################################  #!
    # *  ############################## COLLISION COST ###############################  #!
    # *  #############################################################################  #!

    def collision_avoidance_err_fn(self, optim_vars, aux_vars):
        # unroll Trajectory
        optim_control, current_state, goal_state, trajectory = self.get_state(optim_vars, aux_vars)
        # Create collision pairs
        ids = torch.arange(self.number_player)
        ids_pairs = torch.combinations(ids, 2).to(self.device)

        # get x data of pairs
        x_pairs = torch.stack([trajectory[:, ids_pairs[:, 0], ::self.collision_avoidance_subsampling, c.X_IND],
                               trajectory[:, ids_pairs[:, 1], ::self.collision_avoidance_subsampling, c.X_IND]], dim=1)

        # get y data of pairs
        y_pairs = torch.stack([trajectory[:, ids_pairs[:, 0], ::self.collision_avoidance_subsampling, c.Y_IND],
                               trajectory[:, ids_pairs[:, 1], ::self.collision_avoidance_subsampling, c.Y_IND]], dim=1)

        # determine safe distance (half of car length of both agents)
        safe_distance = torch.stack([self.model.car_lengths[:, ids_pairs[:, 0]] / 2 + self.model.car_lengths[:, ids_pairs[:, 1]] / 2], dim=-1).repeat_interleave(
            self.planning_horizon // self.collision_avoidance_subsampling, dim=-1)

        # compute differences
        dx_pairs = torch.diff(x_pairs, dim=1)
        dy_pairs = torch.diff(y_pairs, dim=1)
        d_pairs = torch.sqrt(dx_pairs ** 2 + dy_pairs ** 2) # distances

        dist = d_pairs[:, 0]

        if self.configs["differentiable_optimization"]["coll_avoid_safety_distance_field"]:
            # Use an additional safety distance with penalizes distances similiar to https://arxiv.org/abs/2101.06832
            # Care this leads to much higher memory requirements
            dist = self.safety_distance_field(dist, safe_distance) # TOD
        else:
            dist = self.normal_distance_field(dist, safe_distance)
        dist = dist.view(-1, self.planning_horizon // self.collision_avoidance_subsampling, self.num_of_distances)
        return einops.rearrange(dist, 'b t d -> b (d t)')

    def safety_distance_field(self, dist: torch.tensor, safe_distance: torch.tensor):
        dist = torch.where(
            dist < safe_distance,
            safe_distance - dist,
            (safe_distance / dist))
        return dist

    def normal_distance_field(self, dist: torch.tensor, safe_distance: torch.tensor):
        # implements a safety inequality constraint like max(0, safe_distance - dist)
        dist = torch.where(dist < safe_distance,
                           safe_distance - dist,
                           torch.zeros(1).to(device=self.device))
        return dist

    def create_collision_avoidance_cost(self):
        col_dim = self.num_of_distances * self.planning_horizon // self.collision_avoidance_subsampling
        collision_cost = th.AutoDiffCostFunction(
            optim_vars=[self.cntrl_dict[f"opt_cntrl"]],
            aux_vars=[self.start_states, self.goal_states],
            err_fn=self.collision_avoidance_err_fn,
            dim=col_dim,
            autograd_mode=self.autograd_mode,
            cost_weight=self.cost_weights_collision,
            name=f"Col_Cost")
        self.obj_func.add(collision_cost)




