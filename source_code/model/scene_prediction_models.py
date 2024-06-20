# ------------------------------------------------------------------------------
# Copyright:    Technical University Dortmund
# Project:      KISSaF
# Created by:   Institute of Control Theory and System Engineering
# ------------------------------------------------------------------------------

import einops
import torch
from tqdm import tqdm
import theseus as th
import torch.profiler
import data.constants as c
import wandb
from abc import ABC, abstractmethod
import numpy as np
from model.differentiable_optimization.dynamics.unicycle import UnicycleModel
from model.differentiable_optimization.objective.multi_agent_game_objective import MultiAgentGameObjective
from model.metrics import Metrics
from model.modules import SceneControlNet, EPONetF
from model.models import create_optimizer, create_lr_scheduler, restore_model



class SceneBaseClass(ABC):

    def __init__(self, configs):
        self.net: torch.nn.Module = None
        self.init_hyperparameters(configs)

    def init_hyperparameters(self, configs):
        self.gradient_clip = configs["hyperparameters"]["gradient_clip"]
        self.decay_epoch = configs["hyperparameters"]["decay_epoch"]

    def optimize_parameters(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.gradient_clip)
        self.optimizer.step()


    def lr_step(self, epoch):
        if (epoch >= self.decay_epoch) and (self.lr_scheduler is not None):
            self.lr_scheduler.step()
        self._epoch = epoch

    def restore_model(self, cfgs: dict) -> None:
        """
        Restores parameters from pretrained model.
        :return: None
        :param cfgs: configs with relevant config parameters merged from pretrained model.
        """
        net_state_dict, optimizer_state_dict, epoch, fps = restore_model(cfgs)
        if self.net:
            self.net.load_state_dict(net_state_dict)

        if (
            "finetune_restored_model" in cfgs["general"]
            and cfgs["general"]["finetune_restored_model"]
        ):
            # start new training initialized with restored model
            epoch = 0
            self.optimizer = create_optimizer(self, cfgs)
            if cfgs["config_params"]["lr_scheduler"] != 0:
                self.lr_scheduler = create_lr_scheduler(self, cfgs)
        else:
            # continue training of restored model
            self.optimizer.load_state_dict(optimizer_state_dict)
            self.optimizer.param_groups[0][
                "capturable"
            ] = True

        self._epoch = epoch
        if fps is not None:
            self.fps = fps
        else:
            self.fps = self.get_dataset_specific_FPS(cfgs)
            print(
                f"fps not stored in model checkpoint. Use data-specific value {self.fps}",
            )

    def get_dataset_specific_FPS(self, cfgs):
        dataset = cfgs["general"]["dataset"]
        if dataset == c.WAYMO:
            fps = c.FPS_WAYMO
        else:
            raise NotImplementedError("Dataset not implemented")
        return fps

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

class ScenePredictionBaseClass(SceneBaseClass):
    def __init__(self,
                 configs,
                 ):
        super(ScenePredictionBaseClass, self).__init__(
            configs=configs)

        # General
        self.configs = configs
        self.batch_size = self.configs["hyperparameters"]["batch_size"]
        self.dataset = self.configs["general"]["dataset"]
        self.device = c.DEVICE

        # Game specific values
        self.num_modes = self.configs["hyperparameters"]["num_modes"]
        self.dynamics_model = self.setup_dynamics_model()
        self._set_dataset_specific_FPS(self.dataset)
        self.dt = 1 / self.fps

        # Losses and Metrics
        self.metrics = Metrics(self.device)

    def setup_dynamics_model(self):
        dynamic_model = UnicycleModel()
        return dynamic_model

    def _set_dataset_specific_FPS(self, configs):
        if self.dataset == c.WAYMO:
            self.fps = c.FPS_WAYMO
        else:
            raise NotImplementedError("Dataset not implemented")
        return self.fps

    def get_planner_init_state(self, mini_batch):
        if self.dataset == c.WAYMO:
            self.dynamics_model.set_car_geometries(mini_batch["agent_lengths"].to(self.device),
                                                   mini_batch["agent_widths"].to(self.device),
                                                   self.num_modes)
            game_init = einops.rearrange(mini_batch["init_state"], 'b p s -> b (p s)')
            game_init = game_init.repeat_interleave(self.num_modes, dim=0)

        else:
            raise NotImplementedError("Game init for this dataset not implemented")

        return game_init.to(self.device)


    def log_metrics(self, loss_dict, netmode, epoch, global_step):
        log_dict = {}
        # add all losses
        for key in loss_dict:
            log_dict[f"{netmode}/Losses/{key}"] = loss_dict[key]

        # add epoch and global step
        log_dict["epoch"] = epoch
        log_dict["global_step"] = global_step

        # add scene probs
        log_dict[f"{netmode}/Probs/Scene-Probs"] = self.scene_probs.view(-1).detach()

        wandb.log(log_dict)



    @abstractmethod
    def get_metrics_init(self):
        pass

    @abstractmethod
    def calculate_loss(self, mini_batch, backbone, netmode, global_step, epoch):
        pass

    @abstractmethod
    def predict_and_calculate_loss(self, mini_batch, netmode, global_step, epoch):
        pass

    @abstractmethod
    def extract_loss(self, loss_dict):
        pass


    def evaluate(self, data_loader, global_step, netmode, epoch):
        metrics_init = self.get_metrics_init()
        metrics_mean = {}
        eval_desc = "Validation: " if netmode == "val" else "Test: "
        self.net.eval()

        with torch.no_grad():
            for i, mini_batch in enumerate(tqdm(data_loader, desc=eval_desc, leave=False, position=1)):

                loss_dict, eval_dict = self.predict_and_calculate_loss(mini_batch, netmode=netmode, global_step=global_step, epoch=epoch)

                for key in metrics_init.keys():
                    metrics_init[key].extend(eval_dict[key].tolist())

            for key in metrics_init:
                metrics_mean[key] = sum(metrics_init[key]) / len(metrics_init[key])
            self.log_metrics(metrics_mean, netmode, epoch, global_step)

        self.net.train()
        return metrics_mean


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class SceneControlNetClass(ScenePredictionBaseClass):
    def __init__(self,
                 configs,
                 ):
        super(SceneControlNetClass, self).__init__(
            configs=configs)
        self.net = SceneControlNet(opts=self.configs, device=self.device)
        self.optimizer = create_optimizer(self, self.configs)
        if self.configs["config_params"]["lr_scheduler"] != 0:
            self.lr_scheduler = create_lr_scheduler(self, self.configs)

    def forward(self, sample):
        agent_data = sample['x'].float()
        lane_graph = sample['map'].float()
        fw_pass_controls = self.net(agent_data, lane_graph)
        return fw_pass_controls

    def predict_and_calculate_loss(self, mini_batch, netmode, global_step, epoch):
        # Set init state
        game_init = self.get_planner_init_state(mini_batch)

        # Neural network prediction
        fw_pass_controls = self.forward(sample=mini_batch)

        if self.configs["differentiable_optimization"]["control_init_w_zeros"]:
            # overwrite init
            fw_pass_controls = torch.zeros_like(fw_pass_controls)

        # Calculate losses
        loss_dict, eval_dict = self.calculate_loss(game_init, mini_batch, fw_pass_controls)

        if netmode == "train" and global_step % self.configs["wandb"]["log_step_training"] == 0:
            self.log_metrics(loss_dict, netmode=netmode, epoch=epoch, global_step = global_step)

        return loss_dict, eval_dict

    def extract_loss(self, loss_dict):
        return loss_dict["minSADE_bb"]  +  loss_dict["SCENE_LOSS"]

    def format_init_state(self, game_init):
        if self.dataset == c.WAYMO:
            current_state = einops.rearrange(game_init, "(b m) (p s) -> (b m) p 1 s", m=self.num_modes, s=self.dynamics_model.num_of_states)
        else:
            raise NotImplementedError
        return current_state

    def unroll_predictions_and_format_gt(self, mini_batch, backbone_controls, current_state):
        gt_trajectories = einops.rearrange(mini_batch["y"], "t b p s -> b p t s")[..., 0:2].to(self.device)
        b, _, t, s = gt_trajectories.shape

        # Get state trajectory by unrolling the dynamics model
        number_of_player = backbone_controls.shape[3]
        backbone_controls = einops.rearrange(backbone_controls, "m t b p c -> (b m) (p t c)")
        backbone_controls = einops.rearrange(backbone_controls, "bm (p t c) -> bm p t c", p=number_of_player, t=t,
                                             c=self.dynamics_model.num_of_controls)
        unrolled_trajectories = self.dynamics_model.state_transition(backbone_controls, current_state, self.dt)
        unrolled_trajectories = einops.rearrange(unrolled_trajectories, "(b m) p t s -> b m p t s", p=number_of_player, m=self.num_modes)
        unrolled_trajectories_OR = unrolled_trajectories[..., 0:3]
        unrolled_trajectories = unrolled_trajectories[..., 0:2]

        return gt_trajectories, unrolled_trajectories, unrolled_trajectories_OR

    def calculate_loss(self, game_init, mini_batch, vn_controls):

        # Get current state of all agents
        current_state = self.format_init_state(game_init)
        gt_trajectories, unrolled_trajectories, unrolled_trajectories_OR = self.unroll_predictions_and_format_gt(mini_batch, vn_controls, current_state)


        # Calculate losses / metrics
        minSADE_bb, ind_sade_bb, eval_minSADE_bb = self.metrics.minSADE(gt_trajectories, unrolled_trajectories)
        minSFDE_bb, ind_sfde_bb, eval_minSFDE_bb = self.metrics.minSFDE(gt_trajectories, unrolled_trajectories)
        minADE_bb, ind_ade_bb, eval_minADE_bb = self.metrics.minADE(gt_trajectories, unrolled_trajectories)
        minFDE_bb, ind_min_FDE_bb, eval_minFDE_bb = self.metrics.minFDE(gt_trajectories, unrolled_trajectories)

        if self.configs["hyperparameters"]["num_modes"] > 1:
            scene_probs, scene_prob_loss = self.metrics.predict_and_calc_scene_prob_loss(mini_batch, self.net, unrolled_trajectories)
        else:
            scene_probs = torch.ones(1, self.num_modes).to(self.device)
            scene_prob_loss = torch.zeros(1).to(self.device)

        # Calculate overlap rate
        if self.configs["metrics"]["overlap_rate"]:
            overlap_rate, overlap_eval = self.metrics.overlap_rate(unrolled_trajectories_OR, scene_probs, self.dynamics_model)

        # Set values for
        self.set_class_values_for_further_processing(unrolled_trajectories, current_state, scene_probs)

        loss_dict = {
            "minSADE_bb": minSADE_bb,
            "minSFDE_bb": minSFDE_bb,
            "minADE_bb": minADE_bb,
            "minFDE_bb": minFDE_bb,
            "SCENE_LOSS": scene_prob_loss,
        }
        eval_dict = {
            "minSADE_bb": eval_minSADE_bb,
            "minSFDE_bb": eval_minSFDE_bb,
            "minADE_bb": eval_minADE_bb,
            "minFDE_bb": eval_minFDE_bb,
        }
        if self.configs["metrics"]["overlap_rate"]:
            eval_dict["OR_th"] = overlap_eval
        return loss_dict, eval_dict

    def set_class_values_for_further_processing(self, unrolled_trajectories, current_state, scene_probs):
        self.theseus_trajectory_all = unrolled_trajectories[..., 0:3]  # needed for plottin, dummy values
        self.backbone_trajectory = unrolled_trajectories[..., 0:3]
        self.start_state = current_state
        self.scene_probs = scene_probs

    def get_metrics_init(self):
        metrics_init = {
            'minSADE_bb': [],
            'minSFDE_bb': [],
            'minADE_bb': [],
            'minFDE_bb': [],
        }
        if self.configs["metrics"]["overlap_rate"]:
            metrics_init["OR_th"] = []
        return metrics_init

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class EPONetFClass(ScenePredictionBaseClass):
    def __init__(self, configs):

        super(EPONetFClass, self).__init__(configs=configs)
        self.game_objective = MultiAgentGameObjective(self.dynamics_model, configs, self.fps)
        self.configs = self.set_num_game_weights_in_config(configs) # this needs to be called before initializing the net, because we overwrite the config based on the objective
        self.net = EPONetF(opts=self.configs, device=self.device)
        self.optimizer = create_optimizer(self, self.configs)
        if self.configs["config_params"]["lr_scheduler"] != 0:
            self.lr_scheduler = create_lr_scheduler(self, self.configs)

        self.planner_inputs = {
            "opt_cntrl": torch.zeros((self.game_objective.batch_size, self.game_objective.optim_input_dof))
        }

        self.setup_optimization_hyperparameters()
        self.setup_optimizer()

        # Setup Theseus Layer
        self.setup_layer()

    # *  #############################################################################  *#
    # *  ############################## THESEUS LAYER ################################  *#
    # *  #############################################################################  *#

    def setup_optimization_hyperparameters(self):
        self.target_point = self.configs["loss"]["goal_point_mode"]
        self.verbose = self.configs["differentiable_optimization"]["verbose"]
        self.optim_method = self.configs["differentiable_optimization"]["optim_method"]
        self.max_train_iter = self.configs["differentiable_optimization"]["max_train_iter"]
        self.max_eval_iter = self.configs["differentiable_optimization"]["max_eval_iter"]
        self.step_size_train = self.configs["differentiable_optimization"]["train_step_size"]
    def setup_layer(self):
        """
            Takes the Optimizer and initializes an TheseusLayer and place it on the GPU
        """
        print("\n############################################")
        print("############### THESEUS LAYER ##############")
        print("############################################\n")
        print("->", self.game_objective.obj_func.size_cost_functions(), "Cost-Functions")
        print("->", self.game_objective.obj_func.size_aux_vars(), "Auxiliary Variables")
        print("->", self.game_objective.obj_func.size_variables(), "Optimization Variable")
        print("-->", self.game_objective.number_player, "Player")
        print("-->", self.game_objective.num_of_modes, "Modes")
        print(f"--> {self.game_objective.total_time} Game Time @{self.game_objective.fps} FPS ({self.game_objective.dt}s ΔT) with {self.game_objective.planning_horizon} Horizon")
        print("-->", self.game_objective.optim_state_dof, "States DOF /", self.game_objective.optim_input_dof, "Input DOF")
        self.game_motion_planner_train_layer = th.TheseusLayer(self.train_optimizer)
        self.game_motion_planner_eval_layer = th.TheseusLayer(self.eval_optimizer)
        self.game_motion_planner_train_layer.to(device=self.device)
        self.game_motion_planner_eval_layer.to(device=self.device)

    def setup_optimizer(self):
        """
        setup the optimizer for the Theseus Layer based on configs parameters

        """
        if self.verbose: print("Setup Optimizer")
        if self.optim_method == "LM":
            self.train_optimizer = th.LevenbergMarquardt(
                self.game_objective.obj_func,
                linear_solver_cls=th.LUDenseSolver,
                step_size=self.step_size_train,
                max_iterations=self.max_train_iter)

            self.eval_optimizer = th.LevenbergMarquardt(
                self.game_objective.obj_func,
                linear_solver_cls=th.LUDenseSolver,
                step_size=self.step_size_train,
                max_iterations=self.max_eval_iter)

        elif self.optim_method == "GN":
            self.train_optimizer = th.GaussNewton(
                self.game_objective.obj_func,
                linear_solver_cls=th.LUDenseSolver,
                step_size=self.step_size_train,
                max_iterations=self.max_train_iter,
                abs_err_tolerance=1e-2)

            self.eval_optimizer = th.GaussNewton(
                self.game_objective.obj_func,
                linear_solver_cls=th.LUDenseSolver,
                step_size=self.step_size_train,
                max_iterations=self.max_eval_iter,
                abs_err_tolerance=1e-2)


    def set_num_game_weights_in_config(self, configs):
        """
        Overwrites tnum_int_costs in config dynamically
        :param configs: current_config
        :param scene_prediction_model: model
        :return:
        """
        # Initialization of current number of output params for decoders
        # Assume N player -> we have #INTERACTIONS = (N * (N - 1)) /2 pairwise interactions
        # e.g., N = 2 -> I = 1,
        # N = 3 -> I = 3,
        # N = 4 -> I = 6, ....
        num_player = configs["differentiable_optimization"]["number_player"]
        num_of_int_costs = int((num_player * (num_player - 1)) / 2)
        configs["differentiable_optimization"]["num_int_weights"] = num_of_int_costs

        num_of_own_costs = self.game_objective.obj_func.size()[0] - 1
        configs["differentiable_optimization"]["num_own_weights"] = num_of_own_costs

        return configs


    def forward(self, sample):
        agent_data = sample['x'].float()
        lane_graph = sample['map'].float()
        predictions, weights_own, weights_int, target_points, _ = self.net(
            agent_data,
            lane_graph
        )
        return predictions, weights_own, weights_int, target_points

    def predict_and_calculate_loss(self, mini_batch, netmode, global_step, epoch):

        # set init state
        game_init = self.get_planner_init_state(mini_batch)

        self.game_objective.set_lanes(mini_batch) # currently not in use

        # neural network prediciton
        vn_controls, vn_own_weights, vn_int_weights, game_goal = self.forward(mini_batch)

        if self.configs["differentiable_optimization"]["control_init_w_zeros"]:
            vn_controls = torch.zeros_like(vn_controls)


        # set game_goalscene target prediction
        self.game_objective.update_boundary_points(game_init, game_goal)

        # update the weights
        self.game_objective.update_weights(weights_own=vn_own_weights, weights_int=vn_int_weights)

        # update controls
        vn_controls = einops.rearrange(vn_controls, "m t b p c -> (b m) (p t c)")  # format controls
        self.update_controls(vn_controls)

        # energy optimization (game-theoretic joint optimization)
        self.energy_optimization(optimization_inputs=self.planner_inputs, netmode=netmode)

        # loss calculation
        loss_dict, eval_dict = self.calculate_loss(mini_batch, vn_controls, game_goal, netmode, global_step, epoch)

        # logging
        if netmode == "train" and global_step % self.configs["wandb"]["log_step_training"] == 0:
            self.log_metrics(loss_dict, netmode=netmode, epoch=epoch, global_step = global_step)

        return loss_dict, eval_dict

    def update_controls(self, vn_controls):
        self.planner_inputs.update([
            ('opt_cntrl', vn_controls)
        ])

    def calculate_loss(self, mini_batch, backbone_controls, game_goal, netmode, global_step, epoch):
        """
        Calculates losses and metrics based on current predictions
        :param mini_batch: mini_batch dict
        :param backbone_controls: initial trajectories
        :param game_goal: predicted game_goal
        :param netmode: train or val or test
        :param global_step: global step
        :param epoch: current epoch
        """

        # Calculate distance based trajectory losses / metrics
        th_train_losses, bb_train_losses, th_eval_metrics, bb_eval_metrics = self.calc_trajectory_losses(mini_batch["y"], backbone_controls=backbone_controls)

        if self.configs["hyperparameters"]["num_modes"] > 1:
            scene_probs, scene_prob_loss = self.metrics.predict_and_calc_scene_prob_loss(mini_batch, self.net, self.theseus_trajectory)
            self.scene_probs = scene_probs
        else:
            scene_probs = torch.ones(1, 1).to(self.device)
            self.scene_probs = scene_probs
            scene_prob_loss = torch.zeros(1).to(self.device)

        # Calculate overlap rate
        if self.configs["metrics"]["overlap_rate"]:
            overlap_rate_th, overlap_eval_th = self.metrics.overlap_rate(self.theseus_trajectory_all, self.scene_probs, self.dynamics_model)
            overlap_rate_bb, overlap_eval_bb = self.metrics.overlap_rate(self.backbone_trajectory_all, self.scene_probs, self.dynamics_model)

        # Endpoint Loss
        if "B" in self.configs["differentiable_optimization"]["cost_token"]:
            endpoint_loss = self.metrics.calc_target_loss(mini_batch, game_goal)
        else:
            endpoint_loss = torch.zeros(1).to(self.device)

        # Calculate weights final loss
        loss = self.calc_loss_based_on_loss_mode(th_train_losses, bb_train_losses, endpoint_loss, scene_prob_loss)

        loss_dict = {
            # Theseus
            "minSADE_th": th_train_losses[0],
            "minSFDE_th": th_train_losses[1],
            "minADE_th": th_train_losses[2],
            "minFDE_th": th_train_losses[3],

            # Backbone
            "minSADE_bb": bb_train_losses[0],
            "minSFDE_bb": bb_train_losses[1],
            "minADE_bb": bb_train_losses[2],
            "minFDE_bb": bb_train_losses[3],

            # Losses
            "SCENE_PROB_LOSS": scene_prob_loss,
            "ENDPOINT_LOSS": endpoint_loss,
            "MULTITASK_LOSS": loss.type(torch.float32),
        }

        eval_dict = {
            # Theseus
            "minSADE_th": th_eval_metrics[0],
            "minSFDE_th": th_eval_metrics[1],
            "minADE_th": th_eval_metrics[2],
            "minFDE_th": th_eval_metrics[3],

            # Backbone
            "minSADE_bb": bb_eval_metrics[0],
            "minSFDE_bb": bb_eval_metrics[1],
            "minADE_bb": bb_eval_metrics[2],
            "minFDE_bb": bb_eval_metrics[3],
        }
        if self.configs["metrics"]["overlap_rate"]:
            eval_dict["OR_th"] = overlap_eval_th
            eval_dict["OR_bb"] = overlap_eval_bb
        return loss_dict, eval_dict

    def calc_loss_based_on_loss_mode(self, th_train_losses, bb_train_losses, endpoint_loss, scene_prob_loss):
        if self.configs["loss"]["multi_task_loss_mode"] == "MT":
            loss = self.calc_multitask_loss(th_train_losses, bb_train_losses, endpoint_loss, scene_prob_loss)
        else:
            raise NotImplementedError("Loss mode not implemented, currently only MT supported. Please check the config.")
        return loss

    def calc_multitask_loss(self, th_train_losses, bb_train_losses, endpoint_loss, scene_prob_loss):
        loss = th_train_losses[0] \
        + self.configs["loss"]["weight_goal_loss"] * endpoint_loss \
        + self.configs["loss"]["weight_scene_prob_loss"] * scene_prob_loss
        return loss

    def extract_loss(self, loss_dict):
        return loss_dict["MULTITASK_LOSS"]


    def get_metrics_init(self):
        metrics_init = {

            # Optimization (Theseus)
            'minSADE_th': [],
            'minSFDE_th': [],
            'minADE_th': [],
            'minFDE_th': [],

            # Backbone (Init)
            'minSADE_bb': [],
            'minSFDE_bb': [],
            'minADE_bb': [],
            'minFDE_bb': [],
        }
        if self.configs["metrics"]["overlap_rate"]:
            metrics_init["OR_th"] = []
            metrics_init["OR_bb"] = []
        return metrics_init


    def calc_trajectory_losses(self, gt_trajectories, backbone_controls):
        """
        Calculates the trajectory losses for the optimized and backbone trajectories
        :param gt_trajectories: ground truth joint state trajectories
        :param backbone_controls: initial joint control trajectories
        :return:
        """

        # Format gt and current state
        gt_trajectories = einops.rearrange(gt_trajectories, "t b p s -> b p t s")[..., 0:2].to(self.device) # ground truth reshaping
        current_state = einops.rearrange(self.game_objective.start_states.tensor, "b (p t s) -> b p t s", p=self.game_objective.number_player, t=1).to(self.device)

        # Unroll initial controls to get initial state trajectories
        backbone_controls = einops.rearrange(backbone_controls, "b (p t c) -> b p t c", p=self.game_objective.number_player, t=self.game_objective.planning_horizon,
                                             c=self.game_objective.num_of_inputs)  # backbone processing
        backbone_trajectory = self.dynamics_model.state_transition(backbone_controls, current_state, self.game_objective.dt)
        backbone_trajectory = einops.rearrange(backbone_trajectory, "(b m) p t s -> b m p t s", p=self.game_objective.number_player, m=self.game_objective.num_of_modes)
        self.backbone_trajectory_all = backbone_trajectory[..., 0:3]
        self.backbone_trajectory = backbone_trajectory[..., 0:2]

        # Unroll optimized controls to get optimized state trajectories
        theseus_controls = self.game_objective.obj_func.optim_vars["opt_cntrl"]
        theseus_controls = einops.rearrange(theseus_controls.tensor, "b (p t c) -> b p t c", p=self.game_objective.number_player, t=self.game_objective.planning_horizon,
                                            c=self.game_objective.num_of_inputs)
        theseus_trajectory = self.dynamics_model.state_transition(theseus_controls, current_state, self.game_objective.dt)
        theseus_trajectory = einops.rearrange(theseus_trajectory, "(b m) p t s -> b m p t s", p=self.game_objective.number_player, m=self.game_objective.num_of_modes)
        self.theseus_trajectory_all = theseus_trajectory[..., 0:3]
        self.theseus_trajectory = theseus_trajectory[..., 0:2]

        # Calculate losses of optimized trajectories
        minSADE_th, ind_sade_th, eval_minSADE_th = self.metrics.minSADE(gt_trajectories, theseus_trajectory[..., 0:2])
        minSFDE_th, ind_sfde_th, eval_minSFDE_th = self.metrics.minSFDE(gt_trajectories, theseus_trajectory[..., 0:2])
        minADE_th, ind_ade_th, eval_minADE_th = self.metrics.minADE(gt_trajectories, theseus_trajectory[..., 0:2])
        minFDE_th, ind_minFDE_th, eval_minFDE_th = self.metrics.minFDE(gt_trajectories, theseus_trajectory[..., 0:2])

        # Calculate losses of the initial trajectories backbone
        minSADE_bb, ind_sade_bb, eval_minSADE_bb = self.metrics.minSADE(gt_trajectories, backbone_trajectory)
        minSFDE_bb, ind_sfde_bb, eval_minSFDE_bb = self.metrics.minSFDE(gt_trajectories, backbone_trajectory)
        minADE_bb, ind_ade_bb, eval_minADE_bb = self.metrics.minADE(gt_trajectories, backbone_trajectory)
        minFDE_bb, ind_min_FDE_bb, eval_minFDE_bb = self.metrics.minFDE(gt_trajectories, backbone_trajectory)

        # Structure output
        theseus_train_losses = [minSADE_th, minSFDE_th, minADE_th, minFDE_th]
        backbone_train_losses = [minSADE_bb, minSFDE_bb, minADE_bb, minFDE_bb]
        theseus_eval_metrics = [eval_minSADE_th, eval_minSFDE_th, eval_minADE_th, eval_minFDE_th]
        backbone_eval_metrics = [eval_minSADE_bb, eval_minSFDE_bb, eval_minADE_bb, eval_minFDE_bb]

        return theseus_train_losses, backbone_train_losses, theseus_eval_metrics, backbone_eval_metrics


    def update_obj_controls_weights(self, control, weights):
        """
        Updates the controls and game weights of the optimization problem
        """
       # Update weights
        self.game_objective.update_weights(weights_own=weights["vn_own_weights"], weights_int=weights["vn_int_weights"])

        # Update controls
        self.planner_inputs.update([
            (
                'opt_cntrl',
                control.type(torch.float32).to("cuda:0"))
        ])

        self.game_objective.update_controls(self.planner_inputs)


    def energy_optimization(self, optimization_inputs, netmode):
        """
        performs the joint energy optimization
        :param optimization_inputs: inputs to the layer
        :param netmode: mode of the network
        """
        if netmode == "val" or netmode == "test":
            with torch.no_grad():
                self.final_values, self.info = self.game_motion_planner_eval_layer.forward(
                    optimization_inputs,
                    optimizer_kwargs={
                        "track_best_solution": False,
                        "verbose": self.verbose,
                        "adaptive_damping": self.configs["differentiable_optimization"]["adaptive_damping"],
                        "ellipsoidal_damping": self.configs["differentiable_optimization"]["ellipsoidal_damping"],
                        "damping": self.configs["differentiable_optimization"]["LM_damping"],
                        "track_err_history": True,
                        "track_state_history": True,
                    }
                )
        if netmode == "train":
            self.final_values, self.info = self.game_motion_planner_train_layer.forward(
                optimization_inputs,
                optimizer_kwargs={
                    "track_best_solution": False,
                    "verbose": self.verbose,
                    "adaptive_damping": self.configs["differentiable_optimization"]["adaptive_damping"],
                    "ellipsoidal_damping": self.configs["differentiable_optimization"]["ellipsoidal_damping"],
                    "damping": self.configs["differentiable_optimization"]["LM_damping"],
                    "track_err_history": True,
                    "track_state_history": True,
                }
            )







