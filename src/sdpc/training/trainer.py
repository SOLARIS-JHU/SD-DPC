"""Thresholded-gradient-descent trainer with iterative pruning (Algorithm 1).

A faithful copy of ``SparseDPC.vectorized.trainer.SparseTrainer`` and its
``move_batch_to_device`` helper: the training loop that alternates AdamW steps on the
rollout/DPC loss with periodic hard-thresholding of the SINDy/policy coefficients
(Sec. 3.1). Used for both system identification and sparse policy learning.
"""
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from neuromancer.loggers import BasicLogger
from neuromancer.problem import Problem
from neuromancer.callbacks import Callback

from ..sindy.model import SINDyVectorized


def move_batch_to_device(batch: dict, device: str = "cpu") -> dict:
    """
    Move batch data to target device.

    Args:
        batch: Dictionary of data, potentially including torch.Tensor values.
        device: Target device (e.g., "cuda", "cpu")

    Returns:
        Dictionary with tensors moved to the specified device.
    """
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class SparseTrainer:
    """
    Trainer for sparse models with iterative pruning using learned coefficients (SINDy-style).

    Args:
        problem: Neuromancer Problem instance
        fx_models: List of sparse models (e.g., SINDy policies) with .coef and .library
        lr: Learning rate
        train_data: Training data loader
        dev_data: Validation data loader
        test_data: Test data loader
        optimizers: Optional list of optimizers (one per fx model)
        logger: Logger for metrics (e.g., wandb or basic)
        callback: Neuromancer Callback for training hooks
        lr_scheduler: Enable the configured step learning-rate decay
        epochs: Number of epochs
        patience: Early stopping patience
        prune_every: Frequency of coefficient pruning
        threshold: Coefficient pruning threshold
        threshold_mult: Threshold growth factor after each prune
        prune_noise: Small noise added to surviving coefficients after prune
        max_unchange: Stop after this many consecutive prunes leave every output size
            unchanged; zero disables this stopping rule
        device: Device for training
    """

    def __init__(self,
                 problem: Problem,
                 sindy: SINDyVectorized,
                 lr: float,
                 train_data: DataLoader = None,
                 dev_data: DataLoader = None,
                 test_data: DataLoader = None,
                 optimizers= None,
                 logger: BasicLogger = None,
                 callback: Callback = Callback(),
                 lr_scheduler: bool = False,
                 epochs: int = 1000,
                 epoch_verbose: int = 1,
                 patience: int = 5,
                 warmup: int = 0,
                 train_metric: str = "train_loss",
                 dev_metric: str = "dev_loss",
                 test_metric: str = "test_loss",
                 eval_metric: str = "dev_loss",
                 eval_mode: str = "min",
                 clip: float = 100.0,
                 multi_fidelity: bool = False,
                 device : torch.device = torch.device("cpu"),
                 prune_every: int = 5,
                 prune_every_decay: int = 200,
                 prune_every_min: int = 10,
                 prune_noise: float = 0.05,
                 change_prune_every: bool = False,
                 threshold_mult: float = 1.07,
                 threshold: float = 1e-3,
                 threshold_max: float = 1e0,
                 max_unchange: int = 0,
                 l1_decay: float = 1.0,
                 weight_decay: float = 0.01,
                 lr_decay_step: int = 2000,
                 lr_decay_gamma: float = 0.8,):

        self.model = problem
        self.sindy = sindy
        self.lr = lr
        self.threshold = threshold
        self.threshold_max = threshold_max
        self.threshold_mult = threshold_mult
        self.prune_every = prune_every
        self.change_prune_every = change_prune_every
        self.prune_every_decay = prune_every_decay
        self.prune_every_min = prune_every_min
        self.prune_noise = prune_noise
        self.max_unchange = self._validate_max_unchange(max_unchange)
        self.unchanged_prune_count = 0
        self.l1_decay = l1_decay
        self.weight_decay = float(weight_decay)

        self.optimizers = optimizers
        self.lr_decay_step = lr_decay_step
        self.lr_decay_gamma = lr_decay_gamma
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.logger = logger
        self.callback = callback
        self.epochs = epochs
        self.current_epoch = 0
        self.epoch_verbose = epoch_verbose
        self.train_metric = train_metric
        self.dev_metric = dev_metric
        self.test_metric = test_metric
        self.eval_metric = eval_metric
        self._eval_min = eval_mode == "min"
        self.clip = clip
        self.device = device
        self.multi_fidelity = multi_fidelity
        self.patience = patience
        self.warmup = warmup
        self.badcount = 0
        self.best_devloss = float("inf") if self._eval_min else 0.
        self.best_model = self.model.state_dict()
        self.prev_active_counts = [parameter.numel() for parameter in self.sindy.Xi]

        self.lr_scheduler_enabled = bool(lr_scheduler)
        self._validate_lr_schedule()

    @staticmethod
    def _validate_max_unchange(value) -> int:
        if value is None:
            return 0
        if isinstance(value, bool):
            raise ValueError("max_unchange must be a nonnegative integer")
        try:
            normalized = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_unchange must be a nonnegative integer") from exc
        if normalized < 0 or normalized != value:
            raise ValueError("max_unchange must be a nonnegative integer")
        return normalized

    def _record_prune_counts(self, retained_counts) -> bool:
        """Update unchanged-prune state and report whether its limit was reached."""
        counts = [int(count) for count in retained_counts]
        if counts == self.prev_active_counts:
            self.unchanged_prune_count += 1
        else:
            self.unchanged_prune_count = 0
        self.prev_active_counts = counts
        return (
            self.max_unchange > 0
            and self.unchanged_prune_count >= self.max_unchange
        )

    def _validate_lr_schedule(self) -> None:
        if not self.lr_scheduler_enabled:
            return
        if self.lr_decay_step <= 0:
            raise ValueError("lr_decay_step must be positive when LR decay is enabled")
        if not 0.0 < self.lr_decay_gamma <= 1.0:
            raise ValueError("lr_decay_gamma must lie in (0, 1]")

    def _scheduled_learning_rate(self, completed_epochs: int) -> float:
        if not self.lr_scheduler_enabled:
            return float(self.lr)
        decay_count = int(completed_epochs) // self.lr_decay_step
        return float(self.lr) * self.lr_decay_gamma ** decay_count

    def _set_scheduled_learning_rate(self, completed_epochs: int) -> float:
        learning_rate = self._scheduled_learning_rate(completed_epochs)
        for group in self.optimizers.param_groups:
            group["lr"] = learning_rate
        return learning_rate

    @staticmethod
    def _mean_component_batches(component_batches: dict) -> dict:
        return {
            name: torch.stack(values).mean()
            for name, values in component_batches.items()
            if values
        }

    @staticmethod
    def _total_control_action_l1(u: torch.Tensor) -> torch.Tensor:
        """Mean trajectory-wise sum of absolute controls over time and channels."""
        if u.ndim < 2:
            raise ValueError("control rollout must include batch and action dimensions")
        return u.abs().sum(dim=tuple(range(1, u.ndim))).mean()

    def _named_loss_components(self, output: dict, split: str) -> dict:
        """Extract weighted, named Neuromancer losses from one batch output."""
        aliases = {
            "reference_loss_position": "tracking_loss",
            "loss_l1_policy": "l1_loss",
            "loss_l1_sindy": "l1_loss",
        }
        components = {}
        l1_loss = None
        for objective in self.model.loss.objectives:
            key = f"{split}_{objective.output_keys[0]}"
            if key not in output:
                continue
            name = aliases.get(objective.name, objective.name)
            if not name.endswith("_loss"):
                name = f"{name}_loss"
            value = output[key]
            components[f"{split}_{name}"] = value
            if name == "l1_loss":
                l1_loss = value

        constraint_values = []
        for constraint in self.model.loss.constraints:
            key = f"{split}_{constraint.output_keys[0]}"
            if key not in output:
                continue
            value = output[key]
            components[f"{split}_{constraint.name}_loss"] = value
            constraint_values.append(value)
        if constraint_values:
            components[f"{split}_constraint_loss"] = sum(constraint_values)

        total = output.get(f"{split}_loss")
        if total is not None:
            components[f"{split}_task_loss"] = (
                total - l1_loss if l1_loss is not None else total
            )
        u = output.get(f"{split}_u")
        if isinstance(u, torch.Tensor):
            components[f"{split}_total_control_action_l1"] = (
                self._total_control_action_l1(u)
            )
        return {name: value.detach() for name, value in components.items()}

    def train(self) -> dict:
        """
        Full training loop with pruning, early stopping, logging.
        Returns:
            Best model state_dict based on eval_metric.
        """
        self.callback.begin_train(self)

        try:
            for ep in range(self.current_epoch, self.current_epoch + self.epochs):
                self.model.train()
                batch_losses = []
                train_components = {}
                stop_for_unchanged_terms = False

                for batch in self.train_data:
                    batch["epoch"] = ep
                    batch = move_batch_to_device(batch, self.device)
                    out = self.model(batch)
                    for name, value in self._named_loss_components(out, "train").items():
                        train_components.setdefault(name, []).append(value)

                    if self.multi_fidelity:
                        for node in self.model.nodes:
                            out[self.train_metric] += node.callable.get_alpha_loss()

                    self.optimizers.zero_grad()
                    out[self.train_metric].backward()
                    clip_grad_norm_(self.model.parameters(), self.clip)
                    self.optimizers.step()

                    batch_losses.append(out[self.train_metric].detach())
                    self.callback.end_batch(self, out)

                out[f"mean_{self.train_metric}"] = torch.mean(torch.stack(batch_losses))
                out[self.train_metric] = out[f"mean_{self.train_metric}"]
                out.update(self._mean_component_batches(train_components))
                self.callback.begin_epoch(self, out)

                completed_epochs = ep + 1
                learning_rate = self._set_scheduled_learning_rate(completed_epochs)

                if ep % self.prune_every == 0 and ep > 0:
                    print("*"*75)
                    print("SINDy BEFORE pruning: ")

                    print("Terms BEFORE: ", [p.shape[0] for p in self.sindy.parameters()])
                    self.sindy.pretty_print()

                    self.sindy.prune_(thresholds = self.threshold, noise_after = self.prune_noise)
                    self.optimizers = torch.optim.AdamW(
                        self.sindy.parameters(), lr=learning_rate,
                        weight_decay=self.weight_decay,
                    )

                    retained_after_prune = [parameter.numel() for parameter in self.sindy.Xi]
                    stop_for_unchanged_terms = self._record_prune_counts(
                        retained_after_prune
                    )


                    print("*"*75)
                    print("SINDy AFTER pruning: ")
                    print("Terms AFTER: ", [p.shape[0] for p in self.sindy.parameters()])

                    self.sindy.pretty_print()
                    print("*"*75)
                    self.model.loss.objectives[-1].weight *= self.l1_decay
                    print("New L1 weight: ", self.model.loss.objectives[-1].weight)
                    print("New pruning threshold: ", self.threshold)
                    if self.max_unchange > 0:
                        print(
                            "Unchanged pruning count: "
                            f"{self.unchanged_prune_count}/{self.max_unchange}"
                        )
                    print("*"*75)

                    self.threshold = min(self.threshold_max, self.threshold * self.threshold_mult)
                    if self.change_prune_every:
                        self.prune_every = max(self.prune_every_min, self.prune_every - self.prune_every_decay)

                out["learning_rate"] = torch.as_tensor(
                    self.optimizers.param_groups[0]["lr"], device=self.device
                )
                active_counts = [
                    int(torch.count_nonzero(parameter.detach()).item())
                    for parameter in self.sindy.Xi
                ]
                retained_counts = [parameter.numel() for parameter in self.sindy.Xi]
                out["active_terms_total"] = torch.as_tensor(
                    sum(active_counts), device=self.device
                )
                for index, count in enumerate(active_counts):
                    out[f"active_terms_u{index}"] = torch.as_tensor(
                        count, device=self.device
                    )
                    out[f"retained_terms_u{index}"] = torch.as_tensor(
                        retained_counts[index], device=self.device
                    )
                out["coefficient_l1_norm"] = torch.stack(
                    [parameter.detach().abs().sum() for parameter in self.sindy.Xi]
                ).sum()
                out["unchanged_prune_count"] = torch.as_tensor(
                    self.unchanged_prune_count, device=self.device
                )

                with torch.set_grad_enabled(self.model.grad_inference):
                    self.model.eval()
                    if self.dev_data:
                        dev_losses = []
                        dev_components = {}
                        for d in self.dev_data:
                            d = move_batch_to_device(d, self.device)
                            dev_out = self.model(d)
                            dev_losses.append(dev_out[self.dev_metric].detach())
                            for name, value in self._named_loss_components(
                                dev_out, "dev"
                            ).items():
                                dev_components.setdefault(name, []).append(value)
                        dev_out[f"mean_{self.dev_metric}"] = torch.mean(torch.stack(dev_losses))
                        dev_out[self.dev_metric] = dev_out[f"mean_{self.dev_metric}"]
                        dev_out.update(self._mean_component_batches(dev_components))
                        out.update(dev_out)

                    self.callback.begin_eval(self, out)

                    self.best_model = {
                        k: v.clone().detach() for k, v in self.model.state_dict().items()
                    }
                    self.best_devloss = out[self.eval_metric]
                    self.badcount = 0

                    if self.logger:
                        self.logger.log_metrics(out, step=ep)

                    self.callback.end_eval(self, out)
                    self.callback.end_epoch(self, out)

                self.current_epoch = ep + 1
                if stop_for_unchanged_terms:
                    print(
                        "Stopping training: retained term counts "
                        f"{self.prev_active_counts} were unchanged for "
                        f"{self.unchanged_prune_count} consecutive pruning processes."
                    )
                    break

        except KeyboardInterrupt:
            print("Training interrupted.")

        self.callback.end_train(self, out)
        self.model.load_state_dict(self.best_model, strict=False)
        return self.best_model
