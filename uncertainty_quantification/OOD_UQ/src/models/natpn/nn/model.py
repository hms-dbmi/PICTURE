from typing import Tuple
import torch
from torch import nn
import src.models.natpn.distributions as D
from .flow import NormalizingFlow
from .output import Output
from .scaler import CertaintyBudget, EvidenceScaler,EvidenceScalerNoClamp


class NaturalPosteriorNetworkModel(nn.Module):
    """
    Implementation of the NatPN module. This class only describes the forward pass through the
    model and can be compiled via TorchScript.
    """

    def __init__(
        self,
        latent_dim: int,
        encoder: nn.Module,
        flow: NormalizingFlow,
        output: Output,
        clamp_scaling: bool = True,
        certainty_budget: CertaintyBudget = "normal",
        locked_encoder: bool = False,
    ):
        """
        Args:
            latent_dim: The dimension of the latent space to which the model's encoder maps.
            config: The model's intrinsic configuration.
            encoder: The model's (deep) encoder which maps input to a latent space.
            flow: The model's normalizing flow which yields the evidence of inputs based on their
                latent representations.
            output: The model's output head which maps each input's latent representation linearly
                to the parameters of the target distribution.
            certainty_budget: The scaling factor for the certainty budget that the normalizing
                flow can draw from.
        """
        super().__init__()
        self.encoder = encoder
        self.flow = flow
        self.output = output
        self.scaler = EvidenceScaler(latent_dim, certainty_budget) if clamp_scaling else EvidenceScalerNoClamp(latent_dim, certainty_budget)
        self.locked_encoder = locked_encoder

    def aleatoric_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the aleatoric confidence of the model for each input independently.
        Scaled between 0 and 1, with 1 being the most confident of being ID and 0 being the most
        confident of being OOD.

        Args:
            x: The inputs that are first passed to the encoder.

        Returns:
            The per-input aleatoric confidence.
        """
        posterior, _ = self.forward(x)
        aleatoric_conf = -posterior.maximum_a_posteriori().uncertainty()

        return torch.sigmoid(aleatoric_conf)

    def epistemic_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the epistemic confidence of the model for each input independently.
        Scaled between 0 and 1, with 1 being the most confident of being ID and 0 being the most
        confident of being OOD.

        Args:
            x: The inputs that are first passed to the encoder.

        Returns:
            The per-input epistemic confidence.
        """
        _, log_prob = self.forward(x)

        return torch.sigmoid(log_prob)


    def forward(self, x: torch.Tensor) -> Tuple[D.Posterior, torch.Tensor]:
        """
        Performs a Bayesian update over the target distribution for each input independently. The
        returned posterior distribution carries all information about the prediction.

        Args:
            x: The inputs that are first passed to the encoder.

        Returns:
            The posterior distribution for every input along with their log-probabilities. The
            same probabilities are returned from :meth:`log_prob`.
        """
        update, log_prob = self.posterior_update(x)
        return self.output.prior.update(update), log_prob

    def posterior_update(self, x: torch.Tensor) -> Tuple[D.PosteriorUpdate, torch.Tensor]:
        """
        Computes the posterior update over the target distribution for each input independently.

        Args:
            x: The inputs that are first passed to the encoder.

        Returns:
            The posterior update for every input and the true log-probabilities.
        """
        if self.locked_encoder:
            with torch.no_grad():
                z = self.encoder.forward(x)
        else:
            z = self.encoder.forward(x)
            
        if z.dim() > 2:
            z = z.permute(0, 2, 3, 1)
        prediction = self.output.forward(z)
        sufficient_statistics = prediction.expected_sufficient_statistics()

        log_prob = self.flow.forward(z)
        log_evidence = self.scaler.forward(log_prob)

        return D.PosteriorUpdate(sufficient_statistics, log_evidence), log_prob

    def log_prob(self, x: torch.Tensor, track_encoder_gradients: bool = True) -> torch.Tensor:
        """
        Computes the (scaled) log-probability of observing the given inputs.

        Args:
            x: The inputs that are first passed to the encoder.
            track_encoder_gradients: Whether to track the gradients of the encoder.

        Returns:
            The per-input log-probability.
        """
        with torch.set_grad_enabled(self.training and track_encoder_gradients):
            z = self.encoder.forward(x)
            if z.dim() > 2:
                z = z.permute(0, 2, 3, 1)
        return self.flow.forward(z)
