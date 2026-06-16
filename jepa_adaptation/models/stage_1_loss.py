import torch
import torch.nn as nn

from .discriminator import AdversarialLoss

try:
    from .modules.losses import LPIPS
except ImportError:
    from modules.losses import LPIPS


class Stage1Loss(nn.Module):
    def __init__(
        self,
        disc_start,
        kl_weight=1.0,
        perceptual_weight=1.0,
        disc_weight=1.0,
        num_channels=3,
        learn_logvar=False,
        use_actnorm=False,
        lora=False,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.perceptual_loss = LPIPS().eval()
        self.logvar = nn.Parameter(torch.zeros(size=()), requires_grad=learn_logvar)
        self.adversarial_loss = AdversarialLoss(
            disc_start=disc_start,
            disc_weight=disc_weight,
            num_channels=num_channels,
            use_actnorm=use_actnorm,
            lora=lora,
        )

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        optimizer_idx,
        global_step,
        last_layer,
        split="train",
    ):
        bsz = inputs.shape[0]

        if optimizer_idx == 0:
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
            p_loss = self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous()
            )
            rec_loss = rec_loss + self.perceptual_weight * p_loss
            nll_loss = (rec_loss / torch.exp(self.logvar) + self.logvar).sum() / bsz

            kl_loss = posteriors.kl().sum() / bsz

            g_term, g_loss, d_weight = self.adversarial_loss.generator_loss(
                reconstructions,
                nll_loss,
                last_layer=last_layer,
                global_step=global_step,
                training=self.training,
            )

            loss = nll_loss + self.kl_weight * kl_loss + g_term

            log = {
                f"{split}/total_loss": loss.clone().detach().mean(),
                f"{split}/logvar": self.logvar.detach(),
                f"{split}/kl_loss": kl_loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/d_weight": d_weight.detach(),
                f"{split}/g_loss": g_loss.detach().mean(),
            }
            return loss, log

        elif optimizer_idx == 1:
            d_loss, logits_real, logits_fake = self.adversarial_loss.discriminator_loss(
                inputs, reconstructions, global_step
            )

            log = {f"{split}/disc_loss": d_loss.clone().detach().mean()}
            if logits_real is not None:
                log[f"{split}/logits_real"] = logits_real.detach().mean()
                log[f"{split}/logits_fake"] = logits_fake.detach().mean()
            return d_loss, log
