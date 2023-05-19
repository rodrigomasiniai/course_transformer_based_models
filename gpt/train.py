# References:
    # https://paul-hyun.github.io/gpt-01/?fbclid=IwAR3jaAPdcWBIkShNDr-NIXE5JCfw-UvoQ2h000r5qnSBj8kjrY4ax1jDeM8
    # https://gaussian37.github.io/dl-pytorch-lr_scheduler/

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import timm
from timm.optim import AdamP
from timm.scheduler import CosineLRScheduler
import matplotlib.pyplot as plt

from process_images import _figure_to_array

training_epochs = 300
cooldown_epochs = 10
num_epochs = training_epochs + cooldown_epochs

model = torch.nn.Linear(2, 1)
optimizer = AdamP(model.parameters(), lr=0.01)


def plot_lrs_for_timm_scheduler(scheduler, batch_size, n_epochs):
    lrs = [optimizer.param_groups[0]["lr"]]

    n_steps = 0
    # for epoch in range(1, n_epochs + 1):
    for epoch in range(n_epochs):
        # n_steps = batch_size * epoch
        for _ in range(batch_size):
            n_steps += 1
            scheduler.step_update(num_updates=n_steps)
        scheduler.step(epoch=epoch + 1)

        lrs.append(optimizer.param_groups[0]["lr"])
    lrs = lrs[: -1]

    fig, axes = plt.subplots(figsize=(int(len(lrs) ** 0.4), 2))
    axes.plot(range(1, n_epochs + 1), lrs)
    # axes.plot(range(n_epochs), lrs)
    axes.set(xticks=range(0, n_epochs + 1, 10))
    # axes.set_xlim([1, n_epochs])
    axes.set_xlim([0, n_epochs])
    axes.tick_params(axis="x", labelrotation=90, labelsize=5)
    axes.grid(axis="x", color="black", alpha=1, linestyle="--", linewidth=0.5)
    arr = _figure_to_array(fig)
    return arr, lrs


n_epochs = 300
batch_size = 16
init_epoch = 50
scheduler = CosineLRScheduler(
    optimizer=optimizer,
    t_initial=init_epoch,
    lr_min=0.001, # Minimum learning rate
    cycle_mul=1.3, # A factor that increases T_{i} after a restart (Default `1``)
    cycle_decay=0.8,
    cycle_limit=n_epochs // init_epoch + 1,
    warmup_t=10,
    warmup_lr_init=0.002,
    t_in_epochs=True # Whether the number iterations is given in terms of epochs rather than the number of batch updates (Default `True`)
)
plot, lrs = plot_lrs_for_timm_scheduler(scheduler, n_epochs=n_epochs, batch_size=batch_size)
show_image(plot)



# We used the Adam optimization scheme with a max learning rate of 2.5e-4. The learning rate was increased linearly from zero over the first 2000 updates and annealed to 0 using a cosine schedule.
max_lr = 2.5e-4
batch_size = 64
n_epochs = 100
gpt = GPT()
optimizer = optim.Adam(params=gpt.parameters(), lr=0)
scheduler = CosineAnnealingWarmUpRestarts(optimizer=optimizer, T_up=2_000, eta_max=2.5e-4, T_0=150, T_mult=1, gamma=1)
