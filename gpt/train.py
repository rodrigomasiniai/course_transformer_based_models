# References:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from timm.optim import AdamP
from timm.scheduler import CosineLRScheduler
import matplotlib.pyplot as plt

from process_images import _figure_to_array, show_image


# def plot_lrs_for_timm_scheduler(scheduler, batch_size, n_steps):
#     batch_size = 32
#     # n_epochs = 30
#     n_steps = 2000

#     n_epochs = n_steps // batch_size
#     968 // 32 * 2
#     n_epochs
#     lrs = [optimizer.param_groups[0]["lr"]]

#     for epoch in range(n_epochs):
#         n_steps = batch_size * epoch
#         for _ in range(batch_size):
#             n_steps += 1
#             # Should be called after each optimizer update with the index of the next update.
#             scheduler.step_update(num_updates=n_steps)

#             lrs.append(optimizer.param_groups[0]["lr"])
#         # Should be called at the end of each epoch, with the index of the next epoch
#         scheduler.step(epoch=epoch + 1)

#         # lrs.append(optimizer.param_groups[0]["lr"])
#     lrs = lrs[: -1]
#     return lrs


def get_lr_schedule(scheduler, n_steps):
    lrs = list()
    for step in range(1, n_steps + 1):
        lrs.append(optimizer.param_groups[0]["lr"])
        # Should be called after each optimizer update with the index of the next update.
        scheduler.step_update(num_updates=step + 1)
    return lrs


def visualize_lrs(lrs, n_steps):
    fig, axes = plt.subplots(figsize=(int(len(lrs) ** 0.2), 3))
    axes.plot(range(1, n_steps + 1), lrs)
    # axes.set(xticks=range(0, n_steps + 1, 50))
    axes.set_xlim([0, n_steps])
    axes.tick_params(axis="x", labelrotation=90, labelsize=5)
    axes.tick_params(axis="y", labelsize=5)
    axes.grid(axis="x", color="black", alpha=1, linestyle="--", linewidth=0.5)
    fig.tight_layout()

    arr = _figure_to_array(fig)
    return arr


if __name__ == "__main__":
    n_steps = 20000
    batch_size = 16
    data_size = 5000
    max_lr = 2.5e-4

    model = torch.nn.Linear(2, 1)
    # optimizer = AdamP(model.parameters(), lr=max_lr)
    optimizer = optim.Adam(params=model.parameters(), lr=max_lr)

    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=n_steps,
        # lr_min=0.001, # Minimum learning rate
        lr_min=0, # Minimum learning rate
        # cycle_mul=1.3,
        # cycle_decay=0.8,
        # cycle_limit=n_epochs // init_epoch + 1,
        warmup_t=2000,
        warmup_lr_init=0,
        warmup_prefix=True,
        t_in_epochs=False # If `True` the number of iterations is given in terms of epochs
            # rather than the number of batch updates.
    )
    lrs = get_lr_schedule(scheduler, n_steps=n_steps)
    vis = visualize_lrs(lrs=lrs, n_steps=n_steps)
    show_image(vis)


    # We used the Adam optimization scheme with a max learning rate of 2.5e-4. The learning rate was increased linearly from zero over the first 2000 updates and annealed to 0 using a cosine schedule.
    batch_size = 64
    n_epochs = 100
