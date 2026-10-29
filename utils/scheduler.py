import torch.optim.lr_scheduler as lr_scheduler


def define_scheduler(args, optimizer):
    if args.scheduler == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, int(round(args.num_epoch * 0.9))),
            gamma=0.1,
        )
    elif args.scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif args.scheduler == 'cosine':
        warmup_epochs = int(getattr(args, "optimizer_warmup_epochs", 0))
        if warmup_epochs <= 0:
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.num_epoch,
                eta_min=0,
            )
        else:
            cosine_epochs = max(int(args.num_epoch) - warmup_epochs, 1)
            scheduler = lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=0.1,
                        end_factor=1.0,
                        total_iters=warmup_epochs,
                    ),
                    lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=cosine_epochs,
                        eta_min=0,
                    ),
                ],
                milestones=[warmup_epochs],
            )
    else:
        raise NotImplementedError('Scheduler [{}] is not implemented'.format(args.scheduler))
    return scheduler
