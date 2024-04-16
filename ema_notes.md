
### Exponential Moving Average(EMA)
Implementation of EMA with pytorch lightning.


[Pytorch >= 2.1.0](https://pytorch.org/docs/stable/optim.html#weight-averaging-swa-and-ema)
```
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
```

Module Initialization:
``` 
if config["use_ema"]:
    ema_decay = config["ema_decay"]
    self.ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
    print('Using EMA model with decay', ema_decay)
```

Inference/ Validation
```
    if self.config["use_ema"]:
        output = self.ema_model(specs)
    else:
        output = self.model(specs)
```

Update ema weights before optimization step:
```
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.config["use_ema"]:
            self.ema_model.update_parameters(self.model)
```