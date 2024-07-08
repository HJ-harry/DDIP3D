"""
LoRA. Adapted from https://github.com/cloneofsimo/lora/tree/master/lora_diffusion
"""
import torch 
import torch.nn as nn 
from typing import List, Optional, Set, Type, Any, Dict
import itertools

UNET_EXTENDED_TARGET_REPLACE = {"AttentionBlock", "ResBlock"}
UNET_TARGET_REPLACE = {"AttentionBlock"}

class LoraInjectedLinear(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, r=4, dropout_p=0.1, scale=1.0
    ):
        super().__init__()

        if r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}"
            )
        self.r = r
        self.linear = nn.Linear(in_features, out_features, bias)
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = scale
        self.selector = nn.Identity()

        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        if self.scale == 0:
            return (self.linear(input))
        else:
            return (self.linear(input) 
                    + self.dropout(self.lora_up(self.selector(self.lora_down(input))))
                    * self.scale)
         
    def realize_as_lora(self):
        return self.lora_up.weight.data * self.scale, self.lora_down.weight.data

    def set_selector_from_diag(self, diag: torch.Tensor):
        # diag is a 1D tensor of size (r,)
        assert diag.shape == (self.r,)
        self.selector = nn.Linear(self.r, self.r, bias=False)
        self.selector.weight.data = torch.diag(diag)
        self.selector.weight.data = self.selector.weight.data.to(
            self.lora_up.weight.device
        ).to(self.lora_up.weight.dtype)


class LoraInjectedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        r: int = 4,
        dropout_p: float = 0.1,
        scale: float = 1.0,
    ):
        super().__init__()
        if r > min(in_channels, out_channels):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_channels, out_channels)}"
            )
        self.r = r
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.lora_down = nn.Conv2d(
            in_channels=in_channels,
            out_channels=r,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Conv2d(
            in_channels=r,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.selector = nn.Identity()
        self.scale = scale

        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        if self.scale == 0: 
            return (self.conv(input))
        else: 
            return (self.conv(input)
                + self.dropout(self.lora_up(self.selector(self.lora_down(input))))
                * self.scale)

    def realize_as_lora(self):
        return self.lora_up.weight.data * self.scale, self.lora_down.weight.data

    def set_selector_from_diag(self, diag: torch.Tensor):
        # diag is a 1D tensor of size (r,)
        assert diag.shape == (self.r,)
        self.selector = nn.Conv2d(
            in_channels=self.r,
            out_channels=self.r,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.selector.weight.data = torch.diag(diag)

        # same device + dtype as lora_up
        self.selector.weight.data = self.selector.weight.data.to(
            self.lora_up.weight.device
        ).to(self.lora_up.weight.dtype)


class LoraInjectedConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        r: int = 4,
        dropout_p: float = 0.1,
        scale: float = 1.0,
    ):
        super().__init__()
        if r > min(in_channels, out_channels):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_channels, out_channels)}"
            )
        self.r = r
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.lora_down = nn.Conv1d(
            in_channels=in_channels,
            out_channels=r,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Conv1d(
            in_channels=r,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.selector = nn.Identity()
        self.scale = scale

        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        if self.scale == 0.:
            return (self.conv(input))
        else:    
            return (self.conv(input)
                + self.dropout(self.lora_up(self.selector(self.lora_down(input))))
                * self.scale)
            

    def realize_as_lora(self):
        return self.lora_up.weight.data * self.scale, self.lora_down.weight.data

    def set_selector_from_diag(self, diag: torch.Tensor):
        # diag is a 1D tensor of size (r,)
        assert diag.shape == (self.r,)
        self.selector = nn.Conv1d(
            in_channels=self.r,
            out_channels=self.r,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.selector.weight.data = torch.diag(diag)

        # same device + dtype as lora_up
        self.selector.weight.data = self.selector.weight.data.to(
            self.lora_up.weight.device
        ).to(self.lora_up.weight.dtype)

def _find_modules(
    model,
    ancestor_class: Optional[Set[str]] = None,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] = [
        LoraInjectedLinear,
        LoraInjectedConv1d,
        LoraInjectedConv2d,
    ],
):
    """
    Find all modules of a certain class (or union of classes) that are direct or
    indirect descendants of other modules of a certain class (or union of classes).

    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    """

    # Get the targets we should replace all linears under
    if ancestor_class is not None:
        ancestors = (
            module
            for module in model.modules()
            if module.__class__.__name__ in ancestor_class
        )
    else:
        # this, incase you want to naively iterate over all modules.
        ancestors = [module for module in model.modules()]

    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.split(".")
                parent = ancestor
                while path:
                    parent = parent.get_submodule(path.pop(0))
                # Skip this linear if it's a child of a LoraInjectedLinear
                if exclude_children_of and any(
                    [isinstance(parent, _class) for _class in exclude_children_of]
                ):
                    continue
                # Otherwise, yield it
                yield parent, name, module

def _include_blocks_in_model(
    model: nn.Module, 
    include_blocks: Set[str] = None,
):
    return nn.ModuleList(
        [child for name, child in model.named_children() if name in include_blocks])

_find_modules = _find_modules

def inject_trainable_lora_extended(
    model: nn.Module,
    target_replace_module: Set[str] = UNET_EXTENDED_TARGET_REPLACE,
    include_blocks: Set[str] = ['input_blocks', 'middle_block', 'output_blocks', 'out'],
    r: int = 4
):
    """
    inject lora into model, and returns lora parameter groups.
    """
    print(f"LoRA injection to all modules")
    require_grad_params = []
    names = []

    for _module, name, _child_module in _find_modules(
        _include_blocks_in_model(model, include_blocks), target_replace_module, search_class=[nn.Conv1d, nn.Conv2d, nn.Linear]
    ):

        if _child_module.__class__ == nn.Linear:
            weight = _child_module.weight
            bias = _child_module.bias
            _tmp = LoraInjectedLinear(
                _child_module.in_features,
                _child_module.out_features,
                _child_module.bias is not None,
                r=r,
            )
            _tmp.linear.weight = weight
            if bias is not None:
                _tmp.linear.bias = bias
        elif _child_module.__class__ == nn.Conv2d:
            weight = _child_module.weight
            bias = _child_module.bias
            _tmp = LoraInjectedConv2d(
                _child_module.in_channels,
                _child_module.out_channels,
                _child_module.kernel_size,
                _child_module.stride,
                _child_module.padding,
                _child_module.dilation,
                _child_module.groups,
                _child_module.bias is not None,
                r=r,
            )

            _tmp.conv.weight = weight
            if bias is not None:
                _tmp.conv.bias = bias
        elif _child_module.__class__ == nn.Conv1d:
            weight = _child_module.weight
            bias = _child_module.bias
            _tmp = LoraInjectedConv1d(
                _child_module.in_channels,
                _child_module.out_channels,
                _child_module.kernel_size,
                _child_module.stride,
                _child_module.padding,
                _child_module.dilation,
                _child_module.groups,
                _child_module.bias is not None,
                r=r,
            )

            _tmp.conv.weight = weight
            if bias is not None:
                _tmp.conv.bias = bias

        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        if bias is not None:
            _tmp.to(_child_module.bias.device).to(_child_module.bias.dtype)

        _module._modules[name] = _tmp

        require_grad_params.append(_module._modules[name].lora_up.parameters())
        require_grad_params.append(_module._modules[name].lora_down.parameters())

        _module._modules[name].lora_up.weight.requires_grad = True
        _module._modules[name].lora_down.weight.requires_grad = True
        names.append(name)

    return require_grad_params, names


def inject_trainable_lora(
    model: nn.Module,
    target_replace_module: Set[str] = UNET_TARGET_REPLACE,
    include_blocks: Set[str] = ['middle_block'],
    r: int = 4
):
    """
    inject lora into model, and returns lora parameter groups.
    """
    print(f"LoRA injection to attention modules only")
    require_grad_params = []
    names = []

    for _module, name, _child_module in _find_modules(
        _include_blocks_in_model(model, include_blocks), target_replace_module, search_class=[nn.Conv1d, nn.Conv2d, nn.Linear]
    ):

        if _child_module.__class__ == nn.Linear:
            weight = _child_module.weight
            bias = _child_module.bias
            _tmp = LoraInjectedLinear(
                _child_module.in_features,
                _child_module.out_features,
                _child_module.bias is not None,
                r=r,
            )
            _tmp.linear.weight = weight
            if bias is not None:
                _tmp.linear.bias = bias
        elif _child_module.__class__ == nn.Conv2d:
            weight = _child_module.weight
            bias = _child_module.bias
            _tmp = LoraInjectedConv2d(
                _child_module.in_channels,
                _child_module.out_channels,
                _child_module.kernel_size,
                _child_module.stride,
                _child_module.padding,
                _child_module.dilation,
                _child_module.groups,
                _child_module.bias is not None,
                r=r,
            )

            _tmp.conv.weight = weight
            if bias is not None:
                _tmp.conv.bias = bias
        elif _child_module.__class__ == nn.Conv1d:
            weight = _child_module.weight
            bias = _child_module.bias
            _tmp = LoraInjectedConv1d(
                _child_module.in_channels,
                _child_module.out_channels,
                _child_module.kernel_size,
                _child_module.stride,
                _child_module.padding,
                _child_module.dilation,
                _child_module.groups,
                _child_module.bias is not None,
                r=r,
            )

            _tmp.conv.weight = weight
            if bias is not None:
                _tmp.conv.bias = bias

        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        if bias is not None:
            _tmp.to(_child_module.bias.device).to(_child_module.bias.dtype)

        _module._modules[name] = _tmp

        require_grad_params.append(_module._modules[name].lora_up.parameters())
        require_grad_params.append(_module._modules[name].lora_down.parameters())

        _module._modules[name].lora_up.weight.requires_grad = True
        _module._modules[name].lora_down.weight.requires_grad = True
        names.append(name)

    return require_grad_params, names


def adapt_model(
    score: nn.Module, 
    impl: str = "lora",
    adapt_kwargs: Optional[Dict] = None,
    verbose: bool = True
    ) -> nn.Module:
    
    for name, param in score.named_parameters():
        param.requires_grad = False
    if impl == 'full':
        for name, param in score.named_parameters():
            param.requires_grad = True
    elif impl == 'decoder': 
        for name, param in score.out.named_parameters():
            if not "emb_layers" in name:
                param.requires_grad = True
        for name, param in score.output_blocks.named_parameters():
            if not "emb_layers" in name:
                param.requires_grad = True
    elif impl == 'lora':
        score.requires_grad_(False)
        for name, param in score.named_parameters():
            if "bias" in name and not "emb_layers" in name:
                param.requires_grad = True
        require_grad_params, names = inject_trainable_lora_extended(score, **adapt_kwargs)

    if verbose:
        num_params = sum([p.numel() for p in score.parameters()])
        trainable_params = sum([p.numel() for p in score.parameters() if p.requires_grad])
        print(f'% of trainable params: {trainable_params/num_params*100}')
        tpm = trainable_params / 1_000_000
        print(f"# of injected lora params: {tpm:.03f}M")
        
    return require_grad_params, names