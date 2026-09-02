"""
classifier.py -- the downstream classifier.

Aligned with the official MedMNIST benchmark implementation
(MedMNIST/experiments, MedMNIST2D/train_and_eval_pytorch.py and models.py),
not merely with the hyperparameters quoted in the paper.  The differences that
a naive torchvision setup gets wrong:

  * TWO ResNet-18 variants.  The official line is

        model = resnet18(pretrained=False, num_classes=n_classes) if resize \\
                else ResNet18(in_channels=n_channels, num_classes=n_classes)

    torchvision's ImageNet ResNet-18 (7x7 stride-2 conv + stride-2 maxpool) is
    used only when images are resized to 224.  At native MNIST-like resolution
    they use a CIFAR-style ResNet-18: 3x3 stride-1 first conv and NO maxpool,
    adapted from kuangliu/pytorch-cifar.  Feeding 28x28 into the ImageNet stem
    leaves 7x7 before the first residual block and 1x1 before the classifier.

  * Model selection is by best validation AUC (`cur_auc = val_metrics[1]`),
    not accuracy.

  * Milestones are proportional: [0.5 * num_epochs, 0.75 * num_epochs].

  * Normalisation is Normalize(mean=[.5], std=[.5]), i.e. (x - 0.5) / 0.5,
    with no augmentation.  Applied in pipeline.FilteredDataset.

Everything else -- Adam at 1e-3, batch 128, 100 epochs, gamma 0.1,
cross-entropy -- matches.

`arch="resnet18"` picks the variant automatically from the input resolution:
the CIFAR-style network below 64 pixels, torchvision at 64 and above.  Force
either with "resnet18_cifar" / "resnet18_imagenet".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


# ---------------------------------------------------------------------------
# Official MedMNIST ResNet (models.py, adapted from kuangliu/pytorch-cifar)
# ---------------------------------------------------------------------------

def _build_cifar_resnet(depth: int, in_channels: int, num_classes: int):
    import torch.nn as nn
    import torch.nn.functional as F

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes))

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + self.shortcut(x)
            return F.relu(out)

    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, in_planes, planes, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                                   kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.expansion * planes)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes))

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out = out + self.shortcut(x)
            return F.relu(out)

    class ResNet(nn.Module):
        def __init__(self, block, num_blocks, in_channels=1, num_classes=2):
            super().__init__()
            self.in_planes = 64
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(512 * block.expansion, num_classes)

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for s in strides:
                layers.append(block(self.in_planes, planes, s))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            return self.linear(out)

    spec = {18: (BasicBlock, [2, 2, 2, 2]), 50: (Bottleneck, [3, 4, 6, 3])}[depth]
    return ResNet(spec[0], spec[1], in_channels=in_channels, num_classes=num_classes)


def build_model(arch: str, n_classes: int, in_channels: int = 3,
                input_size: int = 224):
    """Pick the ResNet variant the official code would use at this resolution."""
    import torchvision.models as tvm

    depth = 50 if "50" in arch else 18
    if arch in ("resnet18", "resnet50"):
        arch = f"{arch}_{'imagenet' if input_size >= 64 else 'cifar'}"

    if arch.endswith("_cifar"):
        return _build_cifar_resnet(depth, in_channels, n_classes)

    factory = {18: tvm.resnet18, 50: tvm.resnet50}[depth]
    model = factory(weights=None, num_classes=n_classes)
    if in_channels != 3:                       # torchvision stem expects RGB
        import torch.nn as nn
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                                padding=3, bias=False)
    return model


def describe_model(arch: str, input_size: int) -> str:
    if arch in ("resnet18", "resnet50"):
        v = ("imagenet stem (torchvision)" if input_size >= 64
             else "cifar-style stem (MedMNIST models.py)")
        return f"{arch}, {v}, {input_size}px input"
    return f"{arch}, {input_size}px input"


# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    milestones: tuple | None = None      # None -> [0.5*epochs, 0.75*epochs]
    gamma: float = 0.1
    arch: str = "resnet18"
    in_channels: int = 3                 # paper: grayscale converted to RGB
    input_size: int = 224
    seed: int = 0
    num_workers: int = 4
    device: str = "auto"
    amp: bool = True
    select_by: str = "auc"               # official: best validation AUC

    def resolved_milestones(self):
        if self.milestones is not None:
            return list(self.milestones)
        return [int(0.5 * self.epochs), int(0.75 * self.epochs)]

    def as_dict(self) -> dict:
        return asdict(self)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _auc(logits, y, n_classes):
    from sklearn.metrics import roc_auc_score
    p = _softmax(logits)
    try:
        if n_classes == 2:
            return float(roc_auc_score(y, p[:, 1]))
        return float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
    except ValueError:                    # a class missing from this split
        return float("nan")


def train_and_predict(train_ds, val_ds, test_ds, n_classes: int,
                      cfg: TrainConfig, log_every: int = 10, verbose: bool = True):
    """Train one classifier and return predictions plus timings.

    `train_ds` / `val_ds` / `test_ds` are `pipeline.FilteredDataset` objects,
    which apply noise and the row filter per item.  Nothing is materialised.
    """
    import time
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if cfg.device in ("auto", "cuda") and torch.cuda.is_available():
        device = torch.device("cuda")
    elif cfg.device in ("auto", "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    pin = device.type == "cuda"
    dl = lambda ds, sh: DataLoader(ds, batch_size=cfg.batch_size, shuffle=sh,
                                   num_workers=cfg.num_workers, pin_memory=pin,
                                   persistent_workers=cfg.num_workers > 0,
                                   drop_last=False)
    tr, va, te = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    model = build_model(cfg.arch, n_classes, cfg.in_channels, cfg.input_size).to(device)
    if verbose:
        print(f"    model: {describe_model(cfg.arch, cfg.input_size)}, "
              f"in_channels={cfg.in_channels}", flush=True)

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=cfg.resolved_milestones(), gamma=cfg.gamma)
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def evaluate(loader):
        model.eval()
        logits, ys = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(xb)
                logits.append(out.float().cpu().numpy())
                ys.append(np.asarray(yb))
        return np.concatenate(logits), np.concatenate(ys)

    best_score, best_state, best_epoch = -np.inf, None, -1
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        vlogits, vy = evaluate(va)
        vacc = float((vlogits.argmax(1) == vy).mean())
        vauc = _auc(vlogits, vy, n_classes)
        score = vauc if (cfg.select_by == "auc" and np.isfinite(vauc)) else vacc
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if verbose and (epoch + 1) % log_every == 0:
            print(f"    epoch {epoch+1:3d}/{cfg.epochs}  val auc {vauc:.4f} "
                  f"acc {vacc:.4f}  best {cfg.select_by} {best_score:.4f} "
                  f"(epoch {best_epoch+1})", flush=True)
    train_time = time.perf_counter() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
    t1 = time.perf_counter()
    tlogits, ty = evaluate(te)
    infer_time = time.perf_counter() - t1

    return {
        "test_logits": tlogits,
        "test_pred": tlogits.argmax(1),
        "test_true": ty,
        "val_score": best_score,
        "val_acc": best_score,          # kept for backward compatibility
        "best_epoch": best_epoch,
        "select_by": cfg.select_by,
        "train_time_s": train_time,
        "infer_time_s": infer_time,
        "config": cfg.as_dict(),
    }
