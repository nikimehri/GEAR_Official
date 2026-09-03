from torch import nn
import torch
import timm
from torchvision.models import resnet50, ResNet50_Weights


class Identity(nn.Module):
    """No-op passthrough module (stands in for a disabled Dropout layer, or for
    CustomResNet's maxpool when using the CIFAR-friendly stem)."""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):
    """Reshapes a [B, ...] tensor down to [B, -1]."""

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv(nn.Sequential):
    """A Conv2d/ConvTranspose2d block with optional BatchNorm and activation.
    Auto-computes 'same'-style padding for odd kernel sizes if none is given."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)


class AllCNN(nn.Module):
    """A small all-convolutional network (no fully-connected layers except the
    final classifier), used for CIFAR-10/FashionMNIST/SVHN/MedMNIST."""

    def __init__(self, n_channels=3, num_classes=10, dropout=False, filters_percentage=1., size=32, batch_norm=True):
        super(AllCNN, self).__init__()
        n_filter1 = int(size*3 * filters_percentage)
        n_filter2 = int(size*6 * filters_percentage)
        self.features = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(inplace=False) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),  # 14
            nn.Dropout(inplace=False) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

    def get_embedding(self, x):
        """Returns the final (pre-classifier) feature vector, used for t-SNE
        plots and whole-model representation comparisons."""
        x = self.features(x)
        return x

    def forward_with_features(self, x, capture_layers=None):
        """Runs the forward pass one feature layer at a time so intermediate
        activations can be captured by integer index (self.features has 12
        children, 0-11; index 11 is the final flattened embedding). This is
        what lets gear.py compute contrastive/entanglement losses on features
        from any layer, not just the final embedding."""
        if capture_layers is None:
            capture_layers = [11]
        activations = {}

        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in capture_layers:
                activations[i] = x

        logits = self.classifier(x)
        return logits, activations


class CustomResNet(nn.Module):
    """A ResNet-50 backbone (pretrained on ImageNet) with a custom classifier
    head. Set cifar_stem=True when training on small (e.g. 32x32 CIFAR) images
    — the default ImageNet stem (7x7 stride-2 conv + maxpool) downsamples too
    aggressively for small inputs, so it's swapped for a 3x3 stride-1 conv with
    no maxpool."""

    def __init__(self, num_classes, cifar_stem=False):
        super(CustomResNet, self).__init__()
        self.resnet_base = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.resnet_base.fc.in_features
        self.resnet_base.fc = nn.Identity()

        if cifar_stem:
            self.resnet_base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_base.maxpool = nn.Identity()

        self._classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes),
        )

    @property
    def features(self):
        """Extracts features from the pretrained network up to the removed fc layer."""
        return nn.Sequential(*(list(self.resnet_base.children())[:-1]))

    @property
    def classifier(self):
        """Gets the classifier part of the network."""
        return self._classifier

    @classifier.setter
    def classifier(self, new_classifier):
        """Allows replacing the classifier head in place (e.g. ensemble.py
        grafts a differently-trained model's classifier onto this backbone).
        Without this setter, assigning to .classifier raises AttributeError,
        since a plain @property has no default setter."""
        self._classifier = new_classifier

    def forward(self, x):
        x = self.resnet_base(x)
        x = torch.flatten(x, 1)
        x = self._classifier(x)
        return x

    def get_embedding(self, x):
        """Returns the final (pre-classifier) feature vector."""
        x = self.resnet_base(x)
        x = torch.flatten(x, 1)
        return x

    def forward_with_features(self, x, capture_layers=None):
        """Runs the forward pass with forward hooks on the named ResNet stages
        (layer1..layer4) to capture their intermediate activations. Hooks are
        always removed afterward so repeated calls don't leak/accumulate."""
        if capture_layers is None:
            capture_layers = ["layer4"]
        activations = {}
        hooks = []
        named = dict(self.resnet_base.named_children())
        for layer_name in capture_layers:
            if layer_name not in named:
                raise ValueError(
                    f"Layer '{layer_name}' not in ResNet. Valid: {list(named.keys())}"
                )
            def make_hook(name):
                def hook(module, inp, out):
                    activations[name] = out
                return hook
            hooks.append(named[layer_name].register_forward_hook(make_hook(layer_name)))
        out = self.resnet_base(x)
        out = torch.flatten(out, 1)
        logits = self._classifier(out)
        for h in hooks:
            h.remove()
        return logits, activations


class ViT(nn.Module):
    """A Vision Transformer backbone (pretrained on ImageNet via timm) with a
    replaced classifier head. Input images must already be resized to
    img_size x img_size — that's the dataset transform pipeline's job, not
    this class's.

    The defaults (timm_model_name='vit_base_patch32_224', img_size=224,
    patch_size=32) are the checkpoint's own native configuration, so the
    pretrained patch embedding loads with no resizing/reinitialization —
    this is what CIFAR-100/TinyImageNet use. trainer.py's clinical pipeline
    instead overrides these to reuse a patch16 checkpoint at img_size=512, a
    genuine patch/resolution mismatch that timm resolves by reinitializing
    the patch-embedding layer."""

    def __init__(self, num_classes, timm_model_name='vit_base_patch32_224', img_size=224, patch_size=32):
        super(ViT, self).__init__()
        self.vit = timm.create_model(timm_model_name, pretrained=True, img_size=img_size, patch_size=patch_size)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

    def get_embedding(self, x):
        """Returns the final pooled representation before the classifier head."""
        features = self.vit.forward_features(x)
        return self.vit.forward_head(features, pre_logits=True)

    def forward_with_features(self, x, capture_layers=None):
        """Runs the forward pass with forward hooks on the named transformer
        blocks (self.vit.blocks[i]) to capture their intermediate
        activations, addressed by integer block index (e.g. capture_layers=[6]
        captures the 6th block's output). Only the CLS token (index 0 of the
        token sequence) is kept per captured block, not the full [B, N, D]
        token sequence — this keeps captured activations a plain [B, D]
        vector, matching what gear.py's prepare_features expects from
        AllCNN/CustomResNet's captured activations. Hooks are always removed
        afterward so repeated calls don't leak/accumulate."""
        num_blocks = len(self.vit.blocks)
        if capture_layers is None:
            capture_layers = [num_blocks - 1]
        activations = {}
        hooks = []
        for layer_idx in capture_layers:
            if not (0 <= layer_idx < num_blocks):
                raise ValueError(
                    f"Block index {layer_idx} out of range. Valid: 0-{num_blocks - 1}"
                )
            def make_hook(idx):
                def hook(module, inp, out):
                    activations[idx] = out[:, 0, :]
                return hook
            hooks.append(self.vit.blocks[layer_idx].register_forward_hook(make_hook(layer_idx)))
        logits = self.vit(x)
        for h in hooks:
            h.remove()
        return logits, activations
