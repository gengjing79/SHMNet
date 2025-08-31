import torch
from torch import nn
from module.WeightGenerator import EMAWeightGenerator
from torchvision.models.swin_transformer import swin_s
from torchvision.models.convnext import convnext_small
from torchvision.models.mobilenet import mobilenet_v3_large
from module.selector import Selector
#-------------------------------------------------------------------
class Net1_1(nn.Module):
    def __init__(self, num_classes, hidden_dim=960):
        super().__init__()
        self.num_classes = num_classes
        self.branch1 = swin_s(num_classes=num_classes)
        self.branch2 = mobilenet_v3_large(num_classes=num_classes)
        self.selector = Selector()
        self.mix_weight = EMAWeightGenerator(feature_dim=hidden_dim + 2)

    def forward(self, x, labels=None):
        handcraft_f = self.selector.get_features(x)

        logits_branch1 = self.branch1(x)
        logits_branch2 = self.branch2(x)

        conf1 = torch.max(torch.softmax(logits_branch1, dim=1), dim=1)[0].unsqueeze(1)
        conf2 = torch.max(torch.softmax(logits_branch2, dim=1), dim=1)[0].unsqueeze(1)
        mix_weight = self.mix_weight(torch.cat([handcraft_f, conf1, conf2], dim=1))

        final_logits = mix_weight * logits_branch1 + (1 - mix_weight) * logits_branch2

        if labels is not None:
            return final_logits
        else:
            return final_logits

def SHHNet1_1(num_classes=10):
    model = Net1_1(num_classes=num_classes)
    return model
#-------------------------------------------------------------------
class Net1_2(nn.Module):
    def __init__(self, num_classes, hidden_dim=960):
        super().__init__()
        self.num_classes = num_classes
        self.branch1 = swin_s(num_classes=num_classes)
        self.branch2 = convnext_small(num_classes=num_classes)
        self.selector = Selector()
        self.mix_weight = EMAWeightGenerator(feature_dim=hidden_dim + 2)

    def forward(self, x, labels=None):
        handcraft_f = self.selector.get_features(x)

        logits_branch1 = self.branch1(x)
        logits_branch2 = self.branch2(x)

        conf1 = torch.max(torch.softmax(logits_branch1, dim=1), dim=1)[0].unsqueeze(1)
        conf2 = torch.max(torch.softmax(logits_branch2, dim=1), dim=1)[0].unsqueeze(1)
        mix_weight = self.mix_weight(torch.cat([handcraft_f, conf1, conf2], dim=1))

        final_logits = mix_weight * logits_branch1 + (1 - mix_weight) * logits_branch2

        if labels is not None:
            return final_logits
        else:
            return final_logits

def SHHNet1_2(num_classes=10):
    model = Net1_2(num_classes=num_classes)
    return model
#-------------------------------------------------------------------
class Net1_3(nn.Module):
    def __init__(self, num_classes, hidden_dim=960):
        super().__init__()
        self.branch1 = swin_s(num_classes=num_classes)
        self.branch2 = mobilenet_v3_large(num_classes=num_classes)
        self.branch3 = convnext_small(num_classes=num_classes)
        self.selector = Selector()
        self.mix_weight = EMAWeightGenerator(feature_dim=hidden_dim + 3, num_experts=3)
    def forward(self, x):
        handcraft_f = self.selector.get_features(x)
        logits1 = self.branch1(x)
        probs1 = torch.exp(logits1)
        conf1 = torch.max(probs1, dim=1)[0].unsqueeze(1)

        logits2 = self.branch2(x)
        probs2 = torch.exp(logits2)
        conf2 = torch.max(probs2, dim=1)[0].unsqueeze(1)

        logits3 = self.branch3(x)
        probs3 = torch.exp(logits3)
        conf3 = torch.max(probs3, dim=1)[0].unsqueeze(1)
        weights = self.mix_weight(torch.cat([handcraft_f, conf1, conf2, conf3], dim=1))
        fused_probs = weights[:, 0:1] * probs1 + weights[:, 1:2] * probs2 + weights[:, 2:3] * probs3
        return torch.log(fused_probs + 1e-9)

def SHHNet1_3(num_classes=10):
    model = Net1_3(num_classes=num_classes)
    return model
#-------------------------------------------------------------------
class Net2_1(nn.Module):
    def __init__(self, num_classes, hidden_dim=960):
        super().__init__()
        self.num_classes = num_classes
        self.branch1 = swin_s(num_classes=num_classes)
        self.branch2 = mobilenet_v3_large(num_classes=num_classes)
        self.selector = Selector()
        self.mix_weight = EMAWeightGenerator(feature_dim=hidden_dim + 2)

    def forward(self, x, labels=None):
        batch_size = x.size(0)
        device = x.device
        selection_probs = self.selector(x)
        handcraft_f = self.selector.get_features(x)
        final_logits = torch.zeros(batch_size, self.num_classes).to(device)

        is_simple = torch.argmax(selection_probs, dim=1) == 0
        simple_indices = torch.where(is_simple)[0]
        if len(simple_indices) > 0:
            x_simple = x[simple_indices]
            logits_branch2 = self.branch2(x_simple)
            probs2 = torch.softmax(logits_branch2, dim=1)
            conf2 = torch.max(probs2, dim=1)[0]

            high_conf_indices = torch.where(conf2 >= 0.90)[0]
            if len(high_conf_indices) > 0:
                final_logits[simple_indices[high_conf_indices]] = logits_branch2[high_conf_indices]

            low_conf_indices = torch.where(conf2 < 0.90)[0]
            if len(low_conf_indices) > 0:
                logits_branch1 = self.branch1(x_simple[low_conf_indices])
                final_logits[simple_indices[low_conf_indices]] = logits_branch1

        is_complex = torch.argmax(selection_probs, dim=1) == 1
        complex_indices = torch.where(is_complex)[0]
        if len(complex_indices) > 0:
            x_complex = x[complex_indices]
            handcraft_f_complex = handcraft_f[complex_indices]
            logits_branch1 = self.branch1(x_complex)
            logits_branch2 = self.branch2(x_complex)

            conf1 = torch.max(torch.softmax(logits_branch1, dim=1), dim=1)[0].unsqueeze(1)
            conf2 = torch.max(torch.softmax(logits_branch2, dim=1), dim=1)[0].unsqueeze(1)
            weights = self.mix_weight(torch.cat([handcraft_f_complex, conf1, conf2], dim=1))
            fused_logits = weights * logits_branch1 + (1 - weights) * logits_branch2
            final_logits[complex_indices] = fused_logits

        if labels is not None:

            balance_loss = self.entropy_balance_loss(selection_probs)
            incentive_loss = self.incentive_loss(selection_probs, final_logits, labels)
            auxiliary_losses = {
                "balance_loss": balance_loss,
                "incentive_loss": incentive_loss
            }
            return final_logits, selection_probs, auxiliary_losses
        else:
            return final_logits

    def entropy_balance_loss(self, selection_probs):
        target_prob = 0.5
        prob_simple = selection_probs[:, 0].mean()
        scale_factor = 1

        return scale_factor * torch.abs(prob_simple - target_prob)

    def incentive_loss(self, selection_probs, logits, labels):

        _, preds = torch.max(logits, dim=1)
        correct = (preds == labels).float()

        multi_indices = torch.where(selection_probs[:, 1] > 0.5)[0]

        if len(multi_indices) == 0:
            return torch.tensor(0.0, device=logits.device)

        accuracy_multi = correct[multi_indices].mean()
        accuracy_all = correct.mean()

        reward = accuracy_multi - accuracy_all

        scale_factor = 1

        log_p_multi = torch.log(selection_probs[:, 1] + 1e-10)
        loss = -reward * scale_factor * log_p_multi[multi_indices].mean()

        return loss

def SHHNet2_1(num_classes=10):
    model = Net2_1(num_classes=num_classes)
    return model
#-------------------------------------------------------
class Net2_2(nn.Module):
    def __init__(self, num_classes, hidden_dim=960):
        super().__init__()
        self.num_classes = num_classes
        self.branch1 = swin_s(num_classes=num_classes)
        self.branch2 = convnext_small(num_classes=num_classes)
        self.selector = Selector()
        self.mix_weight = EMAWeightGenerator(feature_dim=hidden_dim + 2)

    def forward(self, x, labels=None):
        batch_size = x.size(0)
        device = x.device
        selection_probs = self.selector(x)
        handcraft_f = self.selector.get_features(x)
        final_logits = torch.zeros(batch_size, self.num_classes).to(device)

        is_simple = torch.argmax(selection_probs, dim=1) == 0
        simple_indices = torch.where(is_simple)[0]
        if len(simple_indices) > 0:
            x_simple = x[simple_indices]
            logits_branch1 = self.branch1(x_simple)
            probs2 = torch.softmax(logits_branch1, dim=1)
            conf2 = torch.max(probs2, dim=1)[0]

            high_conf_indices = torch.where(conf2 >= 0.90)[0]
            if len(high_conf_indices) > 0:
                final_logits[simple_indices[high_conf_indices]] = logits_branch1[high_conf_indices]

            low_conf_indices = torch.where(conf2 < 0.90)[0]
            if len(low_conf_indices) > 0:
                logits_branch2 = self.branch2(x_simple[low_conf_indices])
                final_logits[simple_indices[low_conf_indices]] = logits_branch2

        is_complex = torch.argmax(selection_probs, dim=1) == 1
        complex_indices = torch.where(is_complex)[0]
        if len(complex_indices) > 0:
            x_complex = x[complex_indices]
            handcraft_f_complex = handcraft_f[complex_indices]
            logits_branch1 = self.branch1(x_complex)
            logits_branch2 = self.branch2(x_complex)

            conf1 = torch.max(torch.softmax(logits_branch1, dim=1), dim=1)[0].unsqueeze(1)
            conf2 = torch.max(torch.softmax(logits_branch2, dim=1), dim=1)[0].unsqueeze(1)
            weights = self.mix_weight(torch.cat([handcraft_f_complex, conf1, conf2], dim=1))
            fused_logits = weights * logits_branch1 + (1 - weights) * logits_branch2
            final_logits[complex_indices] = fused_logits

        if labels is not None:

            balance_loss = self.entropy_balance_loss(selection_probs)
            incentive_loss = self.incentive_loss(selection_probs, final_logits, labels)
            auxiliary_losses = {
                "balance_loss": balance_loss,
                "incentive_loss": incentive_loss
            }
            return final_logits, selection_probs, auxiliary_losses
        else:
            return final_logits

    def entropy_balance_loss(self, selection_probs):
        target_prob = 0.5
        prob_simple = selection_probs[:, 0].mean()
        scale_factor = 0.5
        return scale_factor * torch.abs(prob_simple - target_prob)

    def incentive_loss(self, selection_probs, logits, labels):
        _, preds = torch.max(logits, dim=1)
        correct = (preds == labels).float()
        multi_indices = torch.where(selection_probs[:, 1] > 0.5)[0]
        if len(multi_indices) == 0:
            return torch.tensor(0.0, device=logits.device)
        accuracy_multi = correct[multi_indices].mean()
        accuracy_all = correct.mean()
        reward = accuracy_multi - accuracy_all
        scale_factor = 0.5
        log_p_multi = torch.log(selection_probs[:, 1] + 1e-10)
        loss = -reward * scale_factor * log_p_multi[multi_indices].mean()
        return loss

def SHHNet2_2(num_classes=10):
    model = Net2_2(num_classes=num_classes)
    return model
#------------------------------------------------------------------------
class Net2_3(nn.Module):
    def __init__(self, num_classes, hidden_dim=960):
        super().__init__()
        self.num_classes = num_classes
        self.branch1 = swin_s(num_classes=num_classes)
        self.branch2 = mobilenet_v3_large(num_classes=num_classes)
        self.branch3 = convnext_small(num_classes=num_classes)
        self.selector = Selector()
        self.mix_weight = EMAWeightGenerator(feature_dim=hidden_dim + 3,num_experts=3)

    def forward(self, x, labels=None):
        batch_size = x.size(0)
        device = x.device
        selection_probs = self.selector(x)
        handcraft_f = self.selector.get_features(x)
        final_logits = torch.zeros(batch_size, self.num_classes).to(device)

        is_simple = torch.argmax(selection_probs, dim=1) == 0
        simple_indices = torch.where(is_simple)[0]
        if len(simple_indices) > 0:
            x_simple = x[simple_indices]
            logits_branch2 = self.branch2(x_simple)
            probs2 = torch.softmax(logits_branch2, dim=1)
            conf2 = torch.max(probs2, dim=1)[0]

            high_conf_indices = torch.where(conf2 >= 0.90)[0]
            if len(high_conf_indices) > 0:
                final_logits[simple_indices[high_conf_indices]] = logits_branch2[high_conf_indices]

            low_conf_indices = torch.where(conf2 < 0.90)[0]
            if len(low_conf_indices) > 0:
                logits_branch1 = self.branch2(x_simple[low_conf_indices])
                final_logits[simple_indices[low_conf_indices]] = logits_branch1

        is_complex = torch.argmax(selection_probs, dim=1) == 1
        complex_indices = torch.where(is_complex)[0]
        if len(complex_indices) > 0:
            x_complex = x[complex_indices]
            handcraft_f_complex = handcraft_f[complex_indices]
            logits_branch1 = self.branch1(x_complex)
            probs1 = torch.exp(logits_branch1)
            logits_branch2 = self.branch2(x_complex)
            probs2 = torch.exp(logits_branch2)
            logits_branch3 = self.branch3(x_complex)
            probs3 = torch.exp(logits_branch3)


            conf1 = torch.max(torch.softmax(logits_branch1, dim=1), dim=1)[0].unsqueeze(1)
            conf2 = torch.max(torch.softmax(logits_branch2, dim=1), dim=1)[0].unsqueeze(1)
            conf3 = torch.max(torch.softmax(logits_branch3, dim=1), dim=1)[0].unsqueeze(1)
            weights = self.mix_weight(torch.cat([handcraft_f_complex, conf1, conf2, conf3], dim=1))
            fused_logits = weights[:, 0:1] * probs1 + weights[:, 1:2] * probs2 + weights[:, 2:3] * probs3
            final_logits[complex_indices] = fused_logits

        if labels is not None:

            balance_loss = self.entropy_balance_loss(selection_probs)
            incentive_loss = self.incentive_loss(selection_probs, final_logits, labels)
            auxiliary_losses = {
                "balance_loss": balance_loss,
                "incentive_loss": incentive_loss
            }
            return final_logits, selection_probs, auxiliary_losses
        else:
            return final_logits

    def entropy_balance_loss(self, selection_probs):
        target_prob = 0.5
        prob_simple = selection_probs[:, 0].mean()
        scale_factor = 0.5
        return scale_factor * torch.abs(prob_simple - target_prob)

    def incentive_loss(self, selection_probs, logits, labels):
        _, preds = torch.max(logits, dim=1)
        correct = (preds == labels).float()
        multi_indices = torch.where(selection_probs[:, 1] > 0.5)[0]
        if len(multi_indices) == 0:
            return torch.tensor(0.0, device=logits.device)
        accuracy_multi = correct[multi_indices].mean()
        accuracy_all = correct.mean()
        reward = accuracy_multi - accuracy_all
        scale_factor = 0.5
        log_p_multi = torch.log(selection_probs[:, 1] + 1e-10)
        loss = -reward * scale_factor * log_p_multi[multi_indices].mean()
        return loss
def SHHNet2_3(num_classes=10):
    model = Net2_3(num_classes=num_classes)
    return model