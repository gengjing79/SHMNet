from torch import nn
from module.MobilenetFeature import mobilenet_f
class Selector(nn.Module):
    def __init__(self):
        super(Selector, self).__init__()
        self.feature_extractor = mobilenet_f(reduced_tail=False)
        self.classifier = nn.Sequential(nn.Linear(960, 256),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(256, 2),
                                        nn.Softmax(dim=1)
                                        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)

    def get_features(self, x):
        x = self.feature_extractor(x)
        return x

    def classify(self, x):
        x = self.classifier(x)