import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_fasterRCNN(pretrained=False,num_classes=2):
    # load a model i pretrained is True uses the COCO dataset
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)

    # replace the classifier with a new one, that has num_classes which is user-defined
    num_classes = num_classes

    # get number of input channels for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def count_params(model):
    """Count the number of parameters"""
    param_count = np.sum([torch.numel(p) for p in model.parameters()])
    return param_count
