import torch
from torch import nn


def extract_conv_features(model, image):
    # we will save the conv layer weights in this list
    model_weights = []
    # we will save the 49 conv layers in this list
    conv_layers = []  # get all the model children as list
    model_children = list(
        model.features.resnet.children())  # counter to keep count of the conv layers
    counter = 0  # append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)

    #print(f"Total convolution layers: {counter}")
    # for conv_layer in conv_layers:
    #     print(conv_layer)

    outputs_f = []
    names = []

    for layer in conv_layers[0:]:
        image = layer(image)
        outputs_f.append(image)
        names.append(str(layer))
    print(len(outputs_f))  # print feature_maps
    # for feature_map in outputs_f:
    # print(feature_map.shape)

    processed = []
    for feature_map in outputs_f:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
        # for fm in processed:
        # print(fm.shape)

    # print(processed[-1].shape)
    # print(processed[0].shape)
    return processed
