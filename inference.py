from Preprocessing.utils import to_numpy, simple_preprocess
import skimage
import torch
import albumentations as  A



def has_tomatoes(img_path, model, device='cuda'):
    model.to(device)
    model.eval()
    image = skimage.io.imread(img_path)
    resize_transform = A.Resize(300, 300)(image=image)
    image = resize_transform['image']
    simple_transform = simple_preprocess()(image=image)
    image = simple_transform['image']

    image = image.unsqueeze(0)
    with torch.no_grad():
        prediction = model(image.to(device))
    pred_label = to_numpy(prediction[0]["labels"])  # [kept_boxes]
    return 1 in pred_label
