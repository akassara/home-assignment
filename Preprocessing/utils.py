def create_df(info_df, img_annotations):
    """
    Create df with useful informations for tomatoes allergy detection
    """
    image_paths = []
    labels = []
    bboxs = []
    is_tomates = []
    nb_tomates = 0
    nb_objects = 0
    for key in img_annotations.keys():
        image_path = key
        image_paths.append(key)
        nb_windows = len(img_annotations[key])
        img_labels = []
        img_bboxs = []
        tomate_per_img = []
        for k in range(nb_windows):
            annot_dict = img_annotations[key][k]

            if not annot_dict['is_background']:
                bbox = annot_dict['box']
                id = annot_dict['id']
                fr_label = info_df.loc[info_df['labelling_id'] == id]['labelling_name_fr'].item()
                # The images are going to be downsampled by 2 for computational reasons
                x1, y1, h, w = bbox[0] // 2, bbox[1] // 2, bbox[2] // 2 - 1, bbox[3] // 2 - 1
                # sometimes the bounding box is too big
                h = min(h, 299 - y1)
                w = min(w, 299 - x1)
                # we use the coco bbox convention
                img_bboxs.append([x1, y1, w, h])
                img_labels.append(fr_label)

                if 'Tomate' in fr_label or 'Raviolis sauce tomate' in fr_label:
                    tomate_per_img.append(1)
                    nb_tomates = nb_tomates + 1
                else:
                    nb_objects = nb_objects + 1
                    tomate_per_img.append(0)
        is_tomates.append(tomate_per_img)
        labels.append(img_labels)
        bboxs.append(img_bboxs)
    data = {'image_path': image_paths, 'label': labels, 'bbox': bboxs, 'is_tomato': is_tomates}
    df = pd.DataFrame(data)
    return df, nb_tomates, nb_objects

def complex_preprocess():
    return A.Compose([A.HorizontalFlip(),
                #A.RandomCrop(width=600, height=600,p=.9),
                A.VerticalFlip(),
                #A.Crop(x_min = 0, y_min = 0, x_max = 300, y_max = 300),
                #A.CLAHE(clip_limit=4,p=0.3),
                #A.ChannelDropout(p=0.2),
                #A.HueSaturationValue(p=0.2),
                #A.Posterize(p=0.2),
                #A.RGBShift(50,50,50,p=0.3),
                A.RandomGamma((40,120)),
                A.GaussNoise(p=0.3),
                A.Blur(blur_limit = 8,p=0.3),
                A.RandomContrast((-0.4,0.4))], bbox_params=A.BboxParams(format='coco',label_fields=['class_labels']))

def simple_preprocess():
    return A.Compose([ToTensor()
            ])

def to_numpy(x):
    if not (isinstance(x, np.ndarray) or x is None):
        if x.is_cuda:
            x = x.data.cpu()
        x = x.numpy()
    return x

def unnormalize(img,mean = mean, std = std,image_size=(300,300,3)):

  "Unnormalize a given image tensor and make it plotable"
  img= img.permute(1, 2, 0)
  # plt imshow only accept positive values
  unnormalized_img = torch.tensor(np.ones(image_size))
  for c in range(3):
    unnormalized_img[:,:,c] = std[c]*img[:,:,c] +mean[c]
  return to_numpy(unnormalized_img)

