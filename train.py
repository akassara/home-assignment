from model import create_fasterRCNN, count_params
from Preprocessing.Dataset import FoodDataset
import os
import json
from sklearn.model_selection import train_test_split
from train_utils import *
import torch

img_names = os.listdir('data/assignment_imgs')
img_annotations = open('data/img_annotations.json')
img_annotations = json.load(img_annotations)
info_df = pd.read_csv('data/label_mapping.csv')
img_dir = 'data/'



if torch.cuda.is_available():
    device='cuda'
else:
    device = 'cpu'
# our dataset has 2 classes: 0 for background and 1 for no tomatoes and  2for tomatoes
num_classes = 3
pretrained = True
device = True
# Instanciate model
model = create_fasterRCNN(pretrained,num_classes)
# move model to the right device
model.to(device)
# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.0001,
                            momentum=0.9, weight_decay=0.00005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.5)

print('Total parameters of Faster RCNN: ',count_params(model))

#Create Dataframes
df , nb_tomates, nb_objects = create_df(info_df,img_annotations)
print('proportion of tomatoes ',100*(nb_tomates/(nb_objects+nb_tomates)))
train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)
train_df , val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, shuffle=True)
train_df = train_df.reset_index()
val_df = val_df.reset_index()
test_df = test_df.reset_index()

batch_size = 4
#Creating datasets from dataframes
train_dataset = FoodDataset(image_dir = img_dir ,info_df = train_df ,input_size=(300,300), transform=complex_preprocess())
val_dataset = FoodDataset(image_dir = img_dir ,info_df = val_df ,input_size=(300,300), transform=None)
#Creating DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,sampler=None,
                                           collate_fn=collate_fn,drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=batch_size, shuffle=True,sampler=None,
                                           collate_fn=collate_fn,drop_last=True)

#### TRAINING
num_epochs = 15
save_frequency = 3
for epoch in range(num_epochs):

    # Train for one epoch, printing every 100 iterations
    train_his_ = train_one_epoch(model, optimizer, train_loader,batch_size, device, epoch, print_freq=100)

    # Compute losses over the validation set
    with torch.no_grad():
      val_his_ = validate_one_epoch(model, val_loader, device, print_freq=100)
      d, average_iou, acc = evaluate(model,val_loader,device, batch_size)
    val_his_['f1'] = d['f1']
    val_his_['average_iou'] = average_iou
    val_his_['acc'] = acc
    # Update the learning rate
    lr_scheduler.step()

    # Store loss values to plot learning curves afterwork.
    if epoch == 0:
        train_history = {k: [v] for k, v in train_his_.items()}
        val_history = {k: [v] for k, v in val_his_.items()}
    else:
        for k, v in train_his_.items():train_history[k] += [v]
        for k, v in val_his_.items():val_history[k] += [v]

    # Save the model
    if epoch % save_frequency==0:
      session_name = 'Test_session' + '_' + time.strftime('%m.%d %Hh%M')
      save_path = "/content/drive/My Drive/Foodvisor/challenge/model_zoo/"
      model_path = save_path  + session_name + '_'+ 'Epoch'+ str(epoch)+ '_model.pth.tar'
      torch.save(model, model_path)

    torch.cuda.empty_cache()