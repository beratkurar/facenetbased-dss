from tensorboard.plugins import projector
import os
from tqdm import tqdm
import torch

def visualizeEmbeddings(model, dataloader, writer):
    progress_bar = enumerate(tqdm(dataloader))
    features = None
    class_labels = []
    label_imgages = None
    model.eval()
    with torch.no_grad():
        for batch_index, (data, raw_image, label) in progress_bar:
            data = data.cuda()
            output = model(data)
            class_labels += label

            if features is None:
                features = output.cpu()
            else:
                features = torch.cat((features, output.cpu()))

            if label_imgages is None:
                label_imgages = raw_image
            else:
                label_imgages = torch.cat((label_imgages, raw_image))

    label_imgages = torch.div(label_imgages, 255.)
    label_imgages = label_imgages.permute((0, 3, 1, 2))
    writer.add_embedding(features,
                         metadata=class_labels
                         #label_img=label_imgages
                         )
    model.train()
    print("Done!")


