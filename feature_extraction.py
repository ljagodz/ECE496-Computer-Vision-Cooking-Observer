import os
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import rcnn_data


def main():

    DATA_PATH = './data/dataset_by_run/dataset_by_run/'
    FEATURE_PATH = DATA_PATH + 'features/'

    # construct vgg model.
    vgg = models.vgg16(pretrained=True)

    # transform to apply to every image.
    data_transform = transforms.Compose([transforms.CenterCrop(154), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # initialize datasets, put into list as they are all distinct sequences.
    dataset_list = []
    for i in range(9):
        path_string = str(i) + '/'
        dataset_list.append(rcnn_data.ImageSequenceFolder(os.path.join(DATA_PATH, path_string), transform=data_transform))

    # may need to add mappings here?

    if not os.path.exists(FEATURE_PATH):
        os.mkdir(FEATURE_PATH)

    for i in range(9):
        file_number = 0  # all features assigned an arbitrary number for naming purposes.
        data_loader = DataLoader(dataset_list[i], batch_size=1, shuffle=False)
        for j, (rgb_image, img_image, label, name_img) in enumerate(data_loader):
            rgb_feature = vgg.features(rgb_image)
            img_feature = vgg.features(img_image)


            rgb_feature_tensor = torch.from_numpy(rgb_feature.detach().numpy())
            img_feature_tensor = torch.from_numpy(img_feature.detach().numpy())

            rgb_img_combined_tensor = torch.cat((rgb_feature_tensor, img_feature_tensor), dim=1)

            path_string = str(i) + '/'
            path_string = os.path.join(FEATURE_PATH, path_string)
            if not os.path.exists(path_string):
                os.mkdir(path_string)

            file_name = os.path.join(path_string, name_img[0] + '.tensor')
            torch.save(rgb_img_combined_tensor.squeeze(0), file_name)


if __name__ == "__main__":
    main()