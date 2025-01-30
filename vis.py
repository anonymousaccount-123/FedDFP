import torch
import os
from matplotlib import pyplot as plt

def plot_syn_image(image_pt, n_cnter=2):
    image = torch.load(image_pt)
    n_slide = image.size(0)
    slide_per_cnter = n_slide // n_cnter
    n_patch = image.size(1)
    n_row = 4
    n_col = n_patch // n_row
    fig, axs = plt.subplots(n_row, n_col, figsize=(20, 5))
    for c in range(n_cnter):
        center_save_pth = f'data/cam16/center_{c}'
        if not os.path.exists(center_save_pth):
            os.makedirs(center_save_pth)
        for s in range(slide_per_cnter):
            for i in range(n_row):
                plotted_img = None
                for j in range(n_col):
                    img_to_plot = image[c * slide_per_cnter + s][i * n_col + j].squeeze(0).cpu().permute(1,2,0)
                    img_to_plot[img_to_plot < 0] = 0.0
                    img_to_plot[img_to_plot > 1] = 1.0
                    if plotted_img is None:
                        plotted_img = img_to_plot
                    else:
                        # check if current image is same as previous image
                        if (plotted_img == img_to_plot).all():
                            print(f'Image {i * n_col + j} is same as previous image')
                        else:
                            print(f'Image {i * n_col + j} is different from previous image')
                            # show how different the images ar

                    axs[i, j].imshow(img_to_plot.numpy())
                    axs[i, j].axis('off')
            plt_save_pth = f'{center_save_pth}/slide_{s}.png'
            plt.savefig(plt_save_pth)
    plt.close()

if __name__ == '__main__':
    img_file = 'exp/fed_dm_CLAM_SB_ResNet50_base/best_data_0.pt'
    plot_syn_image(img_file, n_cnter=2)