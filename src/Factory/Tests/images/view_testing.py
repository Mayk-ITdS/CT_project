import matplotlib.pyplot as plt
import numpy as np

def debug_crop(ct_crop, mk_crop, center_idx=2):


    print("="*40)
    print("CT shape:", ct_crop.shape)
    print("Mask shape:", mk_crop.shape)
    print("Classes:", np.unique(mk_crop))
    print("Coverage:", (mk_crop > 0).sum() / mk_crop.size)

    fig, ax = plt.subplots(1, 4, figsize=(14,4))

    ax[0].imshow(ct_crop[center_idx - 2], cmap="gray")
    ax[0].set_title("z-2")

    ax[1].imshow(ct_crop[center_idx], cmap="gray")
    ax[1].set_title("center")

    ax[2].imshow(ct_crop[center_idx + 2], cmap="gray")
    ax[2].set_title("z+2")

    ax[3].imshow(ct_crop[center_idx], cmap="gray")
    ax[3].imshow(mk_crop, alpha=0.4, cmap="tab20")
    ax[3].set_title("overlay")

    for a in ax:
        a.axis("off")

    plt.suptitle("CT sliced & preprocessed")
    plt.tight_layout()
    plt.show()

def show_3slices(ct, mk):

    z = ct.shape[0] // 2
    y = ct.shape[1] // 2
    x = ct.shape[2] // 2

    fig, ax = plt.subplots(2,3, figsize=(12,8))

    # axial
    ax[0,0].imshow(ct[z], cmap="gray")
    ax[0,0].set_title("CT axial")

    ax[0,1].imshow(mk[z], cmap="jet",interpolation="nearest")
    ax[0,1].set_title("Mask axial")

    ax[0,2].imshow(ct[z], cmap="gray")
    ax[0,2].imshow(mk[z], alpha=0.4, cmap="tab20",interpolation="nearest")
    ax[0,2].set_title("Overlay axial")

    # coronal
    ax[1,0].imshow(ct[:,y,:], cmap="gray")
    ax[1,0].imshow(mk[:,y,:], alpha=0.4, cmap="tab20",interpolation="nearest")
    ax[1,0].set_title("Overlay coronal")

    # sagittal
    ax[1,1].imshow(ct[:,:,x], cmap="gray")
    ax[1,1].imshow(mk[:,:,x], alpha=0.4, cmap="tab20",interpolation="nearest")
    ax[1,1].set_title("Overlay sagittal")

    # mask sagittal
    ax[1,2].imshow(mk[:,:,x], cmap="jet",interpolation="nearest")
    ax[1,2].set_title("Mask sagittal")

    plt.tight_layout()
    plt.show()
