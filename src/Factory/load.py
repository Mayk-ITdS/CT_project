import logging
import time
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import faulthandler
import psutil, os
from src.config.logger import setup_logging
from src.tests.volumes_tests import build_dataloaders, load_my_model, custom_lossfn, dice_score, evaluate_model, \
    evaluate_model, CTDataset, visualize_napari, visualize_test_sample

# dataset_path = Path("Dataset")
#
# if dataset_path.exists():
#     shutil.rmtree(dataset_path)
#
# dataset_path.mkdir(parents=True, exist_ok=True)
#
# for f in ["meta_index_raw.csv", "meta_index_full.csv"]:
#     if Path(f).exists():
#         Path(f).unlink()

DATA_MAIN_DIR = Path(r"D:\Kozmin\jadrakostnienia2024\jadrakostnienia2024")

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "src/Factory/dataset/Dataset"
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

setup_logging()

logger = logging.getLogger(__name__)
logger.debug(torch.__version__)
logger.debug(torch.cuda.is_available())
logger.debug(torch.cuda.get_device_name(0))

def arrange_difficulties(idx_list,p33,p66):
    easy, medium, hard = [], [], []

    for i in range(len(idx_list)):
        data = torch.load(idx_list[i], map_location="cpu")
        mk = data["mk"]

        fill = (mk > 0).sum().item() / mk.numel()

        if fill >= p66:
            easy.append(i)
        elif fill >= p33:
            medium.append(i)
        else:
            hard.append(i)

    return easy, medium, hard

class DifficultyBatchSampler(torch.utils.data.BatchSampler):

    def __init__(self, easy, medium, hard, batch_size=4):
        self.easy = easy
        self.medium = medium
        self.hard = hard
        self.batch_size = batch_size
        self.length = len(easy) + len(medium) + len(hard)

    def __iter__(self):

        pool = []
        if self.easy:
            pool += self.easy
        if self.medium:
            pool += self.medium
        if self.hard:
            pool += self.hard

        # bias easy
        if self.easy:
            pool += self.easy

        for _ in range(self.length // self.batch_size):

            batch = []

            while len(batch) < self.batch_size:
                batch.append(random.choice(pool))

            yield batch

    def __len__(self):
        return self.length // self.batch_size



def debug_one_sample(train_files):
    path = train_files[0]
    data = torch.load(path,weights_only=True)

    print("CT:", data["ct"].shape)
    print("MK:", data["mk"].shape)

    print("CT:", data["ct"].shape)

def pipeline():

    time.sleep(2.5)
    logger.info(
        "=================================== Pipeline Start =================================================")

    torch.backends.cudnn.benchmark = True
    process = psutil.Process(os.getpid())
    logger.debug(f"RAM {process.memory_info().rss / 1024 ** 2:.1f} MB")
    faulthandler.enable()
    HERE = Path(__file__).parent
    principal_path = ROOT / "src/Factory/meta_index_full.csv"
    try:
        meta = pd.read_csv(ROOT / "src/Factory/meta_index_full.csv")
        p33 = meta["real_fill"].quantile(0.33)
        p66 = meta["real_fill"].quantile(0.66)
        case_ids = meta['case_id'].unique()

        if len(case_ids) == 0:
            logger.error("No case_ids found!")
            raise ValueError("Empty dataset")

        logger.info(f"IQR coverage: p33={p33:.3f}, p66={p66:.3f}")
        logger.info(f"Pipeline finally finished successfully...")
    except Exception:
        logger.error("Pipeline crashed at meta index...",exc_info=True)
        raise

    logger.info('==============================================Load Preprocessed Data===========================================')

    train_ids, test_ids = train_test_split(case_ids, test_size=0.2,random_state=42)
    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=0.1,
        random_state=42
    )

    train_files_2_5D = meta[meta['case_id'].isin(train_ids)]['file'].to_list()
    val_files_2_5D = meta[meta['case_id'].isin(val_ids)]['file'].tolist()
    test_files_2_5D = meta[meta['case_id'].isin(test_ids)]['file'].tolist()



    # print('Jestem tuz przed zbieraniem train indexow')
    # print("Jestem przed arrange diffs")
    # easy_train,medium_train,hard_train = arrange_difficulties(train_files_2_5D,p33,p66)
    #
    # print("TRAIN DATASET LENGTH:", len(train_files_2_5D))
    # print('Test ids : ',train_ids)
    # print("easy:", len(easy_train))
    # print("medium:", len(medium_train))
    # print("hard:", len(hard_train))
    # sampler_train = DifficultyBatchSampler(easy_train, medium_train, hard_train)
    #
    # train_loader, val_loader, test_loader = build_dataloaders(sampler_train,train_files_2_5D, val_files_2_5D, test_files_2_5D)

    print("==================================== Now starting DataLoaders =========================================")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_my_model('model_last_v.pth',device)
    print(model)
    # results = evaluate_model(model,test_loader,device)
    # test_dataset = CTDataset(test_files_2_5D)
    # print(results)
    # for i in range(500,600,10):
    #     visualize_test_sample(model, test_dataset, device, idx=i, save=False)
    #     visualize_napari(model,test_dataset,device,idx=i)

# def train_model(model, train_loader,custom_lossfn, device):
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#     for epoch in range(30):
#         pred_true_dict = {
#             "GT classes":set(),
#             "Predicted classes":set(),
#         }
#         model.train()
#         train_loss = 0
#
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
#
#         for ct, mask in pbar:
#
#             ct = ct.to(device,non_blocking=True)
#             mask = mask.to(device,non_blocking=True)
#             if mask.max() > 7 or mask.min() < 0:
#                 print("BAD MASK DETECTED")
#                 print("min:", mask.min().item(), "max:", mask.max().item())
#                 print("unique:", torch.unique(mask))
#                 raise ValueError("Mask out of bounds")
#             optimizer.zero_grad()
#
#             pred = model(ct)
#             loss = custom_lossfn(pred,mask)
#
#             loss.backward()
#             optimizer.step()
#
#             train_loss += loss.item()
#
#             pred_true_dict['GT classes'].update(torch.unique(mask).cpu().numpy())
#             pred_true_dict['Predicted classes'].update(torch.unique(torch.argmax(pred,dim=1)).cpu().numpy())
#
#             pbar.set_postfix(loss=loss.item())
#         train_loss /= len(train_loader)
#         print(f"Epoch {epoch} Train Loss: {train_loss}")
#         print(f"Epoch {epoch} GT classes: {pred_true_dict['GT classes']}")
#         print(f"Epoch {epoch} Predicted classes: {pred_true_dict['Predicted classes']}")
#         model.eval()
#
#         val_loss = 0
#         dice_total = 0
#         all_predval_classes = set()
#         all_trueval_classes = set()
#     #
    #     with torch.no_grad():
    #
    #         for ct, mask in tqdm(val_loader):
    #             ct = ct.to(device)
    #             mask = mask.to(device)
    #
    #             pred = model(ct)
    #             pred_mask = torch.argmax(pred, dim=1)
    #             loss = custom_lossfn(pred,mask)
    #             val_loss += loss.item()
    #
    #             dice_total += dice_score(pred, mask)
    #
    #             all_trueval_classes.update(torch.unique(mask).cpu().numpy())
    #             all_predval_classes.update(torch.unique(pred_mask).cpu().numpy())
    #
    #         val_loss /= len(val_loader)
    #         dice_avg = dice_total / len(val_loader)
    #         if dice_avg > best_dice:
    #             best_dice = dice_avg
    #             best_epoch = epoch
    #
    #             torch.save(model.state_dict(), "best_model.pth")
    #     history.append({
    #         "epoch": epoch,
    #         "train_loss": float(train_loss),
    #         "val_loss": float(val_loss),
    #         "val_dice": float(dice_avg)
    #     })
    #     print("GT val classes:", sorted(all_trueval_classes))
    #     print("Pred val classes:", sorted(all_predval_classes))
    #
    #     print(f"Epoch {epoch} Val Loss: {val_loss} Dice: {dice_avg}")
    #     torch.save(model.state_dict(), f"model_last_v.pth")


if __name__ == '__main__':
    try:
        start = time.time()
        pipeline()
        end = time.time()
    except Exception as e:
        print(e)






# print("tu printuje maski",records_ok['mask'])
# mismatched = []
# slices_with_masks = []
# no_zero_voxels_per_mask = []
# print(records_ok)
# print('Unique test')
# print(records.groupby("sample_nr")["ct"].nunique().value_counts())
# print("rekordy ok:", len(records_ok))
#
# print(slices_with_masks)
# print(no_zero_voxels_per_mask)
#
#
# for i,ct_path in enumerate(records_ok["ct"]):
#       ct, mk = load_pair_from_ct(Path(ct_path))
#       if i == 0:
#           view_one3D(ct, mk)
#       if ct.shape != mk.shape:
#           mismatched.append(Path(ct))
#           raise ValueError(f"CT shape mismatch: {ct.shape} != {mk.shape}")
#       slice_count = 0
#       classes, counts = np.unique(mk, return_counts=True)
#
#       print(f"\n{ct_path.name}")
#       for c, n in zip(classes, counts):
#           print(f"class {c}: {n}")
#       for z in range(mk.shape[0]):
#         if np.any(mk[z] > 0):
#             slice_count += 1
#       mask_voxels = np.count_nonzero(mk)
#       ratio = mask_voxels / mk.size
#       # show_3slices(i,ct,mk)
#       no_zero_voxels_per_mask.append(f"{ct_path.name}, no_zero voxels in mask: {mask_voxels}, ratio: {ratio}")
#       slices_with_masks.append(f"{ct_path.name}, slices with mask:, {slice_count}")
#       print(ct.shape, mk.shape)
#       print('Unique classes :',ct.shape, mk.shape, np.unique(mk)[:30])
#       count_vox_per_class(mk)
#       # outlieres verification
#
#       print(f'Ct min-max from path:{ct_path} : {ct.min()}, {ct.max()}')
#       print(f'percentiles :',np.percentile(ct, [1, 5, 50, 95, 99]))
#
# #all classes
#
# plt.hist(ratios, bins=20)
# plt.title("Mask voxel ratio distribution")
# plt.show()
#
# for ct_path in records_ok["ct"]:
#     ct, mk = load_pair_from_ct(Path(ct_path))
#     ratio = np.count_nonzero(mk) / mk.size
#     ratios.append(ratio)
#
