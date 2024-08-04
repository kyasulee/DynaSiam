import logging
import os
import time
import datetime
import torch
import random

import numpy as np
from torchvision.transforms import functional as F
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from compare_exp.utils.lr_scheduler import create_lr_scheduler
from datasetloader import DriveTrainDataset, DriveTestDataset
from train_utils.train_and_eval import train_one_epoch, evaluate
from dynasiam.src.DynaSiam import mmNet
from data_transforms import TDC_Enhance
from comput_mean_std import calculate_mean_and_std


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, normal_img, surface_img, mucosal_img, tone_img, mask):
        for t in self.transforms:
            normal_img, surface_img, mucosal_img, tone_img, mask = t(normal_img, surface_img, mucosal_img, tone_img, mask)
        return normal_img, surface_img, mucosal_img, tone_img, mask

class Resize(object):
    def __call__(self, normal_img, surface_img, mucosal_img, tone_img, mask):
        size = [224, 224]
        normal_img = F.resize(normal_img, size, antialias=True)
        surface_img = F.resize(surface_img, size, antialias=True)
        mucosal_img = F.resize(mucosal_img, size, antialias=True)
        tone_img = F.resize(tone_img, size, antialias=True)
        mask = F.resize(mask, size, antialias=True)
        return normal_img, surface_img, mucosal_img, tone_img, mask

class ToTensor(object):
    def __call__(self, normal_img, surface_img, mucosal_img, tone_img, mask):
        normal_img = F.to_tensor(normal_img)
        surface_img = F.to_tensor(surface_img)
        mucosal_img = F.to_tensor(mucosal_img)
        tone_img = F.to_tensor(tone_img)
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
        return normal_img, surface_img, mucosal_img, tone_img, mask

class Normalize(object):
    def __init__(self, normal_mean, normal_std, surface_mean, surface_std, mocusal_mean, mocusal_std, tone_mean, tone_std):
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.surface_mean = surface_mean
        self.surface_std = surface_std
        self.mocusal_mead = mocusal_mean
        self.mocusal_std = mocusal_std
        self.tone_mean = tone_mean
        self.tone_std = tone_std
    def __call__(self, normal_img, surface_img, mucosal_img, tone_img, mask):
        normal_img = F.normalize(normal_img,
                            mean=self.normal_mean,
                            std=self.normal_std)
        surface_img = F.normalize(surface_img,
                                  mean=self.surface_mean,
                                  std=self.surface_std)
        mucosal_img = F.normalize(mucosal_img,
                                  mean=self.mocusal_mead,
                                  std=self.mocusal_std)
        tone_img = F.normalize(tone_img,
                               mean=self.tone_mean,
                               std=self.tone_std)
        return normal_img, surface_img, mucosal_img, tone_img, mask


class SegmentationPreset:
    def __init__(self, normal_mean, normal_std, surface_mean, surface_std, mocusal_mean, mocusal_std, tone_mean, tone_std):
        self.transforms = Compose([
            Resize(),
            ToTensor(),
            Normalize(normal_mean, normal_std, surface_mean, surface_std, mocusal_mean, mocusal_std, tone_mean, tone_std),
        ])
    #
    def __call__(self,normal_img, surface_img, mucosal_img, tone_img, mask):
        return self.transforms(normal_img, surface_img, mucosal_img, tone_img, mask)

# #es-gastric
def get_transform_train(ori_mean, ori_std, te_mean, te_std, de_mean, de_std, ce_mean, ce_std):
    return SegmentationPreset(normal_mean=ori_mean,
                              normal_std=ori_std,
                              surface_mean=te_mean,
                              surface_std=te_std,
                              mocusal_mean=de_mean,
                              mocusal_std=de_std,
                              tone_mean=ce_mean,
                              tone_std=ce_std,)


def create_model():
    model = mmNet(is_training=True)
    return model

# First Stage: tdc_enhance
def image_enhance(args):
    image_path = Path(args.normal_image_path)

    print("------------------start processing training datasets------------------")

    # train dataset
    train_image_path = os.path.join(image_path, "train")
    train_normal_image_path = os.path.join(train_image_path, "normal")

    texture_enhance_save_path = os.path.join(train_image_path, "surface")
    texture_enhance_save_path = Path(texture_enhance_save_path)
    if not texture_enhance_save_path.exists():
        texture_enhance_save_path.mkdir()

    detail_enhance_save_path = os.path.join(train_image_path, "mocusal")
    detail_enhance_save_path = Path(detail_enhance_save_path)
    if not detail_enhance_save_path.exists():
        detail_enhance_save_path.mkdir()

    color_enhance_save_path = os.path.join(train_image_path, "tone")
    color_enhance_save_path = Path(color_enhance_save_path)
    if not color_enhance_save_path.exists():
        color_enhance_save_path.mkdir()

    train_image_enhance = TDC_Enhance(train_normal_image_path, texture_enhance_save_path, detail_enhance_save_path, color_enhance_save_path)
    train_image_enhance()

    # test dataset
    print("------------------start processing testing datasets------------------")

    test_image_path = os.path.join(image_path, "test")
    test_normal_image_path = os.path.join(test_image_path, "normal")
    texture_enhance_save_path = os.path.join(test_image_path, "surface")
    texture_enhance_save_path = Path(texture_enhance_save_path)
    if not texture_enhance_save_path.exists():
        texture_enhance_save_path.mkdir()

    detail_enhance_save_path = os.path.join(test_image_path, "mocusal")
    detail_enhance_save_path = Path(detail_enhance_save_path)
    if not detail_enhance_save_path.exists():
        detail_enhance_save_path.mkdir()

    color_enhance_save_path = os.path.join(test_image_path, "tone")
    color_enhance_save_path = Path(color_enhance_save_path)
    if not color_enhance_save_path.exists():
        color_enhance_save_path.mkdir()

    test_image_enhance = TDC_Enhance(test_normal_image_path, texture_enhance_save_path, detail_enhance_save_path, color_enhance_save_path)
    test_image_enhance()


# Second Stage: polyp segmentation
def main(args):
    random.seed(args.seed)

    save_path = Path(args.savepath)
    TIME_NAME = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if not save_path.exists():
        save_path.mkdir()
    (new_train := save_path / TIME_NAME).mkdir()
    (pth := new_train / "weight").mkdir()
    (res := new_train / "result").mkdir()
    print(f"Experiments res saved in : {str(new_train)}")

    batch_size = args.batch_size
    num_classes = args.num_classes

    # creat model
    print("------------------Creating Model------------------")
    model = mmNet()
    model.cuda()

    # Setting data
    print("------------------Calculate Mean & Std------------------")
    image_path = Path(args.normal_image_path)
    train_image_path = os.path.join(image_path, "train")

    print("Original images:")
    ori_mean, ori_std = calculate_mean_and_std(args.image_h, args.image_w, args.image_c,
                                               os.path.join(train_image_path, "normal"),
                                               os.path.join(train_image_path, "label"))
    print("Original images mean is: ", ori_mean)
    print("Original images std is: ", ori_std)

    print("Texture_enhance images:")
    te_mean, te_std = calculate_mean_and_std(args.image_h, args.image_w, args.image_c,
                                               os.path.join(train_image_path, "surface"),
                                               os.path.join(train_image_path, "label"))
    print("Texture_enhance images mean is: ", te_mean)
    print("Texture_enhance images std is: ", te_std)

    print("Detail_enhance images:")
    de_mean, de_std = calculate_mean_and_std(args.image_h, args.image_w, args.image_c,
                                             os.path.join(train_image_path, "mocusal"),
                                             os.path.join(train_image_path, "label"))
    print("Detail_enhance images mean is: ", de_mean)
    print("Detail_enhance images std is: ", de_std)

    print("Color_enhance images:")
    ce_mean, ce_std = calculate_mean_and_std(args.image_h, args.image_w, args.image_c,
                                             os.path.join(train_image_path, "tone"),
                                             os.path.join(train_image_path, "label"))
    print("Color_enhance images mean is: ", ce_mean)
    print("Color_enhance images std is: ", ce_std)

    logging.info(str(args))
    print("\n\n")
    print("------------------Loading Datasets------------------")
    train_dataset = DriveTrainDataset(args.train_datapath,train=True,
                                      transforms=get_transform_train(ori_mean, ori_std, te_mean, te_std, de_mean, de_std, ce_mean, ce_std))
    test_dataset = DriveTrainDataset(args.train_datapath, train=False,
                                     transforms=get_transform_train(ori_mean, ori_std, te_mean, te_std, de_mean, de_std, ce_mean, ce_std))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               drop_last=True,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1,
                                             num_workers=0,
                                             shuffle=False,
                                             pin_memory=True)


    # Setting learning schedule and optimizer
    params_to_optimize = [{'params': model.parameters(),
                           'lr': args.lr,
                           'weight_decay': args.weight_decay}]
    optimizer = torch.optim.Adam(params_to_optimize, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    lr_scheduler = create_lr_scheduler(optimizer,
                                       num_step=len(train_loader),
                                       epochs=args.epochs,
                                       warmup_epochs=1,
                                       warmup=True)

    # Pytorch自动混合精度(AMP)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['model'])
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
    logging.info(str(model))

    start_time = time.time()
    writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

    result_file = str(res / "result.txt")

    model_msg = str(model)
    with open(result_file, 'a') as f:
        f.write(model_msg + "\n" )

    logging.info('------------------Start Training------------------')
    print('------------------Start Training------------------')
    for epoch in range(args.start_epoch, args.epochs):
        train_msg = train_one_epoch(model=model,
                                     optimizer=optimizer,
                                     train_loader=train_loader,
                                     writer=writer,
                                     epoch=epoch,
                                     num_classes=num_classes,
                                     total_epochs=args.epochs,
                                     lr_schedule=lr_scheduler)
        print("train messages:\n" + train_msg)
        test_msg = evaluate(model=model,
                             valid_loader=test_loader,
                             num_classes=num_classes,
                             is_valid=True)
        print("test messages:\n" + test_msg)

        with open(result_file, 'a') as f:
            f.write("train messages:\n" + train_msg + "\n\n" +
                    "test messages:\n" + test_msg + "\n\n"
                    )

        train_acc_file = str(res / "train_acc.txt")
        train_acc_idx = train_msg.find("all_acc") + len("all_acc") + 1
        train_acc_value = train_msg[train_acc_idx:train_msg.find("\n", train_acc_idx)]
        with open(train_acc_file, 'a') as f:
            f.write(train_acc_value.strip() + "\n")

        test_acc_file = str(res / "test_acc.txt")
        test_acc_idx = test_msg.find("all_acc") + len("all_acc") + 1
        test_acc_value = test_msg[test_acc_idx:test_msg.find("\n", test_acc_idx)]
        with open(test_acc_file, 'a') as f:
            f.write(test_acc_value.strip() + "\n")

        train_loss_file = str(res / "train_loss.txt")
        train_loss_idx = train_msg.find("fuse_total_loss") + len("fuse_total_loss") + 1
        train_loss_value = train_msg[train_loss_idx:train_msg.find("\n", train_loss_idx)]
        with open(train_loss_file, 'a') as f:
            f.write(train_loss_value.strip() + "\n")

        test_loss_file = str(res / "test_loss.txt")
        test_loss_idx = test_msg.find("fuse_total_loss") + len("fuse_total_loss") + 1
        test_loss_value = test_msg[test_loss_idx:test_msg.find("\n", test_loss_idx)]
        with open(test_loss_file, 'a') as f:
            f.write(test_loss_value.strip() + "\n")

        if epoch == args.epochs - 1:
            torch.save({
                'epoch': epoch,
                'args':args,
                'model': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                },
               pth / f'model_{epoch+1}.pth')

    total_time = time.time() - start_time
    total_time_hours = (total_time) / 3600
    time_msg = "total training time {:.4f}".format(total_time_hours)
    logging.info(time_msg)

    print("total training time {:.4f}".format(total_time_hours))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch multi-modal polyp segmentation training")
    parser.add_argument("--information", default="")

    parser.add_argument("--train_datapath", default="your_path", help="normal dataset root")
    parser.add_argument("--savepath", default="your_path", help="result save root")
    parser.add_argument("--normal_image_path", default="your_path", help="original gastroscopy image root")

    parser.add_argument("--image_h", default=224, type=int, help="image size")
    parser.add_argument("--image_w", default=224, type=int, help="image size")
    parser.add_argument("--image_c", default=3, type=int, help="image channel")

    parser.add_argument("--num_classes", default=2, type=int, help="background + polyp")
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs")
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument("--seed", default=77, type=int)
    parser.add_argument('--save_best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # First Stage: image enhance
    print("First Stage: image enhance")
    image_enhance(args)

    # Second Stage: polyp segmentation
    print("Second Stage: polyp segmentation")
    main(args)

    # os.system('shutdown')
