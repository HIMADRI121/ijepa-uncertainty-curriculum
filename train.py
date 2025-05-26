# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
'''
import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.imagenet1k import make_imagenet1k

from src.helper import (
    load_checkpoint,
    init_model,
    
    init_opt)
from src.transforms import make_transforms
from src.models.student import StudentWithUncertainty
from src.models.teacher import TeacherModel  # assuming you have a teacher
from src.models.vision_transformer import VisionTransformer
from helper import heteroscedastic_loss







# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    target_encoder = copy.deepcopy(encoder)

    # -- make data transforms
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)

    # -- init data-loaders/samplers
    _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=True)
    ipe = len(unsupervised_loader)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            def load_imgs():
                # -- unsupervised imgs
                imgs = udata[0].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2)
            imgs, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_context():
                    z = encoder(imgs, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    return z

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward_target()
                    z = forward_context()
                    loss = loss_fn(z, h)

                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats)
            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)
from src.models.teacher import get_teacher_model
from src.models.student import get_student_model
from src.utils.uncertainty import cosine_uncertainty

# Initialize models
teacher = get_teacher_model(config)
student = get_student_model(config)

for data in dataloader:
    inputs = data["image"]
    with torch.no_grad():
        teacher_output = teacher(inputs)

    student_output = student(inputs)

    # Calculate uncertainty (e.g., patch-wise)
    uncertainty = cosine_uncertainty(student_output, teacher_output)

    # Optionally: select top-k most uncertain patches
    # Apply mask, adjust loss computation accordingly

# Initialize teacher and student
teacher = TeacherModel(...).to(device)
teacher.eval()

student_backbone = VisionTransformer(...)  # or any encoder used
student = StudentWithUncertainty(student_backbone).to(device)

optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

for images, _ in dataloader:
    images = images.to(device)

    with torch.no_grad():
        teacher_embeds = teacher(images)

    student_mean, student_logvar = student(images)

    loss = heteroscedastic_loss(student_mean, student_logvar, teacher_embeds)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

uncertainty = torch.exp(student_logvar).mean(dim=1).detach().cpu().numpy()
avg_uncertainty = uncertainty.mean()
print(f"Epoch {epoch+1}, Avg Uncertainty: {avg_uncertainty:.4f}")

def main(args, resume_preempt=False):
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    # Initialization
    student = ...
    teacher = ...
    dataloader = ...

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for images, _ in dataloader:
            images = images.to(device)

            with torch.no_grad():
                teacher_embeds = teacher(images)

            student_mean, student_logvar = student(images)
            loss = heteroscedastic_loss(student_mean, student_logvar, teacher_embeds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Uncertainty log
        uncertainty = torch.exp(student_logvar).mean(dim=1).detach().cpu().numpy()
        avg_uncertainty = uncertainty.mean()
        print(f"Epoch {epoch+1}, Avg Uncertainty: {avg_uncertainty:.4f}")

print("Dataloader size:", len(dataloader))
'''

'''
import os
import sys
import yaml
import copy
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from src.utils.distributed import init_distributed, AllReduce
from src.utils.custom_logging import CSVLogger, gpu_timer, grad_logger, AverageMeter
from src.utils.tensors import repeat_interleave_batch
from src.datasets.imagenet1k import make_imagenet1k
from src.transforms import make_transforms
from src.helper import load_checkpoint, init_model, init_opt
from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.models.teacher import TeacherModel
from src.models.student import StudentWithUncertainty
from src.models.vision_transformer import VisionTransformer
from helper import heteroscedastic_loss

# Logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main(args, resume_preempt=False):
    # Load parameters from config
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    batch_size = args['data']['batch_size']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']

    patch_size = args['mask']['patch_size']
    crop_size = args['data']['crop_size']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']

    num_epochs = args['optimization']['epochs']
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    # Save config
    with open(os.path.join(folder, 'params.yaml'), 'w') as f:
        yaml.dump(args, f)

    # Init distributed
    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # Logging paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = os.path.join(folder, r_file) if load_model and r_file else latest_path

    csv_logger = CSVLogger(log_file, ('%d', 'epoch'), ('%d', 'itr'), ('%.5f', 'loss'), ('%.5f', 'uncertainty'), ('%d', 'time (ms)'))

    # Init models
    encoder, predictor = init_model(device=device, patch_size=patch_size, crop_size=crop_size, pred_depth=pred_depth, pred_emb_dim=pred_emb_dim, model_name=model_name)
    target_encoder = copy.deepcopy(encoder)
    target_encoder.eval()
    for p in target_encoder.parameters():
        p.requires_grad = False

    student = StudentWithUncertainty(encoder).to(device)
    teacher = target_encoder.to(device)

    student = DistributedDataParallel(student, static_graph=True)
    teacher = DistributedDataParallel(teacher)

    # Data
    transform = make_transforms(crop_size=crop_size, crop_scale=args['data']['crop_scale'], gaussian_blur=args['data']['use_gaussian_blur'], horizontal_flip=args['data']['use_horizontal_flip'], color_distortion=args['data']['use_color_distortion'], color_jitter=args['data']['color_jitter_strength'])

    mask_collator = MBMaskCollator(input_size=crop_size, patch_size=patch_size, pred_mask_scale=args['mask']['pred_mask_scale'], enc_mask_scale=args['mask']['enc_mask_scale'], aspect_ratio=args['mask']['aspect_ratio'], nenc=args['mask']['num_enc_masks'], npred=args['mask']['num_pred_masks'], allow_overlap=args['mask']['allow_overlap'], min_keep=args['mask']['min_keep'])

    _, loader, sampler = make_imagenet1k(transform=transform, batch_size=batch_size, collator=mask_collator, pin_mem=args['data']['pin_mem'], training=True, num_workers=num_workers, world_size=world_size, rank=rank, root_path=root_path, image_folder=image_folder, copy_data=args['meta']['copy_data'], drop_last=True)

    optimizer = torch.optim.Adam(student.parameters(), lr=args['optimization']['lr'])

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}")
        sampler.set_epoch(epoch)
        loss_meter = AverageMeter()
        unc_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, _, _) in enumerate(loader):
            imgs = udata[0].to(device, non_blocking=True)

            def step():
                with torch.no_grad():
                    teacher_output = teacher(imgs)
                student_mean, student_logvar = student(imgs)
                loss = heteroscedastic_loss(student_mean, student_logvar, teacher_output)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                uncertainty = torch.exp(student_logvar).mean(dim=1).detach().mean().item()
                return float(loss), uncertainty

            (loss, uncertainty), elapsed = gpu_timer(step)
            loss_meter.update(loss)
            unc_meter.update(uncertainty)
            time_meter.update(elapsed)

            csv_logger.log(epoch + 1, itr, loss, uncertainty, elapsed)

            if itr % 10 == 0:
                logger.info(f"[{epoch+1}, {itr}] loss: {loss_meter.avg:.4f}, uncertainty: {unc_meter.avg:.4f}, time: {time_meter.avg:.1f}ms")

        torch.save({
            'epoch': epoch + 1,
            'student': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss_meter.avg,
        }, latest_path)

        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch + 1,
                'student': student.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss_meter.avg,
            }, save_path.format(epoch=epoch+1))

if __name__ == '__main__':
    config_path = sys.argv[1]  # pass config yaml path as CLI arg
    args = load_config(config_path)
    main(args)
    '''
    
import torch
from datasets.cifar10_dataset import make_cifar10
from models.teacher import TeacherModel
from models.student import StudentWithUncertainty
from utils.losses import heteroscedastic_loss

# Config
BATCH_SIZE = 128
EPOCHS = 50
LR = 3e-4

def main():
    # Initialize models
    teacher = TeacherModel().cuda()
    student = StudentWithUncertainty().cuda()
    
    # Load dataset
    transform = ... # Use same transforms as before
    train_set, train_loader, _ = make_cifar10(
        transform=transform,
        batch_size=BATCH_SIZE,
        training=True
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    
    # Training loop
    for epoch in range(EPOCHS):
        for images, _ in train_loader:
            images = images.cuda()
            
            with torch.no_grad():
                teacher_features = teacher(images)
            
            student_mean, student_logvar = student(images)
            loss = heteroscedastic_loss(student_mean, student_logvar, teacher_features)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()