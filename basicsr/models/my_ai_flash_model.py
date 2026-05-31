import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.archs.my_restormer_arch import srgb_to_linear
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.visualize import visualize_intermediates

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


class AIFlashModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(AIFlashModel, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        self.best_metric = -float('inf')  # 初始化 best metric
        self.best_iter = 0  # 保存对应迭代次数
        self.metric_results = {}  # 初始化，避免分布式非 rank-0 访问报错

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('luminance_opt'):
            pixel_type = train_opt['luminance_opt'].pop('type')
            cri_illum_cls = getattr(loss_module, pixel_type)
            self.cri_illum = cri_illum_cls(**train_opt['luminance_opt']).to(
                self.device)
        else:
            self.cri_illum = None

        if train_opt.get('ssim_opt'):
            ssim_type = train_opt['ssim_opt'].pop('type')
            cri_ssim_cls = getattr(loss_module, ssim_type)
            self.cri_ssim = cri_ssim_cls(**train_opt['ssim_opt']).to(
                self.device)
        else:
            self.cri_ssim = None

        if train_opt.get('grad_opt'):
            grad_type = train_opt['grad_opt'].pop('type')
            cri_grad_cls = getattr(loss_module, grad_type)
            self.cri_grad = cri_grad_cls(**train_opt['grad_opt']).to(
                self.device)
        else:
            self.cri_grad = None

        if train_opt.get('color_opt'):
            color_type = train_opt['color_opt'].pop('type')
            cri_color_cls = getattr(loss_module, color_type)
            self.cri_color = cri_color_cls(**train_opt['color_opt']).to(
                self.device)
        else:
            self.cri_color = None

        if train_opt.get('perceptual_opt'):
            perceptual_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, perceptual_type)
            self.cri_perceptual = cri_perceptual_cls(**train_opt['perceptual_opt']).to(
                self.device)
        else:
            self.cri_perceptual = None

        # 新增: 输入重建损失 (L_input_recon)
        if train_opt.get('input_recon_opt'):
            recon_type = train_opt['input_recon_opt'].pop('type')
            cri_recon_cls = getattr(loss_module, recon_type)
            self.cri_input_recon = cri_recon_cls(**train_opt['input_recon_opt']).to(self.device)
        else:
            self.cri_input_recon = None

        # 新增: env_light 平滑损失 (TVSmoothLoss)
        if train_opt.get('smooth_opt'):
            smooth_type = train_opt['smooth_opt'].pop('type')
            cri_smooth_cls = getattr(loss_module, smooth_type)
            self.cri_smooth = cri_smooth_cls(**train_opt['smooth_opt']).to(self.device)
        else:
            self.cri_smooth = None

        if self.cri_pix is None and self.cri_ssim is None and self.cri_grad is None and self.cri_perceptual is None:
            raise ValueError('All losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        self.use_amp = False # 或从opt读取
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')

        # 损失记录器（用于绘制损失曲线）
        self.loss_history = {}
        self.val_loss_history = {}
        self.epoch_loss_sum = {}
        self.epoch_loss_count = 0

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.mask = data['mask'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        # 新增: 可控闪光训练 - 随机采样 alpha
        alpha_sampling = self.opt['train'].get('alpha_sampling', False)
        if alpha_sampling and self.is_train:
            self.alpha = float(np.random.beta(2, 2))
        else:
            self.alpha = 1.0

        if self.mixing_flag:
            self.gt, self.lq, self.mask = self.mixing_augmentation(self.gt, self.lq, self.mask)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        mask = data['mask_personmask']
        mask = torch.where(mask > 0, torch.ones_like(mask), torch.zeros_like(mask))
        self.mask = mask.to(self.device)

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        # 改动: 传入 alpha
        if self.use_amp:
            with torch.amp.autocast(device_type='cuda'):
                preds = self.net_g(self.lq, self.mask, alpha=self.alpha)
        else:
            preds = self.net_g(self.lq, self.mask, alpha=self.alpha)

        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        loss_dict = OrderedDict()
        l_total = 0

        # pixel loss
        if self.cri_pix:
            l_pix = 0.
            for pred in preds:
                l_pix += self.cri_pix(pred, self.gt, self.mask)
            loss_dict['l_pix'] = l_pix
            l_total += l_pix

        # luminance loss
        if self.cri_illum:
            l_illum = 0.
            for pred in preds:
                l_illum += self.cri_illum(pred, self.gt, self.mask)
            loss_dict['l_illum'] = l_illum
            l_total += l_illum

        # gradient loss
        if self.cri_grad:
            l_grad = 0.
            for pred in preds:
                l_grad += self.cri_grad(pred, self.gt)
            loss_dict['l_grad'] = l_grad
            l_total += l_grad

        # perceptual loss
        if self.cri_perceptual:
            l_percep = 0.
            for pred in preds:
                l_percep += self.cri_perceptual(pred, self.gt, self.mask)
            loss_dict['l_percep'] = l_percep
            l_total += l_percep

        # 新增: 输入重建损失 (L_input_recon)
        if self.cri_input_recon and hasattr(self.net_g, '_intermediate'):
            inter = self.net_g._intermediate
            # 计算输入重建: R * E 应该近似于输入 lq (转换到 linear 空间)
            input_recon = inter['reflectance'] * inter['env_light']
            input_linear = srgb_to_linear(self.lq)
            l_input_recon = self.cri_input_recon(input_recon, input_linear)
            loss_dict['l_input_recon'] = l_input_recon
            l_total += l_input_recon

            # 新增: env_light 平滑性损失 (Total Variation)，在inputloss的基础上计算，否则env没有空间分布
            if self.cri_smooth:
                env_light = inter['env_light']
                if env_light.shape[-1] > 1:  # 空间分布
                    l_smooth = self.cri_smooth(env_light)
                    loss_dict['l_smooth'] = l_smooth
                    l_total += l_smooth

        # 反向传播和优化
        if self.use_amp:
            self.scaler.scale(l_total).backward()
            if self.opt['train']['use_grad_clip']:
                self.scaler.unscale_(self.optimizer_g)
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
        else:
            l_total.backward()
            if self.opt['train']['use_grad_clip']:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        # 累加 loss（epoch 结束时取均值记录）
        for k, v in loss_dict.items():
            self.epoch_loss_sum[k] = self.epoch_loss_sum.get(k, 0) + (v.item() if torch.is_tensor(v) else v)
        self.epoch_loss_count += 1

    def _save_loss_curves(self, current_epoch):
        """绘制并保存损失曲线（Train 细线 + Val 粗线+标记点）"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        save_dir = os.path.join(self.opt['path']['experiments_root'], 'loss_curves')
        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # 主损失 (l_pix)
        ax = axes[0]
        # Train l_pix (epoch average, solid line)
        if 'l_pix' in self.loss_history:
            data = self.loss_history['l_pix']
            ax.plot(data['steps'], data['values'], 'b-s', linewidth=1.5, markersize=4, label='train l_pix (epoch avg)')
        # Val l_pix (thick, with markers)
        if 'val_l_pix' in self.val_loss_history:
            data = self.val_loss_history['val_l_pix']
            ax.plot(data['steps'], data['values'], 'r-o', linewidth=2, markersize=5, label='val l_pix')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Pixel Loss (l_pix)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 其他损失
        ax = axes[1]
        train_names = [k for k in self.loss_history if k not in ['l_pix', 'l_total']]
        val_base_names = [k.replace('val_', '') for k in self.val_loss_history if k.startswith('val_')]
        other_names = sorted(set(train_names) | set(val_base_names))
        colors = ['green', 'red', 'orange', 'purple', 'brown', 'pink']
        for i, name in enumerate(other_names):
            color = colors[i % len(colors)]
            # Train (epoch average)
            if name in self.loss_history:
                data = self.loss_history[name]
                ax.plot(data['steps'], data['values'], '-s', color=color, linewidth=1.5, markersize=4, label=f'{name} (train)')
            # Val (thick with markers)
            val_key = f'val_{name}' if not name.startswith('val_') else name
            if val_key in self.val_loss_history:
                data = self.val_loss_history[val_key]
                ax.plot(data['steps'], data['values'], '-o', color=color, linewidth=2, markersize=5, label=val_key)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Other Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'loss_curve_epoch_{current_epoch:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

        # 保存最新的一张作为 overview（必须在 close 之前）
        overview_path = os.path.join(save_dir, 'loss_curve_latest.png')
        plt.savefig(overview_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger = get_root_logger()
        logger.info(f'Saved loss curves at epoch {current_epoch}')

    def record_val_loss(self, val_loss_dict, current_epoch):
        """记录 validation loss 用于绘制曲线"""
        for k, v in val_loss_dict.items():
            key = f'val_{k}'
            if key not in self.val_loss_history:
                self.val_loss_history[key] = {'steps': [], 'values': []}
            self.val_loss_history[key]['steps'].append(current_epoch)
            self.val_loss_history[key]['values'].append(v)

    def end_epoch(self, current_epoch):
        """epoch 结束时，将累积的 loss 取均值记录到 loss_history"""
        if self.epoch_loss_count == 0:
            return
        for k, v in self.epoch_loss_sum.items():
            avg = v / self.epoch_loss_count
            if k not in self.loss_history:
                self.loss_history[k] = {'steps': [], 'values': []}
            self.loss_history[k]['steps'].append(current_epoch)
            self.loss_history[k]['values'].append(avg)
        self.epoch_loss_sum = {}
        self.epoch_loss_count = 0

    def boundary_mask(mask, k=5):
        dilated = F.max_pool2d(mask, k, stride=1, padding=k // 2)
        eroded = -F.max_pool2d(-mask, k, stride=1, padding=k // 2)
        return (dilated - eroded).clamp(0, 1)

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        mask = self.mask
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img, mask, alpha=1.0)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img, mask, alpha=1.0)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0., {}

    def _compute_val_losses(self):
        """计算当前验证样本的 loss 值，返回 dict"""
        losses = {}
        l_total = 0.0

        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt, self.mask)
            losses['l_pix'] = l_pix.item()
            l_total += l_pix.item()

        if self.cri_illum:
            l_illum = self.cri_illum(self.output, self.gt, self.mask)
            losses['l_illum'] = l_illum.item()
            l_total += l_illum.item()

        if self.cri_grad:
            l_grad = self.cri_grad(self.output, self.gt)
            losses['l_grad'] = l_grad.item()
            l_total += l_grad.item()

        if self.cri_perceptual:
            l_percep = self.cri_perceptual(self.output, self.gt, self.mask)
            losses['l_percep'] = l_percep.item()
            l_total += l_percep.item()

        if self.cri_input_recon and hasattr(self.net_g, '_intermediate'):
            inter = self.net_g._intermediate
            input_recon = inter['reflectance'] * inter['env_light']
            input_linear = srgb_to_linear(self.lq)
            l_input_recon = self.cri_input_recon(input_recon, input_linear)
            losses['l_input_recon'] = l_input_recon.item()
            l_total += l_input_recon.item()

        if self.cri_smooth and hasattr(self.net_g, '_intermediate'):
            inter = self.net_g._intermediate
            env_light = inter['env_light']
            if env_light.shape[-1] > 1:
                l_smooth = self.cri_smooth(env_light)
                losses['l_smooth'] = l_smooth.item()
                l_total += l_smooth.item()

        losses['l_total'] = l_total
        return losses

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0
        val_loss_accum = {}

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()

            # 计算 validation loss
            with torch.no_grad():
                val_losses = self._compute_val_losses()
            for k, v in val_losses.items():
                val_loss_accum[k] = val_loss_accum.get(k, 0.0) + v

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            max_vis_images = self.opt['val'].get('max_vis_images', 10)

            if save_img:
                if cnt < max_vis_images:
                    if self.opt['is_train']:

                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}.png')

                        save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                    img_name,
                                                    f'{img_name}_{current_iter}_gt.png')
                    else:

                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_gt.png')

                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)

                # 中间张量可视化（R, E, F, illumination）
                if cnt < max_vis_images and hasattr(self.net_g, '_intermediate'):
                    vis_dir = os.path.join(self.opt['path']['experiments_root'], 'val_visualizations')
                    visualize_intermediates(
                        self.net_g._intermediate,
                        save_dir=vis_dir,
                        iter_num=current_iter,
                        prefix=f'{img_name}_{current_iter}_'
                    )

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'].to('cuda'), visuals['gt'].to('cuda'),self.mask, **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

        # Average val losses
        val_loss_dict = {k: v / max(cnt, 1) for k, v in val_loss_accum.items()}

        # Log val losses
        val_log_str = 'Validation losses:'
        for k, v in val_loss_dict.items():
            val_log_str += f' {k}: {v:.4e}'
        logger = get_root_logger()
        logger.info(val_log_str)

        if tb_logger:
            for k, v in val_loss_dict.items():
                tb_logger.add_scalar(f'val_losses/{k}', v, current_iter)

        return current_metric, val_loss_dict

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
