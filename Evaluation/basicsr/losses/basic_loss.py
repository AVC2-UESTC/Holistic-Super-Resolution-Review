import torch
import numpy as np
from torch import nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms.functional as TF

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class CurvMap(nn.Module):
    def __init__(self, scale=1):
        super(CurvMap, self).__init__()
        self.scale = scale
        self.requires_grad = False

    def rgb_to_ycbcr(self, img):
        """
        Convert an RGB image to YCbCr.
        """
    # def rgb_to_ycbcr_y_channel(img):
        if img.dim() != 4 or img.size(1) != 3:
            raise ValueError("Expected input to be a 4D tensor with 3 channels (RGB format)")
        
        mat = torch.tensor([0.299, 0.587, 0.114]).to(img.device)

        # 变换维度以适应矩阵乘法
        reshaped_img = img.permute(0, 2, 3, 1)  # 将形状变为 (batch, height, width, channel)

        # 计算Y通道
        y_channel = torch.tensordot(reshaped_img, mat, dims=([3], [0]))

        # 将结果调整为4维张量 (batch, 1, height, width)
        y_channel_4d = y_channel.unsqueeze(1)

        return y_channel_4d

    def forward(self, img):
        img = img / self.scale
        # Convert image to YCbCr and use only the Y channel
        # img_y, _, _ = TF.rgb_to_ycbcr(img).chunk(3, dim=1)
        img = self.rgb_to_ycbcr(img)
        # img = TF.rgb_to_grayscale(img)
        img_pad = F.pad(img, pad=(1, 1, 1, 1), mode='reflect')

        N, C, H, W = img.shape
        gradX = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        gradY = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        gradXX = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        gradXY = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        gradYY = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)

        gradx = (img[..., 1:, :] - img[..., :-1, :]).abs()
        grady = (img[..., 1:] - img[..., :-1]).abs()
        gradxx = (img_pad[..., 2:, 1:-1] + img_pad[..., :-2, 1:-1] - 2 * img_pad[..., 1:-1, 1:-1]).abs()
        gradyy = (img_pad[..., 1:-1, 2:] + img_pad[..., 1:-1, :-2] - 2 * img_pad[..., 1:-1, 1:-1]).abs()
        gradxy = (img_pad[..., 2:, 2:] + img_pad[..., 1:-1, 1:-1] - img_pad[..., 2:, 1:-1] - img_pad[..., 1:-1, 2:]).abs()

        gradX[..., :-1, :] += gradx
        gradX[..., 1:, :] += gradx
        gradX[..., 1:-1, :] /= 2

        gradY[..., :-1] += grady
        gradY[..., 1:] += grady
        gradY[..., 1:-1] /= 2
        gradXX = gradxx
        gradYY = gradyy
        gradXY = gradxy

        curv = (gradYY*(1 + torch.square(gradX)) - 2 * gradXY * gradX * gradY + gradXX * (1 + torch.square(gradY))) / \
               torch.sqrt(torch.pow((torch.square(gradX) + torch.square(gradY) + 1), 3))

        return curv


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class L1LossEdge(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1LossEdge, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        
        # self.alpha = alpha
        self.loss_weight = loss_weight
        self.reduction = reduction
        # self.loss_cor = LocalCorrelationLoss()
        self.loss_edge = EdgeLoss()
        # self.loss_vgg = VGGPerceptualLoss(False)


    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        # loss_vgg = self.loss_vgg(pred, target)

        return l1_loss(pred, target, weight, reduction=self.reduction) + self.loss_weight * self.loss_edge(pred, target) # + self.alpha * loss_vgg
        # return l1_loss(pred, target, weight, reduction=self.reduction) + self.loss_cor(pred, target)



@LOSS_REGISTRY.register()
class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

@LOSS_REGISTRY.register()
class L1LossCurv(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1LossCurv, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.curv = CurvMap()
        self.loss_curv = nn.L1Loss()
    
    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return l1_loss(pred, target, weight, reduction=self.reduction) + self.loss_weight * self.loss_curv(self.curv(pred), self.curv(target))


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

# @LOSS_REGISTRY.register()
# class LocalCorrelationLoss(nn.Module):
#     def __init__(self, blocks_num=1E10, blocks_ratio=10):
#         super(LocalCorrelationLoss, self).__init__()
#         self.blocks_num = blocks_num
#         self.blocks_ratio = blocks_ratio

#     def calculate_glcm(self, image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
#         device = image.device
#         glcm = torch.zeros((image.shape[0], levels, levels, len(distances), len(angles)), device=device)

#         for d, distance in enumerate(distances):
#             for a, angle in enumerate(angles):
#                 displacement = (int(np.round(distance * np.cos(angle))), int(np.round(distance * np.sin(angle))))
#                 for i in range(image.shape[2]):
#                     for j in range(image.shape[3]):
#                         pixel = image[:, 0, i, j]  
#                         neighbor_i = i + displacement[0]
#                         neighbor_j = j + displacement[1]
                        
#                         if 0 <= neighbor_i < image.shape[2] and 0 <= neighbor_j < image.shape[3]:
#                             neighbor_pixel = image[:, 0, neighbor_i, neighbor_j]
#                             glcm[:, pixel, neighbor_pixel, d, a] += 1
#                             glcm[:, neighbor_pixel, pixel, d, a] += 1

#         return glcm

#     def calculate_pearson_correlation(self, img1, img2):
#         a_flat = img1.reshape(img1.shape[0], -1)
#         b_flat = img2.reshape(img2.shape[0], -1)

#         correlations = 0.0
#         for i in range(a_flat.shape[0]):
#             # correlation = pearsonr(a_flat[i].cpu().numpy(), b_flat[i].cpu().numpy())
#             # correlation = np.corrcoef(a_flat[i].cpu().detach().numpy(), b_flat[i].cpu().detach().numpy())[0,1]
#             flat = torch.stack([a_flat[i], b_flat[i]])
#             correlation = torch.corrcoef(flat)[0, 1].item()
#             correlations += abs(abs(correlation)-1)
            
#         return correlations / (a_flat.shape[0] + 1e-8)

#     def is_strong_texture(self, glcm):
#         energy_values = torch.sum(glcm, dim=(1, 2))  # 能量
#         contrast_values = torch.var(glcm, dim=(1, 2))  # 对比度
#         entropy_values = -torch.sum(glcm * torch.log(glcm + 1e-10), dim=(1, 2))  # 熵

#         combined_scores = energy_values + contrast_values + entropy_values

#         return torch.argsort(combined_scores, descending=True)
    
#     def rgb2YCbCr(self, img):
#         yuv_image = torch.empty_like(img)
#         r = img[:, 0, :, :]
#         g = img[:, 1, :, :]
#         b = img[:, 2, :, :]

#         yuv_image[:, 0, :, :] = 0.299 * r + 0.587 * g + 0.114 * b  # Y
#         yuv_image[:, 1, :, :] = -0.169 * r - 0.331 * g + 0.500 * b + 128  # Cb
#         yuv_image[:, 2, :, :] = 0.500 * r - 0.419 * g - 0.081 * b + 128  # Cr

#         return yuv_image

#     # def calculate_pearson_correlation(self, img1, img2):
#     #     """
#     #     Calculate the Pearson correlation coefficient between two images.
#     #     Assumes images are 4D tensors (batch_size, channels, height, width).

#     #     :param img1: Tensor of the first image.
#     #     :param img2: Tensor of the second image.
#     #     :return: Pearson correlation coefficient.
#     #     """
#     #     # Reshape images to 2D (batch_size, features)
#     #     a_flat = img1.reshape(img1.size(0), -1)
#     #     b_flat = img2.reshape(img2.size(0), -1)

#     #     # Calculate means
#     #     mean_a = torch.mean(a_flat, dim=1, keepdim=True)
#     #     mean_b = torch.mean(b_flat, dim=1, keepdim=True)

#     #     # Subtract means
#     #     a_flat_sub = a_flat - mean_a
#     #     b_flat_sub = b_flat - mean_b

#     #     # Compute Pearson correlation coefficient
#     #     r_num = torch.sum(a_flat_sub * b_flat_sub, dim=1)
#     #     r_den = torch.sqrt(torch.sum(a_flat_sub ** 2, dim=1) * torch.sum(b_flat_sub ** 2, dim=1))
#     #     r = r_num / (r_den + 1e-8)  # Add a small value to avoid division by zero

#     #     # Average over the batch
#     #     return torch.mean(r)
        
#     # def forward(self, img, gt, cmap='YCbCr'):
#     #     assert img.shape == gt.shape and img.shape[1] == 3, "should be the same shape and RGB-channels"
        
#     #     # device = img.device
#     #     _, _, height, width = img.shape

#     #     img_gray = torch.mean(img, dim=1, keepdim=True)
#     #     gt_gray = torch.mean(gt, dim=1, keepdim=True)
        
#     #     window_h = height // self.blocks_ratio
#     #     window_w = width // self.blocks_ratio
        
#     #     num = 0
#     #     total_loss = 0.0
        
#     #     for i in range(0, height - window_h + 1, window_h // 3):
#     #         for j in range(0, width - window_w + 1, window_w // 3):
#     #             end_i = min(i + window_h, height)
#     #             end_j = min(j + window_w, width)
                
#     #             # window_gt = gt_gray[:, :, i:end_i, j:end_j]
#     #             # glcm_gt = self.calculate_glcm(window_gt)

#     #             # idxs = self.is_strong_texture(glcm_gt)
#     #             # top_percentage = 0.7
#     #             # num_to_select = int(idxs.shape[0] * top_percentage)
#     #             # top_indices = idx[:num_to_select]
                
#     #             if cmap == "YCbCr":
#     #                 block_img = self.rgb2YCbCr(img)[:, -1, i:end_i, j:end_j]
#     #                 block_gt = self.rgb2YCbCr(gt)[:, -1, i:end_i, j:end_j]
#     #             elif cmap == "GRAY":
#     #                 block_img = img_gray[:, 0, i:end_i, j:end_j]
#     #                 block_gt = gt_gray[:, 0, i:end_i, j:end_j]

#     #             loss = self.calculate_pearson_correlation(block_img, block_gt)
#     #             # print(loss)
#     #             total_loss += loss
#     #             num += 1
#     #             if num >= self.blocks_num:
#     #                 break

#     #     return total_loss / num
#     def forward(self, img, gt, cmap='YCbCr'):
#         assert img.shape == gt.shape and img.shape[1] == 3, "should be the same shape and RGB-channels"
        
#         _, _, height, width = img.shape
#         img_gray = torch.mean(img, dim=1, keepdim=True)
#         gt_gray = torch.mean(gt, dim=1, keepdim=True)

#         window_h = height // self.blocks_ratio
#         window_w = width // self.blocks_ratio

#         # 预先进行颜色空间转换
#         if cmap == "YCbCr":
#             img_cvt = self.rgb2YCbCr(img)[:, -1, :, :]
#             gt_cvt = self.rgb2YCbCr(gt)[:, -1, :, :]
#         elif cmap == "GRAY":
#             img_cvt = img_gray[:, 0, :, :]
#             gt_cvt = gt_gray[:, 0, :, :]

#         total_loss = 0.0
#         num = 0

#         # 遍历并计算每个窗口的损失
#         for i in range(0, height - window_h + 1, window_h // 3):
#             for j in range(0, width - window_w + 1, window_w // 3):
#                 end_i = min(i + window_h, height)
#                 end_j = min(j + window_w, width)

#                 block_img = img_cvt[:, i:end_i, j:end_j]
#                 block_gt = gt_cvt[:, i:end_i, j:end_j]

#                 loss = self.calculate_pearson_correlation(block_img, block_gt)
#                 total_loss += loss
#                 num += 1
#                 if num >= self.blocks_num:
#                     break

#         return total_loss / num

@LOSS_REGISTRY.register()
class LocalCorrelationLoss(nn.Module):
    def __init__(self, blocks_num=1E10, blocks_ratio=10):
        super(LocalCorrelationLoss, self).__init__()
        self.blocks_num = blocks_num
        self.blocks_ratio = blocks_ratio

    def calculate_glcm(self, image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
        device = image.device
        glcm = torch.zeros((image.shape[0], levels, levels, len(distances), len(angles)), device=device)

        for d, distance in enumerate(distances):
            for a, angle in enumerate(angles):
                displacement = (int(np.round(distance * np.cos(angle))), int(np.round(distance * np.sin(angle))))
                for i in range(image.shape[2]):
                    for j in range(image.shape[3]):
                        pixel = image[:, 0, i, j]  
                        neighbor_i = i + displacement[0]
                        neighbor_j = j + displacement[1]
                        
                        if 0 <= neighbor_i < image.shape[2] and 0 <= neighbor_j < image.shape[3]:
                            neighbor_pixel = image[:, 0, neighbor_i, neighbor_j]
                            glcm[:, pixel, neighbor_pixel, d, a] += 1
                            glcm[:, neighbor_pixel, pixel, d, a] += 1

        return glcm
    
    # @PerformanceAnalyzer.measure_time
    # def calculate_pearson_correlation(self, img1, img2):
    #     a_flat = img1.reshape(img1.shape[0], -1)
    #     b_flat = img2.reshape(img2.shape[0], -1)

    #     correlations = 0.0
    #     for i in range(a_flat.shape[0]):
    #         # correlation = pearsonr(a_flat[i].cpu().numpy(), b_flat[i].cpu().numpy())
    #         # correlation = np.corrcoef(a_flat[i].detach().cpu().numpy(), b_flat[i].detach().cpu().numpy())[0,1]
    #         flat = torch.stack([a_flat[i], b_flat[i]])
    #         correlation = torch.corrcoef(flat)[0,1].item()
    #         correlations += abs(abs(correlation)-1)
            
    #     return correlations / a_flat.shape[0]
    
    # @PerformanceAnalyzer.measure_time
    def calculate_pearson_correlation(self, img1, img2):

        img1_flat = img1.reshape(img1.shape[0], img1.shape[1], -1)
        img2_flat = img2.reshape(img2.shape[0], img2.shape[1], -1)
        
        mean_img1 = img1_flat.mean(dim=2, keepdim=True)
        mean_img2 = img2_flat.mean(dim=2, keepdim=True)
        
        std_img1 = img1_flat.std(dim=2, keepdim=True) + 1e-10
        std_img2 = img2_flat.std(dim=2, keepdim=True) + 1e-10

        correlation = ((img1_flat - mean_img1) * (img2_flat - mean_img2)).mean(dim=2, keepdim=True) / (std_img1 * std_img2)
        correlation = correlation.mean(dim=(0,1))

        return torch.abs((correlation) - 1).squeeze(dim=0)

    # @PerformanceAnalyzer.measure_time
    def forward(self, img, gt, cmap='YCbCr'):
        assert img.shape == gt.shape and img.shape[1] == 3, "should be the same shape and RGB-channels"
 
        B, _, height, width = img.shape
        
        if cmap == "YCbCr":
            device = img.device
            transform_matrix = torch.tensor([[0.299, 0.587, 0.114],
                                             [-0.14713, -0.288862, 0.436],
                                             [0.615, -0.51498, -0.10001]], device=device)
            
            img_yuv = (torch.matmul(img.permute(0, 2, 3, 1), transform_matrix.t())).permute(0, 3, 1, 2)
            gt_yuv = (torch.matmul(gt.permute(0, 2, 3, 1), transform_matrix.t())).permute(0, 3, 1, 2) 
            
            img_yuv = self.normalize_YCbCr(img_yuv)
            gt_yuv = self.normalize_YCbCr(gt_yuv)    
                
        elif cmap == "GRAY":
            img_gray = torch.mean(img, dim=1, keepdim=True) / 255.0
            gt_gray = torch.mean(gt, dim=1, keepdim=True) / 255.0
        
        window_h = height // self.blocks_ratio
        window_w = width // self.blocks_ratio
        
        stride_h = window_h
        stride_w = window_w
        
        if cmap == "YCbCr":
            block_img = img_yuv[:, 0, :, :].unfold(1, window_h, stride_h).unfold(2, window_w, stride_w).reshape(B, -1, window_h, window_w)
            block_gt = gt_yuv[:, 0, :, :].unfold(1, window_h, stride_h).unfold(2, window_w, stride_w).reshape(B, -1, window_h, window_w)
            random_indices = torch.randperm(block_img.shape[1])[:block_img.shape[1] // 2]
            block_img = block_img[:, random_indices, :, :]
            block_gt = block_gt[:, random_indices, :, :]
        elif cmap == "GRAY":
            block_img = img_gray[:, 0, :, :].unfold(1, window_h, stride_h).unfold(2, window_w, stride_w).reshape(B, -1, window_h, window_w)
            block_gt = gt_gray[:, 0, :, :].unfold(1, window_h, stride_h).unfold(2, window_w, stride_w).reshape(B, -1, window_h, window_w)
            random_indices = torch.randperm(block_img.shape[1])[:block_img.shape[1] // 2]
            block_img = block_img[:, random_indices, :, :]
            block_gt = block_gt[:, random_indices, :, :]
        
        loss = self.calculate_pearson_correlation(block_img, block_gt)

        return loss

    def is_strong_texture(self, glcm):
        energy_values = torch.sum(glcm, dim=(1, 2))  
        contrast_values = torch.var(glcm, dim=(1, 2))  
        entropy_values = -torch.sum(glcm * torch.log(glcm + 1e-10), dim=(1, 2))  

        combined_scores = energy_values + contrast_values + entropy_values

        return torch.argsort(combined_scores, descending=True)
    
    def normalize_YCbCr(self, x):
        x[:, 0, :, :] = (x[:, 0, :, :]) / 255.0
        return x


@LOSS_REGISTRY.register()
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss
