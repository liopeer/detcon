# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""DetCon/BYOL losses."""

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from detcon.utils import helpers


def manual_cross_entropy(labels, logits, weight):
  ce = - weight * jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
  return jnp.mean(ce)


def byol_nce_detcon(pred1, pred2, target1, target2,
                    pind1, pind2, tind1, tind2,
                    temperature=0.1, use_replicator_loss=True,
                    local_negatives=True):
  """Compute the NCE scores from pairs of predictions and targets.

  This implements the batched form of the loss described in
  Section 3.1, Equation 3 in https://arxiv.org/pdf/2103.10957.pdf.
  The number of masks (=number of indices in integer masks) is constant
  and set to 16. The latent projections and latent predictions have 256
  in correspondence with the BYOL paper.

  Args:
    pred1 (jnp.array): the prediction from first view (B, 16, 256).
    pred2 (jnp.array): the prediction from second view (B, 16, 256).
    target1 (jnp.array): the projection from first view (B, 16, 256).
    target2 (jnp.array): the projection from second view (B, 16, 256).
    pind1 (jnp.array): mask indices for first view's prediction (B, 16)
    pind2 (jnp.array): mask indices for second view's prediction.
    tind1 (jnp.array): mask indices for first view's projection.
    tind2 (jnp.array): mask indices for second view's projection.
    temperature (float): the temperature to use for the NCE loss.
    use_replicator_loss (bool): use cross-replica samples.
    local_negatives (bool): whether to include local negatives

  Returns:
    A single scalar loss for the XT-NCE objective.

  """
  batch_size = pred1.shape[0]
  num_rois = pred1.shape[1]
  feature_dim = pred1.shape[-1]
  infinity_proxy = 1e9  # Used for masks to proxy a very large number.

  def make_same_obj(ind_0, ind_1):
    same_obj = jnp.equal(ind_0.reshape([batch_size, num_rois, 1]),
                         ind_1.reshape([batch_size, 1, num_rois]))
    return jnp.expand_dims(same_obj.astype("float32"), axis=2)
  
  # these are simply arrays of 
  same_obj_aa = make_same_obj(pind1, tind1)
  same_obj_ab = make_same_obj(pind1, tind2)
  same_obj_ba = make_same_obj(pind2, tind1)
  same_obj_bb = make_same_obj(pind2, tind2)
  assert same_obj_aa.shape == (batch_size, num_rois, 1, num_rois), same_obj_aa.shape

  # L2 normalize the tensors to use for the cosine-similarity
  pred1 = helpers.l2_normalize(pred1, axis=-1)
  pred2 = helpers.l2_normalize(pred2, axis=-1)
  target1 = helpers.l2_normalize(target1, axis=-1)
  target2 = helpers.l2_normalize(target2, axis=-1)

  if jax.device_count() > 1 and use_replicator_loss:
    # Grab tensor across replicas and expand first dimension
    target1_large = jax.lax.all_gather(target1, axis_name="i")
    target2_large = jax.lax.all_gather(target2, axis_name="i")

    # Fold into batch dimension
    target1_large = target1_large.reshape(-1, num_rois, feature_dim)
    target2_large = target2_large.reshape(-1, num_rois, feature_dim)

    # Create the labels by using the current replica ID and offsetting.
    replica_id = jax.lax.axis_index("i")
    labels_idx = jnp.arange(batch_size) + replica_id * batch_size
    labels_idx = labels_idx.astype(jnp.int32)
    enlarged_batch_size = target1_large.shape[0]
    labels_local = hk.one_hot(labels_idx, enlarged_batch_size)
    labels_ext = hk.one_hot(labels_idx, enlarged_batch_size * 2)

  else:
    target1_large = target1
    target2_large = target2
    labels_local = hk.one_hot(jnp.arange(batch_size), batch_size) # same as np.eye(batch_size)
    labels_ext = hk.one_hot(jnp.arange(batch_size), batch_size * 2)

  labels_local = jnp.expand_dims(jnp.expand_dims(labels_local, axis=2), axis=1)
  labels_ext = jnp.expand_dims(jnp.expand_dims(labels_ext, axis=2), axis=1)

  assert labels_local.shape == (batch_size, 1, batch_size, 1), labels_local.shape
  assert labels_ext.shape == (batch_size, 1, batch_size*2, 1), labels_ext.shape

  # Do our matmuls and mask out appropriately.
  logits_aa = jnp.einsum("abk,uvk->abuv", pred1, target1_large) / temperature
  logits_bb = jnp.einsum("abk,uvk->abuv", pred2, target2_large) / temperature
  logits_ab = jnp.einsum("abk,uvk->abuv", pred1, target2_large) / temperature
  logits_ba = jnp.einsum("abk,uvk->abuv", pred2, target1_large) / temperature
  assert logits_aa.shape == (batch_size, num_rois, batch_size, num_rois), logits_aa.shape

  labels_aa = labels_local * same_obj_aa
  labels_ab = labels_local * same_obj_ab
  labels_ba = labels_local * same_obj_ba
  labels_bb = labels_local * same_obj_bb

  logits_aa = logits_aa - infinity_proxy * labels_local * same_obj_aa
  logits_bb = logits_bb - infinity_proxy * labels_local * same_obj_bb
  labels_aa = 0. * labels_aa
  labels_bb = 0. * labels_bb
  if not local_negatives:
    logits_aa = logits_aa - infinity_proxy * labels_local * (1 - same_obj_aa)
    logits_ab = logits_ab - infinity_proxy * labels_local * (1 - same_obj_ab)
    logits_ba = logits_ba - infinity_proxy * labels_local * (1 - same_obj_ba)
    logits_bb = logits_bb - infinity_proxy * labels_local * (1 - same_obj_bb)

  labels_abaa = jnp.concatenate([labels_ab, labels_aa], axis=2)
  labels_babb = jnp.concatenate([labels_ba, labels_bb], axis=2)

  labels_0 = jnp.reshape(labels_abaa, [batch_size, num_rois, -1])
  labels_1 = jnp.reshape(labels_babb, [batch_size, num_rois, -1])

  num_positives_0 = jnp.sum(labels_0, axis=-1, keepdims=True)
  num_positives_1 = jnp.sum(labels_1, axis=-1, keepdims=True)

  labels_0 = labels_0 / jnp.maximum(num_positives_0, 1)
  labels_1 = labels_1 / jnp.maximum(num_positives_1, 1)

  obj_area_0 = jnp.sum(make_same_obj(pind1, pind1), axis=[2, 3])
  obj_area_1 = jnp.sum(make_same_obj(pind2, pind2), axis=[2, 3])

  weights_0 = jnp.greater(num_positives_0[..., 0], 1e-3).astype("float32")
  weights_0 = weights_0 / obj_area_0
  weights_1 = jnp.greater(num_positives_1[..., 0], 1e-3).astype("float32")
  weights_1 = weights_1 / obj_area_1

  logits_abaa = jnp.concatenate([logits_ab, logits_aa], axis=2)
  logits_babb = jnp.concatenate([logits_ba, logits_bb], axis=2)

  logits_abaa = jnp.reshape(logits_abaa, [batch_size, num_rois, -1])
  logits_babb = jnp.reshape(logits_babb, [batch_size, num_rois, -1])

  loss_a = manual_cross_entropy(labels_0, logits_abaa, weights_0)
  loss_b = manual_cross_entropy(labels_1, logits_babb, weights_1)
  loss = loss_a + loss_b

  return loss

import torch
from torch import Tensor
from torch import distributed as torch_dist
from torch.nn import Module
import torch.nn.functional as F

class DetConLoss(Module):
    """Implementation of the DetCon loss. [0]_

    The inputs are two views of feature maps :math:`v_m` and :math:`v_{m'}'`, pooled over the regions
    of the segmentation mask. Those feature maps are first normalized to a norm of
    :math:`\\frac{1}{\\sqrt{\\tau}}`, where :math:`\\tau` is the temperature. The contrastive
    loss is then calculated as follows, where not only different images in the batch
    are considered as negatives, but also different regions of the same image:

    .. math::
        \\mathcal{L} = \\mathbb{E}_{(m, m')\\sim \\mathcal{M}}\\left[ - \\log \\frac{\\exp(v_m \\cdot v_{m'}')}{\\exp(v_m \\cdot v_{m'}') + \\sum_{n}\\exp (v_m \\cdot v_{m'}')} \\right]

    References:
        .. [0] DetCon https://arxiv.org/abs/2103.10957

    Attributes:
        temperature:
            The temperature :math:`\\tau` in the contrastive loss.
        gather_distributed:
            If True, the similarity matrix is gathered across all GPUs before the loss
            is calculated. Else, the loss is calculated on each GPU separately.
    """
    def __init__(self, temperature: float = 0.1, gather_distributed: bool = True):
        super().__init__()
        self.eps = 1e-8
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.eps = 1e-11

        if abs(self.temperature) < self.eps:
            raise ValueError(
                "Illegal temperature: abs({}) < 1e-8".format(self.temperature)
            )
        if self.gather_distributed and not torch_dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )
        
    def forward(
            self,
            pred_view0: Tensor,
            pred_view1: Tensor,
            target_view0: Tensor,
            target_view1: Tensor,
            mask_view0: Tensor,
            mask_view1: Tensor,
    ) -> Tensor:
        """Calculate the contrastive loss under the same mask in the same image.

        The tensor shapes and value ranges are given by variables B, M, D, N, where B is 
            the batch size, M is the sampled number of image masks / regions, D is the 
            embedding size and N is the total number of masks.

        Args:
            pred_view0: Mask-pooled output of the prediction branch for the first view,
                a float tensor of shape (B, M, D).
            pred_view1: Mask-pooled output of the prediction branch for the second view,
                a float tensor of shape (B, M, D).
            target_view0: Mask-pooled output of the target branch for the first view,
                a float tensor of shape (B, M, D).
            target_view1: Mask-pooled output of the target branch for the second view,
                a float tensor of shape (B, M, D).
            mask_view0: Indices corresponding to the sampled masks for the first view,
                an integer tensor of shape (B, M) with (possibly repeated) indices in the
                range [0, N).
            mask_view1: Indices corresponding to the sampled masks for the second view,
                an integer tensor of shape (B, M) with (possibly repeated) indices in the
                range [0, N).

        Returns:
            A scalar tensor containing the contrastive loss.
        """
        b, m, d = pred_view0.size()
        infinity_proxy = 1e9

        # normalize
        pred_view0 = F.normalize(pred_view0, p=2, dim=2)
        pred_view1 = F.normalize(pred_view1, p=2, dim=2)
        target_view0 = F.normalize(target_view0, p=2, dim=2)
        target_view1 = F.normalize(target_view1, p=2, dim=2)

        # gather distributed
        if not self.gather_distributed:
            target_view0_large = target_view0
            target_view1_large = target_view1
            labels_local = torch.eye(b, device=pred_view0.device)
            labels_ext = torch.cat([torch.eye(b, device=pred_view0.device), torch.zeros_like(labels_local)], dim=1)
        else:
            raise NotImplementedError("Gather distributed is not yet implemented.")
        
        labels_local = labels_local[:, None, :, None]
        labels_ext = labels_ext[:, None, :, None]
        assert labels_local.size() == (b, 1, b, 1)
        assert labels_ext.size() == (b, 1, 2*b, 1)

        # calculate similarity matrices
        logits_aa = torch.einsum("abk,uvk->abuv", pred_view0, target_view0_large) / self.temperature
        logits_bb = torch.einsum("abk,uvk->abuv", pred_view1, target_view1_large) / self.temperature
        logits_ab = torch.einsum("abk,uvk->abuv", pred_view0, target_view1_large) / self.temperature
        logits_ba = torch.einsum("abk,uvk->abuv", pred_view1, target_view0_large) / self.temperature
        assert logits_aa.size() == (b, m, b, m)

        # determine where the masks are the same
        same_mask_aa = _same_mask(mask_view0, mask_view0)
        same_mask_bb = _same_mask(mask_view1, mask_view1)
        same_mask_ab = _same_mask(mask_view0, mask_view1)
        same_mask_ba = _same_mask(mask_view1, mask_view0)
        assert same_mask_aa.size() == (b, m, 1, m), same_mask_aa.size()

        # remove similarities between the same masks
        labels_aa = labels_local * same_mask_aa
        labels_bb = labels_local * same_mask_bb
        labels_ab = labels_local * same_mask_ab
        labels_ba = labels_local * same_mask_ba

        logits_aa = logits_aa - infinity_proxy * labels_aa
        logits_bb = logits_bb - infinity_proxy * labels_bb
        labels_aa = 0. * labels_aa
        labels_bb = 0. * labels_bb

        labels_abaa = torch.cat([labels_ab, labels_aa], dim=2)
        labels_babb = torch.cat([labels_ba, labels_bb], dim=2)

        labels_0 = labels_abaa.view(b, m, -1)
        labels_1 = labels_babb.view(b, m, -1)

        num_positives_0 = torch.sum(labels_0, dim=-1, keepdim=True)
        num_positives_1 = torch.sum(labels_1, dim=-1, keepdim=True)

        labels_0 = labels_0 / torch.maximum(num_positives_0, torch.tensor(1))
        labels_1 = labels_1 / torch.maximum(num_positives_1, torch.tensor(1))

        obj_area_0 = torch.sum(_same_mask(mask_view0, mask_view0), dim=(2, 3))
        obj_area_1 = torch.sum(_same_mask(mask_view1, mask_view1), dim=(2, 3))

        weights_0 = torch.gt(num_positives_0[..., 0], 1e-3).float()
        weights_0 = weights_0 / obj_area_0
        weights_1 = torch.gt(num_positives_1[..., 0], 1e-3).float()
        weights_1 = weights_1 / obj_area_1

        logits_abaa = torch.cat([logits_ab, logits_aa], dim=2)
        logits_babb = torch.cat([logits_ba, logits_bb], dim=2)

        logits_abaa = logits_abaa.view(b, m, -1)
        logits_babb = logits_babb.view(b, m, -1)

        loss_a = torch_manual_cross_entropy(labels_0, logits_abaa, weights_0)
        loss_b = torch_manual_cross_entropy(labels_1, logits_babb, weights_1)
        loss = loss_a + loss_b
        return loss

def _same_mask(mask0: Tensor, mask1: Tensor) -> Tensor:
    return (mask0[:, :, None] == mask1[:, None, :]).float()[:, :, None, :]

def torch_manual_cross_entropy(labels: Tensor, logits: Tensor, weight: Tensor) -> Tensor:
    ce = - weight * torch.sum(labels * F.log_softmax(logits, dim=-1), dim=-1)
    return torch.mean(ce)


if __name__ == "__main__":
    bs, num_rois, feature_dim = 2, 2, 2

    key = jax.random.PRNGKey(42)
    pred1_ = jax.random.normal(key, (bs, num_rois, feature_dim))
    pred2_ = jax.random.normal(key, (bs, num_rois, feature_dim))
    target1_ = jax.random.normal(key, (bs, num_rois, feature_dim))
    target2_ = jax.random.normal(key, (bs, num_rois, feature_dim))
    pind1_ = jax.random.randint(key, (bs, num_rois), 0, num_rois)
    pind2_ = jax.random.randint(key, (bs, num_rois), 0, num_rois)
    tind1_ = pind1_.copy()
    tind2_ = pind2_.copy()

    print(pred1_)
    print(pred2_)
    print(target1_)
    print(target2_)
    print(pind1_)
    print(pind2_)
    print(tind1_)
    print(tind2_)

    key = jax.random.PRNGKey(41)
    pred1 = jax.random.normal(key, (bs, num_rois, feature_dim))
    pred2 = jax.random.normal(key, (bs, num_rois, feature_dim))
    target1 = jax.random.normal(key, (bs, num_rois, feature_dim))
    target2 = jax.random.normal(key, (bs, num_rois, feature_dim))
    pind1 = jax.random.randint(key, (bs, num_rois), 0, num_rois)
    pind2 = jax.random.randint(key, (bs, num_rois), 0, num_rois)
    tind1 = pind1.copy()
    tind2 = pind2.copy()

    print(pred1, pred2, target1, target2, pind1, pind2, tind1, tind2)

    pred1 = jnp.concatenate([pred1_, pred1], axis=0)
    pred2 = jnp.concatenate([pred2_, pred2], axis=0)
    target1 = jnp.concatenate([target1_, target1], axis=0)
    target2 = jnp.concatenate([target2_, target2], axis=0)
    pind1 = jnp.concatenate([pind1_, pind1], axis=0)
    pind2 = jnp.concatenate([pind2_, pind2], axis=0)
    tind1 = jnp.concatenate([tind1_, tind1], axis=0)
    tind2 = jnp.concatenate([tind2_, tind2], axis=0)

    loss1 = byol_nce_detcon(pred1, pred2, target1, target2, pind1, pind2, tind1, tind2)

    # torch version
    pred1 = torch.from_numpy(np.array(pred1)).to(torch.float32)
    pred2 = torch.from_numpy(np.array(pred2)).to(torch.float32)
    target1 = torch.from_numpy(np.array(target1)).to(torch.float32)
    target2 = torch.from_numpy(np.array(target2)).to(torch.float32)
    mask1 = torch.from_numpy(np.array(pind1)).to(torch.int64)
    mask2 = torch.from_numpy(np.array(pind2)).to(torch.int64)

    loss2 = DetConLoss(gather_distributed=False)(pred1, pred2, target1, target2, mask1, mask2)
    
    print(loss1, loss2)