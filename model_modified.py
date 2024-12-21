import copy
import torch
import numpy as np
import torch.nn as nn
from functools import partial
from torch.nn import functional as F
from croco.models.blocks import Block
from dust3r.model import AsymmetricCroCo3DStereo


class SpatialMemory():
    def __init__(self, norm_q, norm_k, norm_v, mem_dropout=None, 
                 long_mem_size=4000, work_mem_size=5, 
                 attn_thresh=5e-4, sim_thresh=0.95, 
                 save_attn=False, num_patches=None,
                 soft_prune_threshold=1e-3,              
                 low_priority_retention_frames=10):       
        self.norm_q = norm_q
        self.norm_k = norm_k
        self.norm_v = norm_v
        self.mem_dropout = mem_dropout
        self.attn_thresh = attn_thresh
        self.long_mem_size = long_mem_size
        self.work_mem_size = work_mem_size
        self.top_k = long_mem_size
        self.save_attn = save_attn
        self.sim_thresh = sim_thresh
        self.num_patches = num_patches
        self.soft_prune_threshold = soft_prune_threshold
        self.low_priority_retention_frames = low_priority_retention_frames

        self.init_mem()

    def init_mem(self):
        self.mem_k = None
        self.mem_v = None
        self.mem_c = None
        self.mem_count = None
        self.mem_attn = None
        self.mem_pts = None
        self.mem_imgs = None

        # low priority memory
        self.low_priority_mem_k = None
        self.low_priority_mem_v = None
        self.low_priority_mem_attn = None
        self.low_priority_mem_count = None
        self.low_priority_mem_pts = None
        self.low_priority_mem_imgs = None
        self.low_priority_age = None 

        self.lm = 0
        self.wm = 0
        if self.save_attn:
            self.attn_vis = None

    def add_mem_k(self, feat):
        if self.mem_k is None:
            self.mem_k = feat
        else:
            self.mem_k = torch.cat((self.mem_k, feat), dim=1)
        return self.mem_k
    
    def add_mem_v(self, feat):
        if self.mem_v is None:
            self.mem_v = feat
        else:
            self.mem_v = torch.cat((self.mem_v, feat), dim=1)
        return self.mem_v

    def add_mem_c(self, feat):
        if self.mem_c is None:
            self.mem_c = feat
        else:
            self.mem_c = torch.cat((self.mem_c, feat), dim=1)
        return self.mem_c
    
    def add_mem_pts(self, pts_cur):
        if pts_cur is not None:
            if self.mem_pts is None:
                self.mem_pts = pts_cur
            else:
                self.mem_pts = torch.cat((self.mem_pts, pts_cur), dim=1)
    
    def add_mem_img(self, img_cur):
        if img_cur is not None:
            if self.mem_imgs is None:
                self.mem_imgs = img_cur
            else:
                self.mem_imgs = torch.cat((self.mem_imgs, img_cur), dim=1)

    def add_mem(self, feat_k, feat_v, pts_cur=None, img_cur=None):  
        if self.num_patches is None:
            self.num_patches = feat_k.shape[1]
            
        if self.mem_count is None:
            self.mem_count = torch.zeros_like(feat_k[:, :, :1])
            self.mem_attn = torch.zeros_like(feat_k[:, :, :1])
        else:
            self.mem_count = torch.cat((self.mem_count, torch.zeros_like(feat_k[:, :, :1])), dim=1)
            self.mem_attn = torch.cat((self.mem_attn, torch.zeros_like(feat_k[:, :, :1])), dim=1)
        
        self.add_mem_k(feat_k)
        self.add_mem_v(feat_v)
        self.add_mem_pts(pts_cur)
        self.add_mem_img(img_cur)

    def check_sim(self, feat_k, thresh=0.7):
        if self.mem_k is None or thresh == 1.0:
            return False
        wmem_size = self.wm * self.num_patches
        wm = self.mem_k[:, -wmem_size:].reshape(self.mem_k.shape[0], -1, self.num_patches, self.mem_k.shape[-1])

        feat_k_norm = F.normalize(feat_k, p=2, dim=-1)
        wm_norm = F.normalize(wm, p=2, dim=-1)

        corr = torch.einsum('bpc,btpc->btp', feat_k_norm, wm_norm)
        mean_corr = torch.mean(corr, dim=-1)
        if mean_corr.max() > thresh:
            print('Similarity detected:', mean_corr.max())
            return True
        return False

    def add_mem_check(self, feat_k, feat_v, pts_cur=None, img_cur=None):
        if self.num_patches is None:
            self.num_patches = feat_k.shape[1]

        if self.check_sim(feat_k, thresh=self.sim_thresh):
            return
        
        self.add_mem(feat_k, feat_v, pts_cur, img_cur)
        self.wm += 1

        if self.wm > self.work_mem_size:
            self.wm -= 1
            if self.long_mem_size == 0:
                # hard prune
                self.mem_k = self.mem_k[:, self.num_patches:]
                self.mem_v = self.mem_v[:, self.num_patches:]
                self.mem_count = self.mem_count[:, self.num_patches:]
                self.mem_attn = self.mem_attn[:, self.num_patches:]
                print('Memory pruned (no long mem):', self.mem_k.shape)
            else:
                self.lm += self.num_patches
        
        if self.lm > self.long_mem_size:
            self.multi_stage_memory_prune()  # multi_stage prune
            self.lm = self.top_k - self.wm * self.num_patches

    def memory_read(self, feat, res=True):
        affinity = torch.einsum('bpc,bxc->bpx', self.norm_q(feat), self.norm_k(self.mem_k.reshape(self.mem_k.shape[0], -1, self.mem_k.shape[-1])))
        affinity /= torch.sqrt(torch.tensor(feat.shape[-1]).float())
        
        if self.mem_c is not None:
            affinity = affinity * self.mem_c.view(self.mem_c.shape[0], 1, -1)  
        
        attn = torch.softmax(affinity, dim=-1)
        if self.save_attn:
            if self.attn_vis is None:
                self.attn_vis = attn.reshape(-1)
            else:
                self.attn_vis = torch.cat((self.attn_vis, attn.reshape(-1)), dim=0)
        if self.mem_dropout is not None:
            attn = self.mem_dropout(attn)
        
        if self.attn_thresh > 0:
            attn[attn<self.attn_thresh] = 0
            attn = attn / attn.sum(dim=-1, keepdim=True)
        
        out = torch.einsum('bpx,bxc->bpc', attn, self.norm_v(self.mem_v.reshape(self.mem_v.shape[0], -1, self.mem_v.shape[-1])))
        
        if res:
            out = out + feat
        
        total_attn = torch.sum(attn, dim=-2)
        self.mem_attn += total_attn[..., None]
        
        return out
    
    # Modify Part

    def multi_stage_memory_prune(self):
        self.update_low_priority_age()
        weights = self.mem_attn / (self.mem_count + 1e-8)
        soft_prune_mask = (weights < self.soft_prune_threshold).squeeze(-1)
        if torch.any(soft_prune_mask):
            self.move_to_low_priority(soft_prune_mask)
        self.recover_from_low_priority()
        self.hard_delete_from_low_priority()
        self.final_topk_prune()



    def move_to_low_priority(self, mask):
        print('Moved to low priority')
        def gather_selected(tensor, mask):
            return tensor[mask].view(tensor.shape[0], -1, tensor.shape[-1])

        sel_k = gather_selected(self.mem_k, mask)
        sel_v = gather_selected(self.mem_v, mask)
        sel_attn = gather_selected(self.mem_attn, mask)
        sel_count = gather_selected(self.mem_count, mask)
        
        if self.mem_pts is not None:
            sel_pts = gather_selected(self.mem_pts, mask)
        else:
            sel_pts = None
        
        if self.mem_imgs is not None:
            sel_imgs = gather_selected(self.mem_imgs, mask)
        else:
            sel_imgs = None
        self.add_to_low_priority(sel_k, sel_v, sel_attn, sel_count, sel_pts, sel_imgs)

        keep_mask = ~mask
        self.filter_main_memory(keep_mask)



    def add_to_low_priority(self, k, v, attn, count, pts, imgs):
        def concat_or_init(orig, new):
            if orig is None:
                return new
            if new is None:
                return orig
            if orig.shape[-1] != new.shape[-1]:
                raise ValueError
            return torch.cat((orig, new), dim=1)
        
        self.low_priority_mem_k = concat_or_init(self.low_priority_mem_k, k)
        self.low_priority_mem_v = concat_or_init(self.low_priority_mem_v, v)
        self.low_priority_mem_attn = concat_or_init(self.low_priority_mem_attn, attn)
        self.low_priority_mem_count = concat_or_init(self.low_priority_mem_count, count)
        self.low_priority_age = concat_or_init(self.low_priority_age, torch.zeros_like(count))
        
        if pts is not None:
            self.low_priority_mem_pts = concat_or_init(self.low_priority_mem_pts, pts)
        if imgs is not None:
            self.low_priority_mem_imgs = concat_or_init(self.low_priority_mem_imgs, imgs)



    def filter_main_memory(self, keep_mask):
        def apply_mask(tensor, mask):
            return tensor[mask].reshape(tensor.shape[0], -1, *tensor.shape[2:])

        self.mem_k = apply_mask(self.mem_k, keep_mask)
        self.mem_v = apply_mask(self.mem_v, keep_mask)
        self.mem_attn = apply_mask(self.mem_attn, keep_mask)
        self.mem_count = apply_mask(self.mem_count, keep_mask)
        if self.mem_pts is not None:
            self.mem_pts = self.mem_pts[keep_mask].reshape(self.mem_pts.shape[0], -1, self.mem_pts.shape[2], self.mem_pts.shape[3])
        if self.mem_imgs is not None:
            self.mem_imgs = self.mem_imgs[keep_mask].reshape(self.mem_imgs.shape[0], -1, self.mem_imgs.shape[2], self.mem_imgs.shape[3])

    def update_low_priority_age(self):
        if self.low_priority_age is not None:
            self.low_priority_age = self.low_priority_age + 1



    def hard_delete_from_low_priority(self):
        print('Start hard prune')
        if self.low_priority_mem_k is None:
            return
        weights_lp = self.low_priority_mem_attn / (self.low_priority_mem_count + 1e-8)
        hard_delete_mask = (self.low_priority_age > self.low_priority_retention_frames) & (weights_lp < self.soft_prune_threshold)

        if torch.any(hard_delete_mask):
            keep_mask = ~hard_delete_mask.squeeze(-1)
            self.filter_low_priority_memory(keep_mask)


    def filter_low_priority_memory(self, keep_mask):
        
        # print(f"keep_mask shape {keep_mask.shape}")
        if keep_mask.dim() != 2:
            raise ValueError

        if keep_mask.shape[1] != self.low_priority_mem_k.shape[1]:
            raise ValueError

        def apply_keep_mask(tensor, keep_mask):
            return tensor[keep_mask].view(tensor.shape[0], -1, *tensor.shape[2:])

        self.low_priority_mem_k = apply_keep_mask(self.low_priority_mem_k, keep_mask)
        self.low_priority_mem_v = apply_keep_mask(self.low_priority_mem_v, keep_mask)
        self.low_priority_mem_attn = apply_keep_mask(self.low_priority_mem_attn, keep_mask)
        self.low_priority_mem_count = apply_keep_mask(self.low_priority_mem_count, keep_mask)
        self.low_priority_age = apply_keep_mask(self.low_priority_age, keep_mask)

        if self.low_priority_mem_pts is not None:
            self.low_priority_mem_pts = apply_keep_mask(self.low_priority_mem_pts, keep_mask)
        if self.low_priority_mem_imgs is not None:
            self.low_priority_mem_imgs = apply_keep_mask(self.low_priority_mem_imgs, keep_mask)


    def final_topk_prune(self):
        if self.mem_k is None:
            return
        weights = self.mem_attn / (self.mem_count+1e-8)
        weights[self.mem_count<self.work_mem_size+5] = 1e8

        num_mem_b = self.mem_k.shape[1]

        top_k_values, top_k_indices = torch.topk(weights.squeeze(-1), self.top_k, dim=1)
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, self.mem_k.size(-1))

        self.mem_k = torch.gather(self.mem_k, 1, top_k_indices_expanded)
        self.mem_v = torch.gather(self.mem_v, 1, top_k_indices_expanded)
        self.mem_attn = torch.gather(self.mem_attn, 1, top_k_indices.unsqueeze(-1))
        self.mem_count = torch.gather(self.mem_count, 1, top_k_indices.unsqueeze(-1))

        if self.mem_pts is not None:
            pts_idx_expanded = top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.mem_pts.shape[2], self.mem_pts.shape[3])
            self.mem_pts = torch.gather(self.mem_pts, 1, pts_idx_expanded)

        if self.mem_imgs is not None:
            img_idx_expanded = top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.mem_imgs.shape[2], self.mem_imgs.shape[3])
            self.mem_imgs = torch.gather(self.mem_imgs, 1, img_idx_expanded)

        num_mem_a = self.mem_k.shape[1]
        print('Memory pruned:', num_mem_b, '->', num_mem_a)


    def recover_from_low_priority(self):
        if self.low_priority_mem_k is None:
            print("Low-priority memory empty")
            return

        weights_lp = self.low_priority_mem_attn / (self.low_priority_mem_count + 1e-8)
        recovery_mask = weights_lp > self.soft_prune_threshold

        if torch.any(recovery_mask):
            print(f"Recovering {recovery_mask.sum().item()}")
            self.recover_to_main_memory(recovery_mask)

        self.filter_low_priority_memory(~recovery_mask.squeeze(-1))



    def recover_to_main_memory(self, recovery_mask):

        B, T_lp, _ = recovery_mask.shape
        C = self.low_priority_mem_k.shape[-1]

        recovered_k = []
        recovered_v = []
        recovered_attn = []
        recovered_count = []
        recovered_pts = []
        recovered_imgs = []

        for b in range(B):
            mask_b = recovery_mask[b, :, 0]
            if torch.any(mask_b):
                k_b = self.low_priority_mem_k[b, mask_b, :]
                v_b = self.low_priority_mem_v[b, mask_b, :] 
                attn_b = self.low_priority_mem_attn[b, mask_b, :]  
                count_b = self.low_priority_mem_count[b, mask_b, :]  
                
                recovered_k.append(k_b)
                recovered_v.append(v_b)
                recovered_attn.append(attn_b)
                recovered_count.append(count_b)
                
                if self.low_priority_mem_pts is not None:
                    pts_b = self.low_priority_mem_pts[b, mask_b, :, :]  
                    recovered_pts.append(pts_b)
                if self.low_priority_mem_imgs is not None:
                    imgs_b = self.low_priority_mem_imgs[b, mask_b, :, :]  
                    recovered_imgs.append(imgs_b)

        if recovered_k:
            recovered_k = torch.stack(recovered_k, dim=0) 
            self.mem_k = torch.cat((self.mem_k, recovered_k), dim=1)
        if recovered_v:
            recovered_v = torch.stack(recovered_v, dim=0)
            self.mem_v = torch.cat((self.mem_v, recovered_v), dim=1)
        if recovered_attn:
            recovered_attn = torch.stack(recovered_attn, dim=0)
            self.mem_attn = torch.cat((self.mem_attn, recovered_attn), dim=1)
        if recovered_count:
            recovered_count = torch.stack(recovered_count, dim=0)
            self.mem_count = torch.cat((self.mem_count, recovered_count), dim=1)
        if recovered_pts:
            recovered_pts = torch.stack(recovered_pts, dim=0)
            self.mem_pts = torch.cat((self.mem_pts, recovered_pts), dim=1)
        if recovered_imgs:
            recovered_imgs = torch.stack(recovered_imgs, dim=0)
            self.mem_imgs = torch.cat((self.mem_imgs, recovered_imgs), dim=1)



class Spann3R(nn.Module):
    def __init__(self, dus3r_name="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", 
                 use_feat=False, mem_pos_enc=False, memory_dropout=0.15):
        super(Spann3R, self).__init__()
        # config
        self.use_feat = use_feat
        self.mem_pos_enc = mem_pos_enc

        # DUSt3R
        self.dust3r = AsymmetricCroCo3DStereo.from_pretrained(dus3r_name, landscape_only=True)

        # Memory encoder
        self.set_memory_encoder(enc_embed_dim=768 if use_feat else 1024, memory_dropout=memory_dropout) 
        self.set_attn_head()

    def set_memory_encoder(self, enc_depth=6, enc_embed_dim=1024, out_dim=1024, enc_num_heads=16, 
                           mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                           memory_dropout=0.15):
        
        self.value_encoder = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True, 
                  norm_layer=norm_layer, rope=self.dust3r.rope if self.mem_pos_enc else None)
            for i in range(enc_depth)])
        
        self.value_norm = norm_layer(enc_embed_dim)
        self.value_out = nn.Linear(enc_embed_dim, out_dim)
        
        if not self.use_feat:
            self.pos_patch_embed = copy.deepcopy(self.dust3r.patch_embed)
            self.pos_patch_embed.load_state_dict(self.dust3r.patch_embed.state_dict())
        
        self.norm_q = nn.LayerNorm(1024)
        self.norm_k = nn.LayerNorm(1024)
        self.norm_v = nn.LayerNorm(1024)
        self.mem_dropout = nn.Dropout(memory_dropout)
        
    def set_attn_head(self, enc_embed_dim=1024+768, out_dim=1024):
        self.attn_head_1 = nn.Sequential(
            nn.Linear(enc_embed_dim, enc_embed_dim),
            nn.GELU(),
            nn.Linear(enc_embed_dim, out_dim)
        )
        
        self.attn_head_2 = nn.Sequential(
            nn.Linear(enc_embed_dim, enc_embed_dim),
            nn.GELU(),
            nn.Linear(enc_embed_dim, out_dim)
        )

    def encode_image(self, view):
        img = view['img']
        B = img.shape[0]
        im_shape = view.get('true_shape', torch.tensor(img.shape[-2:])[None].repeat(B, 1))
        
        out, pos, _ = self.dust3r._encode_image(img, im_shape)
        
        return out, pos, im_shape
    
    def encode_image_pairs(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']

        B = img1.shape[0]

        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        
        out, pos, _ = self.dust3r._encode_image(torch.cat((img1, img2), dim=0),
                                                torch.cat((shape1, shape2), dim=0))
        out, out2 = out.chunk(2, dim=0)
        pos, pos2 = pos.chunk(2, dim=0)
        
        return out, out2, pos, pos2, shape1, shape2
    
    def encode_frames(self, view1, view2, feat1, feat2, pos1, pos2, shape1, shape2):
        if feat1 is None:
            feat1, feat2, pos1, pos2, shape1, shape2 = self.encode_image_pairs(view1, view2)
        else:
            feat1, pos1, shape1 = feat2, pos2, shape2
            feat2, pos2, shape2 = self.encode_image(view2)
        return feat1, feat2, pos1, pos2, shape1, shape2
    
    def encode_feat_key(self, feat1, feat2, num=1):
        feat = torch.cat((feat1, feat2), dim=-1)
        feat_k = getattr(self, f'attn_head_{num}')(feat)
        return feat_k
    
    def encode_value(self, x, pos):
        for block in self.value_encoder:
            x = block(x, pos)
        x = self.value_norm(x)
        x = self.value_out(x)
        return x
    
    def encode_cur_value(self, res1, dec1, pos1, shape1):
        if self.use_feat:
            cur_v = self.encode_value(dec1[-1], pos1)     
        else:
            out, pos_v = self.pos_patch_embed(res1['pts3d'].permute(0, 3, 1, 2), true_shape=shape1)
            cur_v = self.encode_value(out, pos_v)
        return cur_v
    
    def decode(self, feat1, pos1, feat2, pos2):
        dec1, dec2 = self.dust3r._decoder(feat1, pos1, feat2, pos2)
        return dec1, dec2
    
    def downstream_head(self, dec, true_shape, num=1):
        with torch.cuda.amp.autocast(enabled=False):
            res = self.dust3r._downstream_head(num, [tok.float() for tok in dec], true_shape)
        return res
    
    def forward(self, frames, return_memory=False):
        if self.training:
            sp_mem = SpatialMemory(self.norm_q, self.norm_k, self.norm_v, mem_dropout=self.mem_dropout, attn_thresh=0)
        else:
            sp_mem = SpatialMemory(self.norm_q, self.norm_k, self.norm_v)

        feat1, feat2, pos1, pos2, shape1, shape2 = None, None, None, None, None, None
        feat_k1, feat_k2 = None, None

        preds = None
        preds_all = []

        for i in range(len(frames)):
            if i == len(frames)-1:
                break
            view1 = frames[i]
            view2 = frames[(i+1)]

            feat1, feat2, pos1, pos2, shape1, shape2 = self.encode_frames(view1, view2, feat1, feat2, pos1, pos2, shape1, shape2)

            if feat_k2 is not None:
                feat_fuse = sp_mem.memory_read(feat_k2, res=True)
            else:
                feat_fuse = feat1
            
            dec1, dec2 = self.decode(feat_fuse, pos1, feat2, pos2)
            
            feat_k1 = self.encode_feat_key(feat1, dec1[-1], 1)
            feat_k2 = self.encode_feat_key(feat2, dec2[-1], 2)

            with torch.cuda.amp.autocast(enabled=False):
                res1 = self.downstream_head(dec1, shape1, 1)
                res2 = self.downstream_head(dec2, shape2, 2)
            
            cur_v = self.encode_cur_value(res1, dec1, pos1, shape1)

            if self.training:
                sp_mem.add_mem(feat_k1, cur_v+feat_k1)
            else:
                sp_mem.add_mem_check(feat_k1, cur_v+feat_k1)
            
            res2['pts3d_in_other_view'] = res2.pop('pts3d')  
             
            if preds is None:
                preds = [res1]
                preds_all = [(res1, res2)]
            else:
                res1['pts3d_in_other_view'] = res1.pop('pts3d')
                preds.append(res1)
                preds_all.append((res1, res2))
                
        preds.append(res2)

        if return_memory:
            return preds, preds_all, sp_mem
        
        return preds, preds_all
