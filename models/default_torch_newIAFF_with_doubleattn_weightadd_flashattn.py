import torch
import torch.nn.functional as F
import open3d.ml.torch as ml3d
import numpy as np
import torch.nn as nn
import copy
import math
from models.ASCC import ContinuousConv as ASCC
import time

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class IAFF(nn.Module):
    def __init__(self, channels=32, inter_channels=64, conv_type='cconv'):
        super(IAFF, self).__init__()
        self.filter_extent = torch.tensor(np.float32(1.5 * 6 * 0.025))
        def window_poly6(r_sqr):
            return torch.clamp((1 - r_sqr) ** 3, 0, 1)

        def Conv(name, activation=None, conv_type='cconv', **kwargs):
            if conv_type == 'cconv':
                conv_fn = ml3d.layers.ContinuousConv
            elif conv_type == 'ascc':
                conv_fn = ASCC
            conv = conv_fn(kernel_size=[4, 4, 4],
                           activation=activation,
                           align_corners=True,
                           normalize=False,
                           window_function=window_poly6,
                           radius_search_ignore_query_points=True,
                           **kwargs)
            return conv

        self.cconv1 = Conv(name="conv1",
             in_channels=channels*2,
             filters=inter_channels,
             activation=None,
             conv_type=conv_type)
        self.batchNorm1 = nn.BatchNorm1d(inter_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.cconv2 = Conv(name="conv2",
                           in_channels=inter_channels,
                           filters=channels,
                           activation=None,
                           conv_type=conv_type)
        self.batchNorm2 = nn.BatchNorm1d(channels)

        self.cconv3 = Conv(name="conv3",
             in_channels=channels,
             filters=inter_channels,
             activation=None,
             conv_type=conv_type)
        self.batchNorm3 = nn.BatchNorm1d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.cconv4 = Conv(name="conv4",
                           in_channels=inter_channels,
                           filters=channels,
                           activation=None,
                           conv_type=conv_type)
        self.batchNorm4 = nn.BatchNorm1d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, pos):
        # print(x.shape,y.shape,pos.shape) #torch.Size([2333, 32]) torch.Size([2333, 32]) torch.Size([2333, 3])
        xa = torch.cat((x, y), -1) 

        xl = self.cconv1(xa, pos, pos, self.filter_extent)
        xl = self.batchNorm1(xl)
        xl = self.relu1(xl)
        xl = self.cconv2(xl, pos, pos, self.filter_extent)
        xl = self.batchNorm2(xl)
        wei1 = self.sigmoid(xl)
        xo = 2 * x * wei1 + 2 * y * (1 - wei1)

        xo = self.cconv3(xo, pos, pos, self.filter_extent)
        xo = self.batchNorm3(xo)
        xo = self.relu2(xo)
        xo = self.cconv4(xo, pos, pos, self.filter_extent)
        xo = self.batchNorm4(xo)
        wei2 = self.sigmoid(xo)
        xo = 2 * x * wei2 + 2 * y * (1 - wei2)
        # print(xo.shape)
        return xo




import torch
import torch.nn as nn
import math
from flash_attn import flash_attn_func
import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_func

class FlashSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads
        self.batch_first = batch_first
        
        # 线性投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = dropout
    
    def forward(self, query, key=None, value=None, key_padding_mask=None, need_weights=False):
        # 自注意力场景：query = key = value
        if key is None and value is None:
            key = value = query
        
        # 确保 batch_first 兼容性
        if not self.batch_first:  # (seq_len, batch_size, embed_dim) -> (batch_size, seq_len, embed_dim)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        batch_size, seqlen_q, _ = query.shape
        _, seqlen_k, _ = key.shape
        
        # 线性投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 重塑为 FlashAttention 所需的四维张量
        q = q.view(batch_size, seqlen_q, self.num_heads, self.head_size)
        k = k.view(batch_size, seqlen_k, self.num_heads, self.head_size)
        v = v.view(batch_size, seqlen_k, self.num_heads, self.head_size)
        
        # 转换为半精度（如果需要）
        if q.dtype == torch.float32:
            q = q.half()
            k = k.half()
            v = v.half()
        
        # 应用 FlashAttention
        # 注意：FlashAttention 不直接支持 key_padding_mask，需要预处理
        attn_output = flash_attn_func(
            q, k, v, 
            dropout_p=self.dropout if self.training else 0.0,
            causal=False  # 非因果注意力（用于编码器）
        )
        
        # 重塑回原始维度
        attn_output = attn_output.reshape(batch_size, seqlen_q, self.embed_dim)
        if attn_output.dtype != torch.float32:
            attn_output = attn_output.float()
        # 最终投影
        output = self.out_proj(attn_output)
        
        # 恢复 batch_first 格式
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        # 返回兼容格式
        if need_weights:
            return output, None  # FlashAttention 不返回注意力权重
        else:
            return output
        
class AFF(nn.Module):
    def __init__(self, channels=64, inter_channels=128, conv_type='cconv', num_heads=4, dropout=0.1):
        super(AFF, self).__init__()
        self.filter_extent = torch.tensor(np.float32(1.5 * 6 * 0.025))
        
        # 窗口函数
        def window_poly6(r_sqr):
            return torch.clamp((1 - r_sqr) ** 3, 0, 1)
        
        # 卷积工厂函数
        def Conv(name, in_channels, filters, activation=None, conv_type='cconv'):
            if conv_type == 'cconv':
                conv_fn = ml3d.layers.ContinuousConv
            elif conv_type == 'ascc':
                conv_fn = ASCC
            return conv_fn(
                kernel_size=[4,4,4],
                activation=activation,
                align_corners=True,
                normalize=False,
                window_function=window_poly6,
                radius_search_ignore_query_points=True,
                in_channels=in_channels,
                filters=filters
            )
        
        # ---------------------- 双分支结构 ----------------------
        # 分支1：处理特征 x
        self.x_branch = nn.Sequential(
            Conv("x_conv1", channels, inter_channels, activation=None, conv_type=conv_type),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True)
        )
        
        # 分支2：处理特征 y
        self.y_branch = nn.Sequential(
            Conv("y_conv1", channels, inter_channels, activation=None, conv_type=conv_type),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True)
        )
        
        # ---------------------- Transformer 注意力 ----------------------
        self.num_heads = num_heads
        if inter_channels == 3:
            self.num_heads = 1  # 避免head维度为0
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=inter_channels,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True
        )

        
        self.norm_attn = nn.LayerNorm(inter_channels)
        self.dropout_attn = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(inter_channels, inter_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(inter_channels * 4, inter_channels),
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(inter_channels)
        
        # ---------------------- 融合输出层 ----------------------
        # 融合方式可选：拼接后卷积 / 直接相加
        self.fusion_type = "add"  # 或 "add"
        if self.fusion_type == "concat":
            self.fusion_conv = Conv(
                "fusion_conv", 
                in_channels=inter_channels*2, 
                filters=channels,  # 恢复原始通道数
                activation=None, 
                conv_type=conv_type
            )
        elif self.fusion_type == "add":
            self.fusion_conv = Conv(
                "fusion_conv", 
                in_channels=inter_channels, 
                filters=channels,  # 恢复原始通道数
                activation=None, 
                conv_type=conv_type
            )
        
        self.fusion_bn = nn.BatchNorm1d(channels)
        self.fusion_relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.self_flash_attn = FlashSelfAttention(
            embed_dim=inter_channels,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True
        )
    def forward(self, x, y, pos):
        # print(x.shape,y.shape,pos.shape)
        # ---------------------- 双分支特征提取 ----------------------
        # 分支1：处理 x
        x_feat = self.x_branch[0](x, pos, pos, self.filter_extent)  # Conv
        x_feat = self.x_branch[1](x_feat)  # BatchNorm
        x_feat = self.x_branch[2](x_feat)  # ReLU
        
        # 分支2：处理 y
        y_feat = self.y_branch[0](y, pos, pos, self.filter_extent)  # Conv
        y_feat = self.y_branch[1](y_feat)  # BatchNorm
        y_feat = self.y_branch[2](y_feat)  # ReLU
        
        # ---------------------- Transformer 注意力（单分支处理示例，可改为双分支交互）----------------------
        # 示例：对 x 分支特征应用 Transformer（可根据需求改为对 y 或拼接后特征处理）
        def process_with_transformer(feat):
            # 重塑为 (batch_size, seq_len, features)
            if feat.dim() == 2:  # 输入为 [N, C]，添加 batch 维度
                feat = feat.unsqueeze(0)  # [1, N, C]
            seq_len, embed_dim = feat.shape[1], feat.shape[2]
            
            # 位置编码
            pos_encoding = self.positional_encoding(pos, embed_dim)  # [1, N, C]
            feat_with_pos = feat + pos_encoding  # 融合位置信息
            
            # Transformer 自注意力
            # attn_output, _ = self.self_attn(
            #     feat_with_pos, feat_with_pos, feat_with_pos
            # )
            attn_output=self.self_flash_attn(feat_with_pos)
            

            feat = feat + self.dropout_attn(attn_output)
            feat = self.norm_attn(feat)
            
            # 前馈网络
            ffn_output = self.ffn(feat)
            feat = feat + ffn_output
            feat = self.norm_ffn(feat)
            
            # 恢复维度：[1, N, C] -> [N, C]
            return feat.squeeze(0) if feat.dim() == 3 else feat
        
        x_feat = process_with_transformer(x_feat)
        y_feat = process_with_transformer(y_feat)  # 可选：对 y 分支也应用 Transformer
        
        # ---------------------- 特征融合 ----------------------
        if self.fusion_type == "concat":
            # 拼接后卷积融合
            fused_feat = torch.cat([x_feat, y_feat], dim=1)  # [N, 2*C]
            fused_feat = self.fusion_conv(fused_feat, pos, pos, self.filter_extent)  # 降维
        elif self.fusion_type == "add":
            # 直接相加融合（需确保 x_feat 和 y_feat 维度一致）
            fused_feat = x_feat + y_feat
            # fused_feat = self.fusion_conv(fused_feat, pos, pos, self.filter_extent)
        # 归一化和激活（可选）
        # fused_feat = self.fusion_bn(fused_feat)
        # fused_feat = self.fusion_relu(fused_feat)
        wei2 = self.sigmoid(fused_feat)  # 权重生成
        fused_output = 2 * x * wei2 + 2 * y * (1 - wei2)  # 等价于直接输出
        return fused_output

    def positional_encoding(self, pos, embed_dim):
        """基于点云坐标创建位置编码"""
        # 使用正弦余弦编码
        device = pos.device
        batch_size = 1
        
        # 创建编码矩阵
        pe = torch.zeros(batch_size, pos.shape[0], embed_dim, device=device)
        
        # 使用点云坐标的归一化值作为位置编码的基础
        pos_normalized = pos - pos.mean(0, keepdim=True)
        pos_normalized = pos_normalized / (pos_normalized.std(0, keepdim=True) + 1e-8)
        
        # 对每个维度应用正弦余弦编码
        div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device) * 
                             (-math.log(10000.0) / embed_dim))
        
        for i in range(embed_dim // 2):
            pe[0, :, 2*i] = torch.sin(pos_normalized[:, 0] * div_term[i])
            pe[0, :, 2*i+1] = torch.cos(pos_normalized[:, 0] * div_term[i])
            
            if embed_dim<=3:
                continue

            if embed_dim > 2*i+2:
                pe[0, :, 2*i+2] = torch.sin(pos_normalized[:, 1] * div_term[i])
                pe[0, :, 2*i+3] = torch.cos(pos_normalized[:, 1] * div_term[i])
                
            if embed_dim > 2*i+4:
                pe[0, :, 2*i+4] = torch.sin(pos_normalized[:, 2] * div_term[i])
                pe[0, :, 2*i+5] = torch.cos(pos_normalized[:, 2] * div_term[i])
                
        return pe


import torch
import torch.nn as nn
import math

class IAFF(nn.Module):
    def __init__(self, channels=32, inter_channels=64, conv_type='cconv', num_heads=4, dropout=0.1):
        super(IAFF, self).__init__()
        self.filter_extent = torch.tensor(np.float32(1.5 * 6 * 0.025))
        
        # 窗口函数
        def window_poly6(r_sqr):
            return torch.clamp((1 - r_sqr) ** 3, 0, 1)
        
        # 卷积工厂函数
        def Conv(name, in_channels, filters, activation=None, conv_type='cconv'):
            if conv_type == 'cconv':
                conv_fn = ml3d.layers.ContinuousConv
            elif conv_type == 'ascc':
                conv_fn = ASCC
            return conv_fn(
                kernel_size=[4,4,4],
                activation=activation,
                align_corners=True,
                normalize=False,
                window_function=window_poly6,
                radius_search_ignore_query_points=True,
                in_channels=in_channels,
                filters=filters
            )
        
        # ---------------------- 双分支结构（第一阶段）----------------------
        # 分支1：处理特征 x（第一阶段）
        self.x_branch1 = nn.Sequential(
            Conv("x_conv1", channels, inter_channels, conv_type=conv_type),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            Conv("x_conv2", inter_channels, channels, conv_type=conv_type),
            nn.BatchNorm1d(channels)
        )
        # 分支2：处理特征 y（第一阶段）
        self.y_branch1 = nn.Sequential(
            Conv("y_conv1", channels, inter_channels, conv_type=conv_type),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            Conv("y_conv2", inter_channels, channels, conv_type=conv_type),
            nn.BatchNorm1d(channels)
        )
        
        # ---------------------- Transformer 注意力（第一阶段融合后）----------------------
        self.self_attn = nn.MultiheadAttention(
            embed_dim=channels,  # 第一阶段输出通道为 channels
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_attn = nn.LayerNorm(channels)
        self.dropout_attn = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(channels)
        
        # ---------------------- 双分支结构（第二阶段）----------------------
        # 分支1：处理特征 x（第二阶段）
        self.x_branch2 = nn.Sequential(
            Conv("x_conv3", channels, inter_channels, conv_type=conv_type),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            Conv("x_conv4", inter_channels, channels, conv_type=conv_type),
            nn.BatchNorm1d(channels)
        )
        
        # 分支2：处理特征 y（第二阶段）----------------------
        self.y_branch2 = nn.Sequential(
            Conv("y_conv3", channels, inter_channels, conv_type=conv_type),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            Conv("y_conv4", inter_channels, channels, conv_type=conv_type),
            nn.BatchNorm1d(channels)
        )
        self.sigmoid = nn.Sigmoid()
        self.self_flash_attn = FlashSelfAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
    def forward(self, x, y, pos):
        # ---------------------- 第一阶段：双分支独立处理 ----------------------
        # 分支1：x 处理
        x_branch1 = self.x_branch1[0](x, pos, pos, self.filter_extent)  # Conv1
        x_branch1 = self.x_branch1[1](x_branch1)  # BN1
        x_branch1 = self.x_branch1[2](x_branch1)  # ReLU1
        x_branch1 = self.x_branch1[3](x_branch1, pos, pos, self.filter_extent)  # Conv2
        x_branch1 = self.x_branch1[4](x_branch1)  # BN2
        
        # 分支2：y 处理
        y_branch1 = self.y_branch1[0](y, pos, pos, self.filter_extent)  # Conv1
        y_branch1 = self.y_branch1[1](y_branch1)  # BN1
        y_branch1 = self.y_branch1[2](y_branch1)  # ReLU1
        y_branch1 = self.y_branch1[3](y_branch1, pos, pos, self.filter_extent)  # Conv2
        y_branch1 = self.y_branch1[4](y_branch1)  # BN2
        
        
        # ---------------------- 第一阶段融合 + Transformer 注意力 ----------------------
        # 加权融合（保留原逻辑）
        # wei1 = self.sigmoid(x_branch1 + y_branch1)  # 简单相加生成权重（可改为卷积生成）
        # fused_stage1 = 2 * x * wei1 + 2 * y * (1 - wei1)
        fused_stage1=x_branch1+y_branch1
        # 应用 Transformer 注意力
        fused_stage1 = self.apply_transformer(fused_stage1, pos)
        
        # ---------------------- 第二阶段：双分支独立处理 ----------------------
        x_branch2 = self.x_branch2[0](fused_stage1, pos, pos, self.filter_extent)  # Conv3
        x_branch2 = self.x_branch2[1](x_branch2)  # BN3
        x_branch2 = self.x_branch2[2](x_branch2)  # ReLU2
        x_branch2 = self.x_branch2[3](x_branch2, pos, pos, self.filter_extent)  # Conv4
        x_branch2 = self.x_branch2[4](x_branch2)  # BN4
        
        y_branch2 = self.y_branch2[0](fused_stage1, pos, pos, self.filter_extent)  # Conv3
        y_branch2 = self.y_branch2[1](y_branch2)  # BN3
        y_branch2 = self.y_branch2[2](y_branch2)  # ReLU2
        y_branch2 = self.y_branch2[3](y_branch2, pos, pos, self.filter_extent)  # Conv4
        y_branch2 = self.y_branch2[4](y_branch2)  # BN4
        
        # ---------------------- 第二阶段融合 ----------------------
        wei2 = self.sigmoid(x_branch2 + y_branch2)  # 权重生成
        fused_output = 2 * x * wei2 + 2 * y * (1 - wei2)  # 等价于直接输出
        # fused_output=x_branch2+y_branch2
        return fused_output

    def apply_transformer(self, feat, pos):
        """对特征应用 Transformer 注意力"""
        # 重塑为 (batch_size, seq_len, features)
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)  # 添加 batch 维度 [1, N, C]
        seq_len, embed_dim = feat.shape[1], feat.shape[2]
        
        # 位置编码
        pos_encoding = self.positional_encoding(pos, embed_dim)  # [1, N, C]
        feat_with_pos = feat + pos_encoding
        
        # 自注意力
        # attn_output, _ = self.self_attn(feat_with_pos, feat_with_pos, feat_with_pos)
        attn_output=self.self_flash_attn(feat_with_pos)
        feat = feat + self.dropout_attn(attn_output)
        feat = self.norm_attn(feat)
        
        # 前馈网络
        ffn_output = self.ffn(feat)
        feat = feat + ffn_output
        feat = self.norm_ffn(feat)
        
        # 恢复维度
        return feat.squeeze(0) if feat.dim() == 3 else feat

    def positional_encoding(self, pos, embed_dim):
        """基于点云坐标的正弦余弦位置编码"""
        device = pos.device
        pos = pos.unsqueeze(0)  # [1, N, 3]
        pos_normalized = (pos - pos.mean(dim=1, keepdim=True)) / (pos.std(dim=1, keepdim=True) + 1e-8)
        
        pe = torch.zeros(1, pos.shape[1], embed_dim, device=device)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device) * (-math.log(10000.0) / embed_dim))
        
        for i in range(embed_dim // 2):
            pe[0, :, 2*i] = torch.sin(pos_normalized[:, :, 0] * div_term[i])
            pe[0, :, 2*i+1] = torch.cos(pos_normalized[:, :, 0] * div_term[i])
            if 2*i+2 < embed_dim:
                pe[0, :, 2*i+2] = torch.sin(pos_normalized[:, :, 1] * div_term[i])
                pe[0, :, 2*i+3] = torch.cos(pos_normalized[:, :, 1] * div_term[i])
            if 2*i+4 < embed_dim:
                pe[0, :, 2*i+4] = torch.sin(pos_normalized[:, :, 2] * div_term[i])
                pe[0, :, 2*i+5] = torch.cos(pos_normalized[:, :, 2] * div_term[i])
        return pe

class MyParticleNetwork(torch.nn.Module):
    def __init__(
            self,
            kernel_size=[4, 4, 4],
            radius_scale=1.5,
            coordinate_mapping='ball_to_cube_volume_preserving',
            interpolation='linear',
            use_window=True,
            particle_radius=0.025,
            timestep=1 / 50,
            gravity=(0, -9.81, 0),
            other_feats_channels=0,
    ):
        super().__init__()
        self.layer_channels = [32, 64, 128, 64, 3]
        self.kernel_size = kernel_size
        self.radius_scale = radius_scale
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        self.particle_radius = particle_radius
        self.filter_extent = np.float32(self.radius_scale * 6 *
                                        self.particle_radius)
        self.timestep = timestep
        gravity = torch.FloatTensor(gravity)
        self.register_buffer('gravity', gravity)


        def window_poly6(r_sqr):
            return torch.clamp((1 - r_sqr)**3, 0, 1)

        def Conv(name, activation=None, conv_type='cconv', **kwargs):
            if conv_type == 'cconv':
                conv_fn = ml3d.layers.ContinuousConv
            elif conv_type == 'ascc':
                conv_fn = ASCC
            window_fn = None
            if self.use_window == True:
                window_fn = window_poly6

            conv = conv_fn(kernel_size=self.kernel_size,
                           activation=activation,
                           align_corners=True,
                           interpolation=self.interpolation,
                           coordinate_mapping=self.coordinate_mapping,
                           normalize=False,
                           window_function=window_fn,
                           radius_search_ignore_query_points=True,
                           **kwargs)
            if conv_type == 'cconv':
                self._all_convs_cconv.append((name, conv))
            elif conv_type == 'ascc':
                self._all_convs_ascc.append((name, conv))
            return conv

    #cconv
        self.aff_cconv = IAFF(channels=32, inter_channels=64, conv_type='cconv')
        self._all_convs_cconv = []
        self.conv0_fluid_cconv = Conv(name="cconv0_fluid",
                                in_channels=4 + other_feats_channels,
                                filters=self.layer_channels[0],
                                activation=None,
                                conv_type='cconv')
        self.conv0_obstacle_cconv = Conv(name="cconv0_obstacle",
                                   in_channels=3,
                                   filters=self.layer_channels[0],
                                   activation=None,
                                   conv_type='cconv')
        self.dense0_fluid_cconv = torch.nn.Linear(in_features=4 +
                                            other_feats_channels,
                                            out_features=self.layer_channels[0])
        torch.nn.init.xavier_uniform_(self.dense0_fluid_cconv.weight)
        torch.nn.init.zeros_(self.dense0_fluid_cconv.bias)

        self.convs_cconv = []
        self.denses_cconv = []
        for i in range(1, len(self.layer_channels)):
            in_ch = self.layer_channels[i - 1]
            if i == 1:
                in_ch = 64
            out_ch = self.layer_channels[i]
            dense_cconv = torch.nn.Linear(in_features=in_ch, out_features=out_ch)
            torch.nn.init.xavier_uniform_(dense_cconv.weight)
            torch.nn.init.zeros_(dense_cconv.bias)
            setattr(self, 'dense_cconv{0}'.format(i), dense_cconv)
            conv = Conv(name='cconv{0}'.format(i),
                        in_channels=in_ch,
                        filters=out_ch,
                        activation=None,
                        conv_type='cconv')
            setattr(self, 'cconv{0}'.format(i), conv)
            self.denses_cconv.append(dense_cconv)
            self.convs_cconv.append(conv)

    #ASCC
        self.aff_ascc = IAFF(channels=32, inter_channels=64, conv_type='ascc')
        self._all_convs_ascc = []
        self.conv0_fluid_ascc = Conv(name="ascc0_fluid",
                                in_channels=4 + other_feats_channels,
                                filters=self.layer_channels[0],
                                activation=None,
                                conv_type='ascc')
        self.conv0_obstacle_ascc = Conv(name="ascc0_obstacle",
                                   in_channels=3,
                                   filters=self.layer_channels[0],
                                   activation=None,
                                   conv_type='ascc')
        self.dense0_fluid_ascc = torch.nn.Linear(in_features=4 +
                                                        other_feats_channels,
                                            out_features=self.layer_channels[0])
        torch.nn.init.xavier_uniform_(self.dense0_fluid_ascc.weight)
        torch.nn.init.zeros_(self.dense0_fluid_ascc.bias)

        self.convs_ascc = []
        self.denses_ascc = []
        for i in range(1, len(self.layer_channels)):
            in_ch = self.layer_channels[i - 1]
            if i == 1:
                in_ch = 64
            out_ch = self.layer_channels[i]
            dense_ascc = torch.nn.Linear(in_features=in_ch, out_features=out_ch)
            torch.nn.init.xavier_uniform_(dense_ascc.weight)
            torch.nn.init.zeros_(dense_ascc.bias)
            setattr(self, 'dense_ascc{0}'.format(i), dense_ascc)
            conv_ascc = Conv(name='ascc{0}'.format(i),
                        in_channels=in_ch,
                        filters=out_ch,
                        activation=None,
                        conv_type='ascc')
            setattr(self, 'ascc{0}'.format(i), conv_ascc)
            self.denses_ascc.append(dense_ascc)
            self.convs_ascc.append(conv_ascc)
    # AFF
        self.affs = []
        self.aff0 = AFF(channels=self.layer_channels[0]*2, inter_channels=self.layer_channels[0]*2, conv_type='cconv')
        for i in range(1, len(self.layer_channels)):
            ch = self.layer_channels[i]
            aff = AFF(channels=ch, inter_channels=ch, conv_type='cconv')
            setattr(self, 'aff'+str(i), aff)
            self.affs.append(aff)
        self.resAff = AFF(channels=64, inter_channels=64, conv_type='cconv')
    def integrate_pos_vel(self, pos1, vel1):
        """Apply gravity and integrate position and velocity"""
        dt = self.timestep
        vel2 = vel1 + dt * self.gravity
        pos2 = pos1 + dt * (vel2 + vel1) / 2
        return pos2, vel2

    def compute_new_pos_vel(self, pos1, vel1, pos2, vel2, pos_correction):
        """Apply the correction
        pos1,vel1 are the positions and velocities from the previous timestep
        pos2,vel2 are the positions after applying gravity
        """
        dt = self.timestep
        pos = pos2 + pos_correction
        # vel = 2 * (pos - pos1) / dt - vel1
        vel = (pos - pos1) / dt
        return pos, vel

    def compute_correction(self,
                           pos,
                           vel,
                           other_feats,
                           box,
                           box_feats,
                           fixed_radius_search_hash_table=None):
        """Expects that the pos and vel has already been updated with gravity and velocity"""

        # compute the extent of the filters (the diameter)
        filter_extent = torch.tensor(self.filter_extent)
        fluid_feats = [torch.ones_like(pos[:, 0:1]), vel]
        if not other_feats is None:
            fluid_feats.append(other_feats)
        fluid_feats = torch.cat(fluid_feats, axis=-1)
    #     print('fluid_feats',fluid_feats.shape)
    # #cconvfluid_feats torch.Size([2372, 4])
    #     fluid_feats torch.Size([2339, 4])
    #     fluid_feats torch.Size([2339, 4])
    #     fluid_feats torch.Size([2440, 4])
        # 经过第一层网络
        self.ans_conv0_fluid_cconv = self.conv0_fluid_cconv(fluid_feats, pos, pos,
                                                filter_extent)
        self.ans_dense0_fluid_cconv = self.dense0_fluid_cconv(fluid_feats)
        self.ans_conv0_obstacle_cconv = self.conv0_obstacle_cconv(box_feats, box, pos,
                                                      filter_extent)
        
        self.ans_dense0_fluid_cconv = self.dense0_fluid_cconv(fluid_feats)

        #IAFF
        self.hybrid_aff_cconv = self.aff_cconv(self.ans_conv0_fluid_cconv, self.ans_conv0_obstacle_cconv, pos)

        feats_cconv = torch.cat([
            self.hybrid_aff_cconv, self.ans_dense0_fluid_cconv
        ], axis=-1)

    # ascc
        self.ans_conv0_fluid_ascc = self.conv0_fluid_ascc(fluid_feats, pos, pos,
                                                          filter_extent)
        self.ans_dense0_fluid_ascc = self.dense0_fluid_ascc(fluid_feats)
        self.ans_conv0_obstacle_ascc = self.conv0_obstacle_ascc(box_feats, box, pos,
                                                                filter_extent)
        self.ans_dense0_fluid_ascc = self.dense0_fluid_ascc(fluid_feats)

        # IAFF
        self.hybrid_aff_ascc = self.aff_ascc(self.ans_conv0_fluid_ascc, self.ans_conv0_obstacle_ascc, pos)

        feats_ascc = torch.cat([
            self.hybrid_aff_ascc, self.ans_dense0_fluid_ascc
        ], axis=-1)

        feats_select = self.aff0(feats_cconv, feats_ascc, pos)

        self.ans_convs = [feats_select]

        for conv_cconv, dense_cconv, conv_ascc, dense_ascc, aff in zip(self.convs_cconv, self.denses_cconv, self.convs_ascc, self.denses_ascc, self.affs):
            inp_feats = F.relu(self.ans_convs[-1])
            #cconv
            ans_conv_cconv = conv_cconv(inp_feats, pos, pos, filter_extent)
            ans_dense_cconv = dense_cconv(inp_feats)
            ans_cconv = ans_conv_cconv + ans_dense_cconv
            #ascc
            ans_conv_ascc = conv_ascc(inp_feats, pos, pos, filter_extent)
            ans_dense_ascc = dense_ascc(inp_feats)
            ans_ascc = ans_conv_ascc + ans_dense_ascc
            #aff
            ans_select = aff(ans_cconv, ans_ascc, pos)
            #ResAFF
            if len(self.ans_convs) == 3 and ans_dense_cconv.shape[-1] == self.ans_convs[-2].shape[-1]:
                ans_select = self.resAff(ans_select, self.ans_convs[-2], pos)
            self.ans_convs.append(ans_select)

        # compute the number of fluid neighbors.
        # this info is used in the loss function during training.
        self.num_fluid_neighbors = ml3d.ops.reduce_subarrays_sum(
            torch.ones_like(self.conv0_fluid_cconv.nns.neighbors_index,
                            dtype=torch.float32),
            self.conv0_fluid_cconv.nns.neighbors_row_splits)

        # scale to better match the scale of the output distribution
        self.pos_correction = (1.0 / 128) * self.ans_convs[-1]

        return self.pos_correction

    def forward(self, inputs, fixed_radius_search_hash_table=None):
        """computes 1 simulation timestep
        inputs: list or tuple with (pos,vel,feats,box,box_feats)
          pos and vel are the positions and velocities of the fluid particles.
          feats is reserved for passing additional features, use None here.
          box are the positions of the static particles and box_feats are the
          normals of the static particles.
        """
        #print(inputs)
        pos, vel, feats, box, box_feats = inputs
        
        # import torch
        # torch.save(inputs, r"/media/data1/cyucopy/code/AAAI-simple/models/inputs.pt")
        # import os
        # print( os.path.exists(r"/media/data1/cyucopy/code/AAAI-simple/models/inputs.pt"))
        
        # torch.save(pos,r"/media/data1/cyucopy/code/AAAI-simple/models/out.pt")
        
        pos2, vel2 = self.integrate_pos_vel(pos, vel)
        pos_correction = self.compute_correction(
            pos2, vel2, feats, box, box_feats, fixed_radius_search_hash_table)
        
        pos2_corrected, vel2_corrected = self.compute_new_pos_vel(
            pos, vel, pos2, vel2, pos_correction)

        return pos2_corrected, vel2_corrected