"""
结构感知修正器 V5 - 最终正确版本

核心机制：
1. 学习：产品名称 → 规格型号的结构模式（特定产品类型的专属模式）
2. 推理：OCR识别 + 结构模式 → 修正错误

示例：
产品名称: "储物盒支撑钢丝A"
         ↓ 学习到的结构模式
规格结构: [字母, 字母, 数字, 数字, 数字, 数字, 数字, 数字]
         ↓
产品名称: "储物盒支撑钢丝B"
         ↓ 推断规格结构
推断结构: [字母, 字母, 数字, 数字, 数字, 数字, 数字, 数字]
         ↓
OCR识别: DL381O97 (第6位误识别为O)
         ↓ 应用结构模式
修正:    DL381097 (第6位应为数字)
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter
import math
import numpy as np
import pandas as pd

# ==================== 结构模式学习 ====================
class StructurePatternLearner:
    def __init__(self, max_len=128, name_similarity_threshold=0.8):
        self.max_len = max_len
        self.name_similarity_threshold = name_similarity_threshold  # 相似度阈值
        self.name_to_structure = defaultdict(list)
        self.name_to_pattern = {}
        
        # 新增：构建名称到标准名称的映射
        self.name_to_canonical = {}  # 标准化名称映射
        self.canonical_names = set()  # 所有标准名称

    def analyze_structure(self, spec_str):
        """分析规格字符串的结构模式"""
        spec = str(spec_str).strip().upper()
        spec = re.sub(r'\s+', '', spec)
        
        # 分析每个字符的类型
        structure = []
        for c in spec:
            if re.match(r'[0-9]', c):
                structure.append('D')  # Digit
            elif re.match(r'[A-Z]', c):
                structure.append('A')  # Alpha
            elif re.match(r'[\u4e00-\u9fff]', c):
                structure.append('C')  # Chinese
            else:
                structure.append('S')  # Symbol
        
        return structure
    
    def normalize_name(self, name_str):
        """标准化产品名称，处理常见的识别错误"""
        name = str(name_str).strip().upper()
        name = re.sub(r'\s+', '', name)
        
        # 移除末尾的型号后缀
        pattern = r'[A-Z0-9]$'
        if re.search(pattern, name):
            name = re.sub(pattern, '', name)
        
        
        # 2. 移除特殊字符和标点
        name = re.sub(r'[^A-Z0-9\u4e00-\u9fff]', '', name)
        
        return name
    
    def get_name_key(self, name_str):
        """
        提取产品名称的类别标识
        去除末尾的型号后缀（如A/B/C/1/2/3）
        """
        name = str(name_str).strip().upper()
        name = re.sub(r'\s+', '', name)
        
        # 移除末尾的单个字母或数字（型号标识）
        pattern = r'[A-Z0-9]$'
        if re.search(pattern, name):
            name = re.sub(pattern, '', name)
        
        return name    
    
    def find_canonical_name(self, name_str):
        """查找最匹配的标准名称（模糊匹配）"""
        normalized = self.normalize_name(name_str)
        
        # 如果标准化后的名称已存在，直接返回
        if normalized in self.canonical_names:
            return normalized
        
        # 计算与所有标准名称的相似度
        best_match = None
        best_similarity = 0.0
        
        for canonical_name in self.canonical_names:
            similarity = self._calculate_similarity(normalized, canonical_name)
            if similarity > best_similarity and similarity >= self.name_similarity_threshold:
                best_similarity = similarity
                best_match = canonical_name
        
        # 如果找到匹配的，返回标准名称；否则返回标准化后的名称
        if best_match is not None:
            return best_match
        else:
            # 添加为新的标准名称
            self.canonical_names.add(normalized)
            return normalized
    
    def _calculate_similarity(self, name1, name2):
        """计算两个名称的相似度（使用编辑距离）"""
        if name1 == name2:
            return 1.0
        
        # 计算编辑距离
        distance = self._edit_distance(name1, name2)
        max_len = max(len(name1), len(name2))
        
        # 相似度 = 1 - 编辑距离 / 最大长度
        similarity = 1.0 - distance / max_len if max_len > 0 else 0.0
        return similarity
    
    def _edit_distance(self, s1, s2):
        """计算编辑距离（Levenshtein距离）"""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _get_most_common_pattern(self, structures):
        """获取最常见的结构模式"""
        # 转换为字符串以便统计
        structure_strs = [''.join(s) for s in structures]
        
        # 找到最常见的
        counter = Counter(structure_strs)
        most_common = counter.most_common(1)[0][0]
        
        # 转换回列表
        pattern = [c for c in most_common]
        
        return pattern
    
    def learn_from_data(self, df, name_col, spec_col):
        """从数据中学习结构模式（改进版）"""
        for r in df.itertuples():
            name = getattr(r, name_col)
            spec = getattr(r, spec_col)
            
            if pd.isna(name) or pd.isna(spec):
                continue
            
            # 标准化名称并查找标准名称
            canonical_name = self.find_canonical_name(name_str=name)
            
            # 分析规格结构
            structure = self.analyze_structure(spec)
            
            # 记录到标准名称下
            self.name_to_structure[canonical_name].append(structure)
        
        # 为每个标准名称生成结构模式
        for canonical_name, structures in self.name_to_structure.items():
            if len(structures) == 0:
                continue
            
            pattern = self._get_most_common_pattern(structures)
            self.name_to_pattern[canonical_name] = pattern
    
    def infer_structure(self, name_str):
        """根据产品名称推断规格结构（改进版）"""
        # 查找标准名称（模糊匹配）
        canonical_name = self.find_canonical_name(name_str)
        
        if canonical_name in self.name_to_pattern:
            pattern = self.name_to_pattern[canonical_name]
            # 计算置信度
            all_structures = self.name_to_structure[canonical_name]
            pattern_str = ''.join(pattern)
            count = sum(1 for s in all_structures if ''.join(s) == pattern_str)
            confidence = count / len(all_structures)
        else:
            # 未知产品类型，返回空模式
            pattern = []
            confidence = 0.0
        
        # 填充到max_len
        if len(pattern) < self.max_len:
            pattern = pattern + ['X'] * (self.max_len - len(pattern))
        else:
            pattern = pattern[:self.max_len]
        
        return pattern, confidence
    
    def encode_structure(self, pattern):
        """将结构模式编码为数字ID"""
        structure_map = {'D': 0, 'A': 1, 'C': 2, 'S': 3, 'X': 4}
        return [structure_map.get(c, 4) for c in pattern]

# ==================== 结构类型编码 ====================
STRUCTURE_TYPES = {
    'EXPECT_DIGIT': 0,    # 预期数字
    'EXPECT_ALPHA': 1,    # 预期字母
    'EXPECT_CHINESE': 2,  # 预期中文
    'EXPECT_SYMBOL': 3,  # 预期符号
    'EXPECT_ANY': 4      # 任意（未知/填充）
}

NUM_STRUCTURE_TYPES = len(STRUCTURE_TYPES)

def encode_structure_pattern(pattern):
    """编码结构模式"""
    mapping = {'D': 0, 'A': 1, 'C': 2, 'S': 3, 'X': 4}
    return [mapping.get(c, 4) for c in pattern]

# ==================== 位置编码 ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ==================== Transformer编码器 ====================
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        key_padding_mask = (mask == 1) if mask is not None else None
        return self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)


class StructureAwareCorrectorV5(nn.Module):
    """
    结构感知修正器 V5 
    核心机制：
    1. 学习产品名称到规格结构模式的映射
    2. 根据产品名称推断规格的预期结构
    3. 对比OCR识别结果与预期结构，修正错误位置
    """
    
    def __init__(self, vocab_size_ocr=200, vocab_size_name=500,
                 d_model=256, n_heads=8, n_layers=4, dropout=0.2):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size_ocr = vocab_size_ocr
        self.vocab_size_name = vocab_size_name
        
        # 1. OCR编码器
        self.ocr_encoder = TransformerEncoder(vocab_size_ocr, d_model, n_heads, n_layers, dropout)
        
        # 2. 产品名称编码器
        self.name_encoder = TransformerEncoder(vocab_size_name, d_model, n_heads//2, n_layers//2, dropout)
        
        # 3. 结构模式编码器
        self.structure_encoder = nn.Embedding(NUM_STRUCTURE_TYPES, d_model)
        
        # 4. 交叉注意力：OCR关注产品名称
        self.ocr_name_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_ocr_name = nn.LayerNorm(d_model)
        
        # 5. 结构一致性检查模块
        self.structure_consistency = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # OCR特征 + 结构特征
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2)  # 0=一致, 1=不一致（需要修正）
        )
        
        # 6. 一致性信息编码器
        self.consistency_encoder = nn.Linear(2, d_model)
        
        # 7. 字符修正头（输入维度改为 d_model * 2）
        self.char_correction_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # OCR特征 + 一致性特征
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size_ocr)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, ocr_tokens, name_tokens, structure_pattern,
                ocr_mask=None, name_mask=None):
        """
        Args:
            ocr_tokens: (B, L_ocr) - OCR识别结果
            name_tokens: (B, L_name) - 产品名称
            structure_pattern: (B, L_ocr) - 从产品名称推断的结构模式
            ocr_mask: (B, L_ocr) - OCR padding mask
            name_mask: (B, L_name) - 名称padding mask
        
        Returns:
            correction_logits: (B, L_ocr, vocab_size_ocr) - 字符修正logits
            consistency_logits: (B, L_ocr, 2) - 结构一致性logits
        """
        # 1. 编码OCR和产品名称
        ocr_features = self.ocr_encoder(ocr_tokens, ocr_mask)
        name_features = self.name_encoder(name_tokens, name_mask)

        # 2. 编码结构模式
        structure_embeds = self.structure_encoder(structure_pattern)

        # 3. OCR关注产品名称
        name_key_padding_mask = (name_mask == 1) if name_mask is not None else None
        ocr_with_name, _ = self.ocr_name_attn(
            ocr_features, name_features, name_features,
            key_padding_mask=name_key_padding_mask
        )
        ocr_with_name = self.norm_ocr_name(ocr_with_name)

        # 4. 结构一致性检查
        # 注意：OCR tokens包含<SOS>和<EOS>，但结构模式直接对应实际字符
        # 我们需要跳过<SOS>(位置0)，将OCR位置1+与结构模式对齐

        # 跳过<SOS>，从位置1开始
        ocr_without_sos = ocr_with_name[:, 1:, :]  # (B, L_ocr-1, d_model)

        # 跳过结构模式中的padding标记（X）
        structure_mask = (structure_pattern != 4)  # (B, L_pattern), 4是'X'

        # 对齐：取两者的最小长度
        min_len = min(ocr_without_sos.size(1), structure_embeds.size(1))
        ocr_aligned = ocr_without_sos[:, :min_len, :]
        structure_aligned = structure_embeds[:, :min_len, :]

        # 5. 结构一致性检查
        combined_features = torch.cat([ocr_aligned, structure_aligned], dim=-1)  # (B, min_len, 2*d_model)
        consistency_logits = self.structure_consistency(combined_features)  # (B, min_len, 2)

        # 6. 将一致性信息编码为特征
        consistency_prob = F.softmax(consistency_logits, dim=-1)  # (B, min_len, 2)
        consistency_embeds = self.consistency_encoder(consistency_prob)  # (B, min_len, d_model)

        # 7. 对齐一致性特征与原始OCR特征
        # consistency_logits 只覆盖 min_len（跳过了<SOS>），需要补齐到 L_ocr
        # 在位置0（<SOS>）填充零，在 min_len 之后的padding位置也填充零
        batch_size = ocr_with_name.size(0)
        ocr_len = ocr_with_name.size(1)

        # 创建完整的一致性特征张量
        full_consistency_embeds = torch.zeros(
            batch_size, ocr_len, self.d_model,
            device=ocr_with_name.device, dtype=ocr_with_name.dtype
        )

        # 将对齐后的一致性特征填充到正确位置（跳过<SOS>，从位置1开始）
        full_consistency_embeds[:, 1:1+min_len, :] = consistency_embeds

        # 8. 融合OCR特征和一致性特征进行字符修正
        ocr_with_consistency = torch.cat(
            [ocr_with_name, full_consistency_embeds], 
            dim=-1
        )  # (B, L_ocr, 2*d_model)

        correction_logits = self.char_correction_head(ocr_with_consistency)  # (B, L_ocr, vocab_size_ocr)

        return correction_logits, consistency_logits


# ==================== 结构感知修正器 V5 ====================
class StructureAwareCorrectorV5(nn.Module):
    """
    结构感知修正器 V5 - 融合一致性信息
    
    核心机制：
    1. 学习产品名称到规格结构模式的映射
    2. 根据产品名称推断规格的预期结构
    3. 对比OCR识别结果与预期结构，生成一致性信息
    4. 融合一致性信息进行字符修正
    """
    
    def __init__(self, vocab_size_ocr=200, vocab_size_name=500,
                 d_model=256, n_heads=8, n_layers=4, dropout=0.2):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size_ocr = vocab_size_ocr
        self.vocab_size_name = vocab_size_name
        
        # 1. OCR编码器
        self.ocr_encoder = TransformerEncoder(vocab_size_ocr, d_model, n_heads, n_layers, dropout)
        
        # 2. 产品名称编码器
        self.name_encoder = TransformerEncoder(vocab_size_name, d_model, n_heads//2, n_layers//2, dropout)
        
        # 3. 结构模式编码器
        self.structure_encoder = nn.Embedding(NUM_STRUCTURE_TYPES, d_model)
        
        # 4. 交叉注意力：OCR关注产品名称
        self.ocr_name_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_ocr_name = nn.LayerNorm(d_model)
        
        # 5. 结构一致性检查模块
        self.structure_consistency = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # OCR特征 + 结构特征
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2)  # 0=一致, 1=不一致（需要修正）
        )
        
        # 6. 字符修正头（输入维度改为 d_model + 1）
        self.char_correction_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model),  # OCR特征 + 一致性概率
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size_ocr)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, ocr_tokens, name_tokens, structure_pattern,
                ocr_mask=None, name_mask=None):
        """
        Args:
            ocr_tokens: (B, L_ocr) - OCR识别结果
            name_tokens: (B, L_name) - 产品名称
            structure_pattern: (B, L_ocr) - 从产品名称推断的结构模式
            ocr_mask: (B, L_ocr) - OCR padding mask
            name_mask: (B, L_name) - 名称padding mask
        
        Returns:
            correction_logits: (B, L_ocr, vocab_size_ocr) - 字符修正logits
            consistency_logits: (B, L_ocr-1, 2) - 结构一致性logits（跳过<SOS>）
        """
        # 1. 编码OCR和产品名称
        ocr_features = self.ocr_encoder(ocr_tokens, ocr_mask)
        name_features = self.name_encoder(name_tokens, name_mask)

        # 2. 编码结构模式
        structure_embeds = self.structure_encoder(structure_pattern)

        # 3. OCR关注产品名称
        name_key_padding_mask = (name_mask == 1) if name_mask is not None else None
        ocr_with_name, _ = self.ocr_name_attn(
            ocr_features, name_features, name_features,
            key_padding_mask=name_key_padding_mask
        )
        ocr_with_name = self.norm_ocr_name(ocr_with_name)

        # 4. 结构一致性检查
        # 跳过<SOS>，从位置1开始
        ocr_without_sos = ocr_with_name[:, 1:, :]  # (B, L_ocr-1, d_model)

        # 对齐：取两者的最小长度
        min_len = min(ocr_without_sos.size(1), structure_embeds.size(1))
        ocr_aligned = ocr_without_sos[:, :min_len, :]
        structure_aligned = structure_embeds[:, :min_len, :]

        # 5. 结构一致性检查
        combined_features = torch.cat([ocr_aligned, structure_aligned], dim=-1)  # (B, min_len, 2*d_model)
        consistency_logits = self.structure_consistency(combined_features)  # (B, min_len, 2)

        # 6. 将一致性概率补齐到完整长度
        # consistency_logits 只覆盖 min_len，需要补齐到 L_ocr
        batch_size = ocr_with_name.size(0)
        ocr_len = ocr_with_name.size(1)

        # 创建完整的一致性logits张量（包括<SOS>位置）
        full_consistency_logits = torch.zeros(
            batch_size, ocr_len, 2,
            device=ocr_with_name.device, dtype=ocr_with_name.dtype
        )

        # 将对齐后的一致性logits填充到正确位置（跳过<SOS>，从位置1开始）
        full_consistency_logits[:, 1:1+min_len, :] = consistency_logits

        # 7. 将一致性概率与OCR特征拼接
        consistency_prob = F.softmax(full_consistency_logits, dim=-1)[..., 1:2]  # (B, L_ocr, 1) 取不一致的概率
        ocr_with_consistency = torch.cat([ocr_with_name, consistency_prob], dim=-1)  # (B, L_ocr, d_model+1)

        # 8. 使用融合后的特征进行修正预测
        correction_logits = self.char_correction_head(ocr_with_consistency)  # (B, L_ocr, vocab_size_ocr)

        return correction_logits, consistency_logits
    

# ==================== 损失函数 ====================
def structure_aware_corrector_loss_V5(
    correction_logits, target_tokens,
    consistency_logits, consistency_mask,
    pad_id=0,
    alpha_correction=1.0,
    alpha_consistency=0.3
):
    """
    损失函数：
    1. 字符修正损失：预测正确的字符
    2. 结构一致性损失：识别需要修正的位置
    """
    # 1. 字符修正损失
    mask = (target_tokens != pad_id)
    if mask.sum() > 0:
        correction_loss = F.cross_entropy(
            correction_logits[mask],
            target_tokens[mask],
            reduction='mean'
        )
    else:
        correction_loss = torch.tensor(0.0, device=correction_logits.device)
    
    # 2. 结构一致性损失
    if consistency_mask is not None:
        min_len = min(consistency_logits.size(1), consistency_mask.size(1))
        consistency_logits = consistency_logits[:, :min_len, :]
        consistency_mask = consistency_mask[:, :min_len]
        
        if (consistency_mask != -1).sum() > 0:
            valid_mask = (consistency_mask != -1)
            consistency_loss = F.cross_entropy(
                consistency_logits[valid_mask],
                consistency_mask[valid_mask],
                reduction='mean'
            )
        else:
            consistency_loss = torch.tensor(0.0, device=consistency_logits.device)
    else:
        consistency_loss = torch.tensor(0.0, device=consistency_logits.device)
    
    # 总损失
    total_loss = (
        alpha_correction * correction_loss +
        alpha_consistency * consistency_loss
    )
    
    return total_loss, {
        'correction_loss': correction_loss.item(),
        'consistency_loss': consistency_loss.item()
    }

# ==================== 数据集 ====================
class StructureAwareCorrectorDatasetV5(Dataset):
    """
    结构感知修正器数据集 V5
    
    关键改进：
    - 使用StructurePatternLearner学习产品名称到规格结构的映射
    - 为每个样本生成结构一致性标签
    """
    
    def __init__(self, df, char2id, name2id, structure_learner, max_len=128):
        self.data = []
        self.max_len = max_len
        self.structure_learner = structure_learner
        
        for r in df.itertuples():
            ocr = str(r.ppocrstr3).strip().upper()
            gt = str(r.cinvstd).strip().upper()
            name = str(r.cinvname).strip().upper()
            
            # 标准化
            ocr = re.sub(r'\s+', '', ocr)
            gt = re.sub(r'\s+', '', gt)
            name = re.sub(r'\s+', '', name)
            
            # 从产品名称推断结构模式
            pattern, confidence = structure_learner.infer_structure(name)
            structure_pattern = structure_learner.encode_structure(pattern)
            
            # 生成结构一致性标签
            consistency_mask = self._generate_consistency_mask(ocr, gt, structure_pattern)
            
            # 编码
            ocr_tokens = self._encode(ocr, char2id)
            name_tokens = self._encode(name, name2id)
            target_tokens = self._encode(gt, char2id)
            
            self.data.append({
                'ocr_tokens': torch.LongTensor(ocr_tokens),
                'name_tokens': torch.LongTensor(name_tokens),
                'structure_pattern': torch.LongTensor(structure_pattern),
                'target_tokens': torch.LongTensor(target_tokens),
                'consistency_mask': torch.LongTensor(consistency_mask),
                'confidence': confidence,
                'ocr_str': ocr,
                'gt_str': gt,
                'name_str': name
            })
    
    def _encode(self, s, char2id):
        """编码字符串"""
        ids = [char2id["<SOS>"]]
        unk = char2id.get("<UNK>", char2id["<PAD>"])
        
        for c in str(s):
            ids.append(char2id.get(c, unk))
        
        ids.append(char2id["<EOS>"])
        
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids = ids + [char2id["<PAD>"]] * (self.max_len - len(ids))
        
        return ids
    
    def _generate_consistency_mask(self, ocr, gt, structure_pattern):
        """
        生成结构一致性标签
        
        基于逻辑：
        - 如果OCR字符在错误位置（不符合结构模式），标记为不一致
        - 如果OCR字符位置正确但字符错误，标记为不一致
        - 如果OCR字符正确，标记为一致
        """
        mask = []
        
        # 获取OCR字符类型
        ocr_types = []
        for c in ocr:
            if re.match(r'[0-9]', c):
                ocr_types.append(0)  # DIGIT
            elif re.match(r'[A-Z]', c):
                ocr_types.append(1)  # ALPHA
            elif re.match(r'[\u4e00-\u9fff]', c):
                ocr_types.append(2)  # CHINESE
            else:
                ocr_types.append(3)  # SYMBOL
        
        # 对比OCR类型和预期结构
        for i in range(min(len(ocr_types), len(structure_pattern))):
            if i >= len(ocr):
                mask.append(-1)
            else:
                expected_type = structure_pattern[i]
                actual_type = ocr_types[i] if i < len(ocr_types) else 4
                
                # 如果预期类型和实际类型不一致，需要修正
                if expected_type != actual_type and expected_type != 4:
                    mask.append(1)  # 不一致
                else:
                    mask.append(0)  # 一致
        
        # 填充
        while len(mask) < self.max_len:
            mask.append(-1)
        
        return mask
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
