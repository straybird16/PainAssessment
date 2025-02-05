import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, Set
from typing_extensions import Literal
from functools import reduce

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=5000):
        """
        Sinusoidal Positional Encoding.

        Args:
            embed_dim (int): Dimension of the embedding vector.
            max_seq_len (int): Maximum sequence length.
        """
        super(SinusoidalPositionalEncoding, self).__init__()
        self.scale = 1e-2   # scale of positional encoding
        self.scale = 1/math.sqrt(embed_dim)
        
        # Create the positional encoding matrix
        position = torch.arange(0, max_seq_len).unsqueeze(1)  # Shape: (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices

        # Register as a buffer so it doesn't get updated during backprop
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_seq_len, embed_dim)

    def forward(self, x):
        """
        Add positional encoding to input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            Tensor: Input tensor with positional encoding added.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len] * self.scale  # Add positional encoding (broadcasted)

class KinematicsTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, seq_len, embed_dim, num_heads, num_layers, dropout:float=0.1, per_time_step:bool=False):
        """
        Transformer model for kinematics-based pain level prediction.

        Args:
            input_dim (int): Number of features (e.g., 26 of kinematic features in EmoPain dataset).
            num_classes (int): Number of output classes (for cross-entropy, use 1 for regression-like classification).
            seq_len (int): Length of the input sequence (window length).
            embed_dim (int): Embedding dimension for transformer.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
            dropout (float): Dropout rate.
        """
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.pretrain = False
        self.per_time_step = per_time_step

        # Linear projection for input features to embedding dimension
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            #nn.ReLU(),
            #nn.LayerNorm(embed_dim,),
            #nn.Linear(embed_dim, embed_dim),
            #nn.LayerNorm(embed_dim,),
            #nn.ReLU(),
            #nn.Linear(embed_dim, embed_dim),
        )
       
        # Positional encoding
        #self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        #nn.init.uniform_(self.positional_encoding, -0.1, 0.1)  # Initialize position embeddings
        self.positional_encoding = SinusoidalPositionalEncoding(embed_dim, max_seq_len=seq_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=0, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.backbone = nn.ModuleList([self.feature_embedding, self.positional_encoding, self.transformer_encoder])
        # Fully connected layer for output prediction
        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim,),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim,),
            nn.Linear(embed_dim, num_classes),
        )
        self.prediction_head =nn.Linear(embed_dim, num_classes)
        # Reconstruction head for masked sequence prediction
        self.reconstruction_head = nn.Sequential(
            #nn.Linear(embed_dim, embed_dim),
            #nn.ReLU(),
            #nn.LayerNorm(embed_dim,),
            nn.Linear(embed_dim, input_dim),
        )
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the Transformer model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Tensor: Output probabilities of shape (batch_size, num_classes).
        """
        # Project input features to embedding dimension
        x = self.feature_embedding(x)  # Shape: (batch_size, seq_len, embed_dim)
        # Add positional encoding
        #x = x + self.positional_encoding  # Shape: (batch_size, seq_len, embed_dim)
        x = self.positional_encoding(x)
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # Shape: (batch_size, seq_len, embed_dim)
        if self.pretrain:
            # Reconstruct the original features for masked elements
            reconstructed = self.reconstruction_head(x)  # Shape: (batch_size, seq_len, input_dim)
            return reconstructed
        # Aggregate over the sequence (e.g., take the mean over time dimension)
        #x = x.mean(dim=1)  # Shape: (batch_size, embed_dim)
        # Pass through fully connected layer and softmax activation
        x = self.prediction_head(self.dropout(x))  # Shape: (batch_size, seq_len, num_classes)
        if not self.per_time_step:
            x = x.mean(dim=-2) # Shape: (batch_size, num_classes)
        #return x
        #return nn.Softmax(dim=-1)(x)
        return nn.Sigmoid()(x)
        return F.sigmoid(x, dim=-1)  # Output log probabilities for cross-entropy loss

class MultiTaskTransformer(nn.Module):
    def __init__(self, input_dim, joint_dim, embed_dim, num_heads, num_layers, num_classes_pain, num_classes_behavior, num_classes_activity, dropout=0.1, seq_len=180):
        """
        Multi-task Transformer Model with dynamic task selection.
        """
        super().__init__()

        # Backbone encoder
        self.kinematics_embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(embed_dim, max_seq_len=seq_len)
        self.kinematics_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )

        # Decoders
        self.joint_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.reconstruction_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )

        # Task-specific heads
        self.joint_output_head = nn.Linear(embed_dim, joint_dim)
        self.reconstruction_output_head = nn.Linear(embed_dim, input_dim)
        self.pain_level_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim,),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim,),
            nn.Linear(embed_dim, num_classes_pain),
        )
        self.behavior_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim,),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim,),
            nn.Linear(embed_dim, num_classes_behavior),
        )
        self.activity_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim,),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim,),
            nn.Linear(embed_dim, num_classes_activity),
        )
        self.dropout = nn.Dropout(dropout)
        self.backbone = nn.ModuleList([self.kinematics_embedding, self.positional_encoding, self.kinematics_encoder])
        # output shape of tasks
        self.per_frame_behavior = False  # output shape of behavior head is (B, L, D) if True else (B, D)

        # Default task selection
        self.set_task('predict_behavior')

    def set_task(self, task_name:Literal["predict_joint", "reconstruct_kinematics", "predict_pain",
            "predict_behavior", "predict_activity"]):
        """
        Set the current task mode for forward pass.
        """
        valid_tasks = [
            "predict_joint", "reconstruct_kinematics", "predict_pain",
            "predict_behavior", "predict_activity"
        ]
        if task_name not in valid_tasks:
            raise ValueError(f"Invalid task '{task_name}'. Choose from {valid_tasks}.")
        self.current_task = task_name
    
    def get_task_modules(self, task_name:Literal["predict_joint", "reconstruct_kinematics", "predict_pain",
            "predict_behavior", "predict_activity"]):
        """
        Return the corresponding module other than the backbone for the given task 
        """
        task_modules = {
            "predict_joint": nn.ModuleList([self.joint_decoder, self.joint_output_head]),
            "reconstruct_kinematics": nn.ModuleList([self.reconstruction_decoder, self.reconstruction_output_head]),
            "predict_pain": self.pain_level_head,
            "predict_behavior": self.behavior_head,
            "predict_activity": self.activity_head
        }
        return task_modules[task_name]

    def forward(self, kinematics, joint_targets=None):
        """
        Forward pass with task-based output control.
        """
        if self.current_task is None:
            raise RuntimeError("Task not set! Call model.set_task(task_name) before forward().")

        encoded_kinematics = self.kinematics_encoder(self.positional_encoding(self.kinematics_embedding(kinematics)))

        if self.current_task == "predict_joint":
            joint_memory = self.joint_decoder(encoded_kinematics, joint_targets) if joint_targets is not None else encoded_kinematics
            return self.joint_output_head(joint_memory)

        elif self.current_task == "reconstruct_kinematics":
            reconstruction_memory = self.reconstruction_decoder(encoded_kinematics, encoded_kinematics)
            return self.reconstruction_output_head(reconstruction_memory)

        elif self.current_task == "predict_pain":
            return self.pain_level_head(self.dropout(encoded_kinematics))

        elif self.current_task == "predict_behavior":
            output = self.behavior_head(self.dropout(encoded_kinematics))
            if self.per_frame_behavior == True:
                return output   # Output per-frame probabilities of behavio
            return output.mean(dim=-2)

        elif self.current_task == "predict_activity":
            return self.activity_head(self.dropout(encoded_kinematics))



class KinematicsLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, dropout=0.1, bidirectional=False):
        """
        LSTM model for per-time-step classification.

        Args:
            input_dim (int): Number of features (e.g., 20 in your dataset).
            hidden_dim (int): Number of hidden units in the LSTM.
            num_classes (int): Number of output classes (e.g., 1 for binary or multi-class classification).
            num_layers (int): Number of LSTM layers (default: 2).
            dropout (float): Dropout rate for regularization (default: 0.1).
            bidirectional (bool): Whether to use bidirectional LSTM (default: False).
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        # Fully connected layer to output classification for each time step
        direction_multiplier = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * direction_multiplier, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 128),
            )

    def forward(self, x):
        """
        Forward pass of the LSTM model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, num_classes).
        """
        # LSTM output
        lstm_out, _ = self.lstm(x)  # Shape: (batch_size, seq_len, hidden_dim * direction_multiplier)

        # Fully connected layer (applied to each time step)
        out = self.fc(lstm_out)  # Shape: (batch_size, seq_len, num_classes)

        return out

class BANet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, lstm_layers=3, body_part_feature_num=13, dropout=0, batch_first=True):
        super().__init__()
        self.pretrain=False
        self.hidden_dim = hidden_dim
        self.body_part_feature_num = body_part_feature_num
        # LSTM Encoder
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=lstm_layers, batch_first=batch_first,dropout=0)
        
        # Temporal Attention
        self.temporal_attn = nn.Conv1d(hidden_dim, 1, kernel_size=1)  # 1x1 Conv
        
        # Set-wise Attention (2-layer FC)
        self.set_attn_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.set_attn_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Final classification layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * body_part_feature_num, num_classes)  # Flattened input

        # Reconstruction head for masked sequence prediction
        self.reconstruction_head = nn.LSTM(hidden_dim, input_dim, num_layers=lstm_layers, batch_first=batch_first, dropout=0)

        self.backbone = nn.ModuleList([self.lstm, self.temporal_attn, self.set_attn_fc1, self.set_attn_fc2])
        self.prediction_head = self.fc
    
    def forward(self, x):
        """
        x: (B, L, D')   # Input shape. Will be converted to shape (B, S, L, D) where S*D=D'
        """
        L, D_ = x.shape[-2], x.shape[-1]
        S = self.body_part_feature_num
        if D_ % S != 0:
            raise ValueError("Invalid input shape. Expected last dimension a mutiple of body_part_feature_num. But got dimension {} \
                            and body_part_feature_num={}".format(D_, S))
        x = x.view(-1, L, D_//S, S)
        x = x.permute(0, 1, 3, 2)   # Permute to swap the last two dimensions so that interleaved body feature groups are now together; (B, L, S, D)
        x = x.permute(0, 2, 1, 3)   # (B, S, L, D)
        # Ensure input has 4 dimensions (B, S, L, D)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        B, S, L, D = x.shape
        K = self.hidden_dim
        # Reshape for LSTM (treat each set element as a separate batch)
        x = x.reshape(B * S, L, D)
        H, (hidden, cell) = self.lstm(x)  # H: (B*S, L, K)
        H = H.view(B, S, L, -1)  # Reshape back: (B, S, L, K)

        
        # Temporal Attention
        H_permuted = H.permute(0, 1, 3, 2).reshape(-1, K, L)  # (B, S, K, L) for Conv1d
        A = self.temporal_attn(H_permuted).squeeze(-2).reshape(B, S, L)  # (B, S, L)
        A = F.softmax(A / (self.hidden_dim ** 0.5), dim=-1)  # Softmax along L
        
        # Weighted sum along L
        H_weighted = (A.unsqueeze(-1) * H).sum(dim=2)  # (B, S, K)
        if self.pretrain:
            # Reconstruct the original sequence for pretraining
            reconstructed, _ = self.reconstruction_head(H_weighted.reshape(-1, K).unsqueeze(-2).expand(B*S, L, K))  # Shape: (B*S, L, D)
            reconstructed = reconstructed.view(B, S, L, -1)
            reconstructed = reconstructed.permute(0, 2, 1, 3) # Shape: (B, L, S, D)
            reconstructed = reconstructed.permute(0, 1, 3, 2).reshape(B, L, D_)
            return reconstructed
        
        # Set-wise Attention
        B_attn = torch.tanh(self.set_attn_fc1(H_weighted))  # (B, S, K)
        B_attn = self.set_attn_fc2(B_attn)  # (B, S, K)
        B_attn = F.softmax(B_attn / (self.hidden_dim ** 0.5), dim=1)  # Softmax along S
        
        # Hadamard product
        H_final = H_weighted * B_attn  # (B, S, K)
        
        # Flatten and classify
        out = H_final.view(B, -1)  # (B, S*K)
        out = self.fc(self.dropout(out))  # (B, num_classes)
        return out
        return nn.Softmax(dim=-1)(out)

class MultiClassBANet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, lstm_layers=3, body_part_feature_num=13, dropout=0, batch_first=True):
        super().__init__()
        self.pretrain=False
        self.hidden_dim = hidden_dim
        self.body_part_feature_num = body_part_feature_num
        # LSTM Encoder
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=lstm_layers, batch_first=batch_first,dropout=0)
        
        # Temporal Attention
        self.temporal_attn = nn.Conv1d(hidden_dim, 1, kernel_size=1)  # 1x1 Conv
        
        # Set-wise Attention (2-layer FC)
        self.set_attn_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.set_attn_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Final classification layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * body_part_feature_num, num_classes)  # Flattened input

        # Reconstruction head for masked sequence prediction
        self.reconstruction_head = nn.LSTM(hidden_dim, input_dim, num_layers=lstm_layers, batch_first=batch_first, dropout=0)

        self.backbone = nn.ModuleList([self.lstm, self.temporal_attn, self.set_attn_fc1, self.set_attn_fc2])
        self.prediction_head = self.fc
    
    def forward(self, x):
        """
        x: (B, L, D')   # Input shape. Will be converted to shape (B, S, L, D) where S*D=D'
        """
        L, D_ = x.shape[-2], x.shape[-1]
        S = self.body_part_feature_num
        if D_ % S != 0:
            raise ValueError("Invalid input shape. Expected last dimension a mutiple of body_part_feature_num. But got dimension {} \
                            and body_part_feature_num={}".format(D_, S))
        x = x.view(-1, L, D_//S, S)
        x = x.permute(0, 1, 3, 2)   # Permute to swap the last two dimensions so that interleaved body feature groups are now together; (B, L, S, D)
        x = x.permute(0, 2, 1, 3)   # (B, S, L, D)
        # Ensure input has 4 dimensions (B, S, L, D)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        B, S, L, D = x.shape
        K = self.hidden_dim
        # Reshape for LSTM (treat each set element as a separate batch)
        x = x.reshape(B * S, L, D)
        H, (hidden, cell) = self.lstm(x)  # H: (B*S, L, K)
        H = H.view(B, S, L, -1)  # Reshape back: (B, S, L, K)

        
        # Temporal Attention
        H_permuted = H.permute(0, 1, 3, 2).reshape(-1, K, L)  # (B, S, K, L) for Conv1d
        A = self.temporal_attn(H_permuted).squeeze(-2).reshape(B, S, L)  # (B, S, L)
        A = F.softmax(A / (self.hidden_dim ** 0.5), dim=-1)  # Softmax along L
        
        # Weighted sum along L
        H_weighted = (A.unsqueeze(-1) * H).sum(dim=2)  # (B, S, K)
        if self.pretrain:
            # Reconstruct the original sequence for pretraining
            reconstructed, _ = self.reconstruction_head(H_weighted.reshape(-1, K).unsqueeze(-2).expand(B*S, L, K))  # Shape: (B*S, L, D)
            reconstructed = reconstructed.view(B, S, L, -1)
            reconstructed = reconstructed.permute(0, 2, 1, 3) # Shape: (B, L, S, D)
            reconstructed = reconstructed.permute(0, 1, 3, 2).reshape(B, L, D_)
            return reconstructed
        
        # Set-wise Attention
        B_attn = torch.tanh(self.set_attn_fc1(H_weighted))  # (B, S, K)
        B_attn = self.set_attn_fc2(B_attn)  # (B, S, K)
        B_attn = F.softmax(B_attn / (self.hidden_dim ** 0.5), dim=1)  # Softmax along S
        
        # Hadamard product
        H_final = H_weighted * B_attn  # (B, S, K)
        
        # Flatten and classify
        out = H_final.view(B, -1)  # (B, S*K)
        out = self.fc(self.dropout(out))  # (B, num_classes)
        return out


def _generate_autoregressive_mask(seq_len, device):
    """
    Generates a causal mask for autoregressive decoding.
    
    Args:
        seq_len (int): Sequence length (T).
        device (torch.device): Device for tensor computation.
    
    Returns:
        Tensor: Autoregressive mask of shape (T, T).
    """
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    return mask.to(device)  # Ensure mask is on the correct device




"""
Loss functions: FocalLoss
"""

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=3.0, reduction='mean', label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing  # Smoothing parameter for label smoothing

    def forward(self, inputs, targets):
        ce = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing, reduction='none')
        #ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        ce_loss = ce(inputs, targets)  # compute cross entropy loss
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss







import types

# Get all names in this module, excluding private ones and modules
__all__ = [name for name, thing in globals().items()
          if not (name.startswith('_') or isinstance(thing, types.ModuleType))]
del types
