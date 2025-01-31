import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
            nn.ReLU(),
            nn.LayerNorm(embed_dim,),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim,),
            #nn.ReLU(),
            #nn.Linear(embed_dim, embed_dim),
        )
       

        # Positional encoding
        #self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        #nn.init.uniform_(self.positional_encoding, -0.1, 0.1)  # Initialize position embeddings
        self.positional_encoding = SinusoidalPositionalEncoding(embed_dim, max_seq_len=seq_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
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
        # Reconstruction head for masked sequence prediction
        self.reconstruction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim,),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim,),
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
        return nn.Sigmoid()(x)
        return F.sigmoid(x, dim=-1)  # Output log probabilities for cross-entropy loss
    
    def reconstruct(self, x, mask=None):
        """
        Forward pass of the pretraining model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            mask (Tensor): Boolean tensor of shape (batch_size, seq_len, input_dim), indicating masked elements.

        Returns:
            Tensor: Reconstructed sequence of shape (batch_size, seq_len, input_dim).
        """
        # Project input features to embedding dimension
        x = self.feature_embedding(x)  # Shape: (batch_size, seq_len, embed_dim)
        # Add positional encoding
        x = x + self.positional_encoding  # Shape: (batch_size, seq_len, embed_dim)
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # Shape: (batch_size, seq_len, embed_dim)

        # Reconstruct the original features for masked elements
        reconstructed = self.reconstruction_head(x)  # Shape: (batch_size, seq_len, input_dim)

        return reconstructed
    


class KinematicsLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.1, bidirectional=False):
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
        super(KinematicsLSTM, self).__init__()

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
        self.fc = nn.Linear(hidden_dim * direction_multiplier, num_classes)

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
    def __init__(self, input_dim, hidden_dim, num_classes, lstm_layers=3, body_part_feature_num=13):
        super(BANet, self).__init__()
        self.hidden_dim = hidden_dim
        self.body_part_feature_num = body_part_feature_num
        # LSTM Encoder
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=lstm_layers, batch_first=True)
        
        # Temporal Attention
        self.temporal_attn = nn.Conv1d(hidden_dim, 1, kernel_size=1)  # 1x1 Conv
        
        # Set-wise Attention (2-layer FC)
        self.set_attn_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.set_attn_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Final classification layer
        self.fc = nn.Linear(hidden_dim * input_dim, num_classes)  # Flattened input
    
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
        # Reshape for LSTM (treat each set element as a separate batch)
        x = x.view(B * S, L, D)
        H, _ = self.lstm(x)  # H: (B*S, L, K)
        H = H.view(B, S, L, -1)  # Reshape back: (B, S, L, K)
        
        # Temporal Attention
        H_permuted = H.permute(0, 1, 3, 2)  # (B, S, K, L) for Conv1d
        A = self.temporal_attn(H_permuted).squeeze(2)  # (B, S, L)
        A = F.softmax(A / (self.hidden_dim ** 0.5), dim=-1)  # Softmax along L
        
        # Weighted sum along L
        H_weighted = (A.unsqueeze(-1) * H).sum(dim=2)  # (B, S, K)
        
        # Set-wise Attention
        B_attn = torch.tanh(self.set_attn_fc1(H_weighted))  # (B, S, K)
        B_attn = self.set_attn_fc2(B_attn)  # (B, S, K)
        B_attn = F.softmax(B_attn / (self.hidden_dim ** 0.5), dim=1)  # Softmax along S
        
        # Hadamard product
        H_final = H_weighted * B_attn  # (B, S, K)
        
        # Flatten and classify
        out = H_final.view(B, -1)  # (B, S*K)
        out = self.fc(out)  # (B, num_classes)
        return out

















class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=3.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none')
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
