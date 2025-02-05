import torch
import torch.nn as nn

def get_first_attention_weights(model, x):
    """
    Extracts attention weights from the first multi-head self-attention layer by hooking into self_attn.forward().
    
    Args:
        model (nn.Module): Transformer model.
        x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

    Returns:
        Tensor: Attention weights of shape (batch_size, num_heads, seq_len, seq_len).
    """
    q, k, v = None, None, None  # To store query, key, value
    attn_weights = None

    # Get the first encoder layer's self-attention module
    first_attn_layer = model.transformer_encoder.layers[0].self_attn

    # Store the original forward method
    original_forward = first_attn_layer.forward

    # Define a wrapper function that captures attention weights
    def forward_with_attn(*args, **kwargs):
        nonlocal q, k, v
        q, k, v = args[:3]
        output, _ = original_forward(*args, **kwargs)  # Extract second output (attention weights)
        return output  # Return only the transformed embeddings

    # Replace the original forward method with our wrapped function
    first_attn_layer.forward = forward_with_attn

    # Perform a forward pass to trigger our modified function
    _ = model(x)

    # Restore the original forward method
    first_attn_layer.forward = original_forward

    if q is None or k is None:
        raise RuntimeError("Failed to capture Q, K, V. Check model architecture.")
    
    model.eval()
    _, attention = model.transformer_encoder.layers[0].self_attn.forward(q, k, v)

    return attention  # Shape: (batch_size, num_heads, seq_len, seq_len)
