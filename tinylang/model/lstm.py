import torch
import torch.nn as nn
from .model import Model

class LSTM(Model):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_layer: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize embeddings
        self.embedding = nn.Embedding(vocab_size, n_embd)
        
        # Initialize LSTM
        self.lstm = nn.LSTM(
            input_size=n_embd,
            hidden_size=n_embd,
            num_layers=n_layer,
            batch_first=True,
            dropout=0.0
        )
        
        # Initialize output layer
        self.fc = nn.Linear(n_embd, vocab_size)

        self.model = nn.Sequential(
            self.embedding,
            self.lstm,
            self.fc
        )
        
        # Move to device
        self.model.to(self.device)
    
    def step(self, input_ids: torch.Tensor, labels: torch.Tensor):
        """Run a single step.
        
        Args:
            input_ids: The input tokens
            labels: The labels

        Returns:
            A tuple containing the logits and the loss
        """
        # Move tensors to device
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        
        # Get embeddings
        x = self.embedding(input_ids)
        
        # Run LSTM
        lstm_out, _ = self.lstm(x)
        
        # Get logits
        logits = self.fc(lstm_out)
        
        # Calculate loss
        loss = nn.CrossEntropyLoss()(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        return {
            "logits": logits.cpu(),
            "loss": loss,
            "hidden_states": [lstm_out.cpu()],  # For consistency with other models
            "attentions": [],  # LSTM doesn't have attention
        }
    
    def save(self, path: str):
        """Save the model to a file."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.embedding.num_embeddings,
            'n_embd': self.embedding.embedding_dim,
            'n_layer': self.lstm.num_layers,
        }, path) 