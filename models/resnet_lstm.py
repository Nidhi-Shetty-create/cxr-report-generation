import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNetLSTM(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=1, vocab_size=30522, embed_dim=256):
        super(ResNetLSTM, self).__init__()

        # Load pretrained ResNet-50 and remove final classifier
        self.cnn = resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # remove fc layer

        # Linear layer to project CNN features
        self.img_proj = nn.Linear(2048, embed_dim)

        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM decoder
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)

        # Output projection
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, input_ids, labels=None):
        batch_size = images.size(0)

        # Extract image features from ResNet-50
        with torch.no_grad():
            img_feats = self.cnn(images).squeeze(-1).squeeze(-1)  # (B, 2048)

        img_embeds = self.img_proj(img_feats).unsqueeze(1)  # (B, 1, embed_dim)

        # Token embeddings
        text_embeds = self.embedding(input_ids)  # (B, seq_len, embed_dim)

        # Concatenate image embedding as first token
        lstm_input = torch.cat([img_embeds, text_embeds], dim=1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(lstm_input)  # (B, seq_len+1, hidden_dim)

        # Predict vocabulary tokens
        logits = self.fc_out(lstm_out)  # (B, seq_len+1, vocab_size)

        if labels is not None:
            # Pad labels with ignore index for image token
            labels = torch.cat([
                torch.full((batch_size, 1), -100, dtype=torch.long, device=labels.device),
                labels
            ], dim=1)
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return {"loss": loss, "logits": logits}

        return {"logits": logits}

    def generate(self, images, tokenizer, max_len=50, device='cuda'):
        with torch.no_grad():
            img_feats = self.cnn(images).squeeze(-1).squeeze(-1)
            img_embeds = self.img_proj(img_feats).unsqueeze(1)

            inputs = torch.full((images.size(0), 1), tokenizer.bos_token_id, dtype=torch.long).to(device)
            outputs = []

            hidden = None
            for _ in range(max_len):
                text_embeds = self.embedding(inputs)
                lstm_input = torch.cat([img_embeds, text_embeds], dim=1)
                lstm_out, hidden = self.lstm(lstm_input, hidden)
                next_token_logits = self.fc_out(lstm_out[:, -1, :])
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                outputs.append(next_token)
                inputs = next_token
                img_embeds = torch.zeros_like(img_embeds)  # Only use image embed for first step

            return torch.cat(outputs, dim=1)
