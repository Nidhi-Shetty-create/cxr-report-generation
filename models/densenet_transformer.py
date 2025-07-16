import torch
import torch.nn as nn
from torchvision.models import densenet121
from transformers import BertTokenizer, BertConfig

class DenseNetTransformerDecoder(nn.Module):
    def __init__(self, vocab_size=30522, hidden_dim=768, max_len=512):
        super(DenseNetTransformerDecoder, self).__init__()

        # Load pretrained DenseNet
        densenet = densenet121(pretrained=True)
        self.cnn = nn.Sequential(*list(densenet.features.children()))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Project CNN output to match transformer input dim
        self.img_proj = nn.Linear(1024, hidden_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # Token embeddings and output projection
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        self.max_len = max_len

    def forward(self, images, input_ids, labels=None):
        batch_size = images.size(0)

        # CNN feature extraction
        with torch.no_grad():
            features = self.cnn(images)
            pooled = self.pool(features).view(batch_size, -1)  # (B, 1024)

        img_feats = self.img_proj(pooled).unsqueeze(0)  # (1, B, hidden_dim)

        # Prepare token embeddings
        tgt_embeds = self.token_embedding(input_ids).permute(1, 0, 2)  # (T, B, D)

        # Decode
        out = self.transformer_decoder(tgt=tgt_embeds, memory=img_feats)  # (T, B, D)
        out = out.permute(1, 0, 2)  # (B, T, D)
        logits = self.fc_out(out)  # (B, T, vocab_size)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return {"loss": loss, "logits": logits}

        return {"logits": logits}

    def generate(self, images, tokenizer, max_len=50, device='cuda'):
        with torch.no_grad():
            features = self.cnn(images)
            pooled = self.pool(features).view(images.size(0), -1)
            memory = self.img_proj(pooled).unsqueeze(0)

            generated = torch.full((images.size(0), 1), tokenizer.bos_token_id, dtype=torch.long).to(device)

            for _ in range(max_len):
                tgt_embeds = self.token_embedding(generated).permute(1, 0, 2)
                out = self.transformer_decoder(tgt=tgt_embeds, memory=memory)
                logits = self.fc_out(out.permute(1, 0, 2))  # (B, T, vocab_size)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

            return generated[:, 1:]  # remove BOS
