import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from transformers import BertConfig, BertModel

class ConvNeXtTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, max_len=128, device='cuda'):
        super(ConvNeXtTransformerDecoder, self).__init__()
        self.device = device

        # Load ConvNeXt as image encoder
        convnext = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.cnn_encoder = nn.Sequential(*list(convnext.children())[:-2])  # remove final classifier

        self.img_proj = nn.Linear(1024, 768)  # ConvNeXt output (B, 1024, H, W) → match decoder dim

        # Transformer decoder (BERT-style used as decoder)
        config = BertConfig(
            is_decoder=True,
            add_cross_attention=True,
            vocab_size=vocab_size,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=6,
            max_position_embeddings=max_len,
            pad_token_id=0
        )
        self.transformer_decoder = BertModel(config)
        self.lm_head = nn.Linear(768, vocab_size)

    def forward(self, images, input_ids, attention_mask, labels=None):
        # Encode image
        img_feat = self.cnn_encoder(images)  # (B, 1024, H, W)
        B, C, H, W = img_feat.shape
        img_feat = img_feat.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        img_feat = self.img_proj(img_feat)  # (B, N, 768)

        # Decode
        decoder_outputs = self.transformer_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=img_feat,
            encoder_attention_mask=torch.ones(img_feat.shape[:2], dtype=torch.long).to(self.device)
        )

        logits = self.lm_head(decoder_outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {"loss": loss, "logits": logits}

    def generate(self, images, tokenizer, max_len=64):
        self.eval()
        with torch.no_grad():
            # Encode image
            img_feat = self.cnn_encoder(images)  # (B, 1024, H, W)
            B, C, H, W = img_feat.shape
            img_feat = img_feat.view(B, C, -1).permute(0, 2, 1)
            img_feat = self.img_proj(img_feat)

            # Start token
            input_ids = torch.full((B, 1), tokenizer.cls_token_id, dtype=torch.long).to(self.device)

            for _ in range(max_len):
                attention_mask = torch.ones_like(input_ids)
                decoder_outputs = self.transformer_decoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    encoder_hidden_states=img_feat,
                    encoder_attention_mask=torch.ones(img_feat.shape[:2], dtype=torch.long).to(self.device)
                )
                logits = self.lm_head(decoder_outputs.last_hidden_state)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                if (next_token == tokenizer.sep_token_id).all():
                    break

            return input_ids
