import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torchvision.models import vit_b_16
from torchvision import transforms

class ViTGPT2(nn.Module):
    def __init__(self, image_embed_dim=768, text_embed_dim=768, gpt_model_name='gpt2'):
        super(ViTGPT2, self).__init__()

        # Load pretrained ViT encoder
        self.vit = vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()  # Remove classification head

        # GPT-2 for language generation
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt_model_name)

        # Freeze GPT-2 position embeddings if needed
        self.gpt2.resize_token_embeddings(len(self.tokenizer))

        # Map ViT output to GPT-2 hidden size
        self.proj = nn.Linear(image_embed_dim, text_embed_dim)

    def forward(self, images, input_ids, attention_mask=None, labels=None):
        # Extract image features using ViT
        image_feats = self.vit(images)  # (batch_size, image_embed_dim)
        image_feats = self.proj(image_feats)  # (batch_size, text_embed_dim)

        # Expand image features to 1 token and prepend to input_ids
        batch_size = input_ids.size(0)
        image_embeds = image_feats.unsqueeze(1)  # (batch_size, 1, text_embed_dim)

        # Get embeddings from GPT-2's embedding layer
        token_embeds = self.gpt2.transformer.wte(input_ids)  # (batch_size, seq_len, text_embed_dim)

        # Concatenate image embedding as the first token
        gpt_inputs = torch.cat([image_embeds, token_embeds], dim=1)

        # Shift labels for GPT-2
        if labels is not None:
            labels = torch.cat([
                torch.full((batch_size, 1), -100, dtype=torch.long, device=labels.device),
                labels
            ], dim=1)

        # Pass through GPT-2
        outputs = self.gpt2(inputs_embeds=gpt_inputs, attention_mask=None, labels=labels)
        return outputs

    def generate(self, images, max_length=50):
        with torch.no_grad():
            image_feats = self.vit(images)  # (batch_size, image_embed_dim)
            image_feats = self.proj(image_feats)  # (batch_size, text_embed_dim)
            image_embeds = image_feats.unsqueeze(1)

            input_ids = torch.tensor([[self.tokenizer.bos_token_id]] * images.size(0)).to(images.device)
            generated = input_ids

            for _ in range(max_length):
                token_embeds = self.gpt2.transformer.wte(generated)
                gpt_inputs = torch.cat([image_embeds, token_embeds], dim=1)
                outputs = self.gpt2(inputs_embeds=gpt_inputs)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
                generated = torch.cat((generated, next_token), dim=1)

            return generated
