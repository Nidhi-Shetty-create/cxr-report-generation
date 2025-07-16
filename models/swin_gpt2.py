import torch
import torch.nn as nn
from transformers import SwinModel, SwinConfig, GPT2LMHeadModel, GPT2Config

class SwinGPT2Model(nn.Module):
    def __init__(self, swin_model_name='microsoft/swin-base-patch4-window7-224', 
                 gpt2_model_name='gpt2', device='cuda'):
        super(SwinGPT2Model, self).__init__()

        self.device = device

        # Swin Transformer as image encoder
        self.vision_encoder = SwinModel.from_pretrained(swin_model_name)
        self.vision_hidden_size = self.vision_encoder.config.hidden_size

        # GPT-2 decoder
        self.text_decoder = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.text_hidden_size = self.text_decoder.config.n_embd

        # Project vision output to GPT-2 embedding space
        self.proj = nn.Linear(self.vision_hidden_size, self.text_hidden_size)

    def forward(self, pixel_values, input_ids, labels=None):
        # Encode image
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        img_embedding = vision_outputs.last_hidden_state.mean(dim=1)  # (B, D)
        projected_embedding = self.proj(img_embedding)  # (B, D)

        # Add image embedding as prefix to input tokens
        batch_size = input_ids.size(0)
        prefix_tokens = projected_embedding.unsqueeze(1)  # (B, 1, D)
        input_embeddings = self.text_decoder.transformer.wte(input_ids)  # (B, T, D)
        combined_embeddings = torch.cat([prefix_tokens, input_embeddings], dim=1)  # (B, T+1, D)

        # Generate attention mask (prefix gets 1s)
        attention_mask = torch.ones(combined_embeddings.size()[:-1], dtype=torch.long).to(self.device)

        outputs = self.text_decoder(inputs_embeds=combined_embeddings, attention_mask=attention_mask, labels=labels)

        return {"loss": outputs.loss, "logits": outputs.logits}

    def generate(self, pixel_values, tokenizer, max_length=50):
        # Get image embedding
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        img_embedding = vision_outputs.last_hidden_state.mean(dim=1)  # (B, D)
        prefix = self.proj(img_embedding).unsqueeze(1)  # (B, 1, D)

        # Generate using GPT-2
        generated = self.text_decoder.generate(
            inputs_embeds=prefix,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

        return generated
