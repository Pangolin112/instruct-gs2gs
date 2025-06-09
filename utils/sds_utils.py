import torch


def encode_prompt_with_a_prompt_and_n_prompt(batch_size, prompt, a_prompt, n_prompt, tokenizer, text_encoder, device, particle_num_vsd):
    text_input = tokenizer(
        [prompt + a_prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [n_prompt] * batch_size, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    return torch.cat([uncond_embeddings[:particle_num_vsd], text_embeddings[:particle_num_vsd]])
