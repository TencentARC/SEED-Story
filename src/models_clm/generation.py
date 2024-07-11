import torch
from transformers import LogitsProcessor

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'


class AutoImageTokenGenerationProcessor(LogitsProcessor):

    def __init__(self, tokenizer, num_img_gen_tokens=64) -> None:
        super().__init__()
        # self.boi_token_id = tokenizer.encode(BOI_TOKEN)[0]
        # self.eoi_token_id = tokenizer.encode(EOI_TOKEN)[0]
        img_all_token_str = ''.join([BOI_TOKEN] + [IMG_TOKEN.format(int(item))
                                                   for item in range(num_img_gen_tokens)] + [EOI_TOKEN])
        self.img_ids_list = tokenizer.encode(img_all_token_str, add_special_tokens=False)

    def __call__(self, input_ids, scores):
        bz = input_ids.shape[0]
        for i in range(bz):
            cur_input_id = input_ids[i, -1].item()
            if cur_input_id in self.img_ids_list[:-1]:

                output_id = self.img_ids_list[self.img_ids_list.index(cur_input_id) + 1]
                scores[i, ..., output_id] = scores[i, ...].max() + 10.
            else:

                scores[i, ..., torch.tensor(self.img_ids_list[1:]).to(dtype=torch.long)] = 0.0

        return scores
