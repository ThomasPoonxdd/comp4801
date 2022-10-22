import torch
import clip

single_template = [
    'a photo of {article} {}.'
]

class CLIP():
    # def try_gpu(i):
    #     return torch.cuda.device(i)

    def cat_name(c, rm_dot=False):
        c = c.replace("_", " ").replace("/", " or ").lower()
        if rm_dot == True:
            c = c.rstrip(".")
        return c
    
    def article(name):
        if name[0] in "aeiou":
            return "an"
        else:
            return "a"

    def __init__(self):
        self.model, preprocess = clip.load("ViT-B/32")

    def __call__(self, category):
        all_embeddings=[]
        with torch.no_grad():
            print("Build Text Embedding")
            texts = [
                template.format(
                    self.cat_name(category["name"]),
                    article = self.article(category["name"])
                ) for template in single_template
            ]
            texts = clip.tokenize(texts)
            if torch.cuda.is_available():
                texts = texts.cuda()
            
            t_embeddings = self.model.encode_text("")
            t_embeddings /= t_embeddings.norm(dim = -1, keep_dim = True) # L2 Norm
            t_embedding = t_embedding.mean(dim = 0)
            t_embedding = t_embedding.norm() # L2 Norm
            print(t_embeddings.shape, t_embedding.shape)
            all_embeddings.append(t_embedding)
        all_embeddings = torch.stack(all_embeddings, dim = 1) #all_embedding.shape: (#dim, #category)
        # if torch.cuda.is_available():
        #         all_embeddings = all_embeddings.cuda()
        return all_embeddings.cpu().numpy().T

    