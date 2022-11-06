import torchvision
import torch
from torch import nn
# from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from vild_pytorch.text_embedding import CLIP
# import torchvision
class vild(nn.Module):
    def __init__(self):
        super(vild,self).__init__()
        self.clip = CLIP()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
                # frcnn_params = params["fcrnn_head"]
        layer_list = []
        flatten_size = self.model.backbone.out_channels*self.model.roi_heads.box_roi_pool.output_size[0]**2
        # input size: [B,N_proposal,256,7,7]
        for i in range(4):
            layer_list.append(nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=(1,1)))
        layer_list.append(nn.Flatten(start_dim=-3))
        input_size = flatten_size
        for i in range(2):
            layer_list.append(nn.Linear(input_size,1024))
            input_size = 1024
        
        layer_list.append(nn.BatchNorm1d(1024))

        self.region = nn.Sequential( *layer_list )

        self.vild_projection = nn.Sequential(
            nn.Linear(input_size, 512)
        )
        self.model.rpn.training=False
        self.background = nn.Parameter(torch.rand(1,512))
    
    def forward(self, images,categories, targets=None ):
        """
            images (list[Tensor]): images to be processed
            categories (list[String]): label list
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
            =============
            this function is modified on torchvision.models.detection.generalized_rcnn
        """
        categories = ['background'] + categories
        categories = [{'name': item, 'id': idx+1,} for idx, item in enumerate(categories)]

        original_image_sizes = [] # list[Tuple[int,int]]
        for image in images:
            shape = image.shape
            # print(shape)
            original_image_sizes.append([shape[-2], shape[-1]])
            
        images, targets = self.model.transform(images, targets)
        features = self.model.backbone(images.tensors)
        proposals, proposal_losses = self.model.rpn(images, features, targets)  # proposal:list[Tensor[1000, 4]]
        # print(type(proposals))
        # return proposals
        # features = list(features.values())
        # objectness, pred_bbox_deltas = self.model.rpn.head(features)
        # anchors = self.model.rpn.anchor_generator(images, features)
        # num_images = len(anchors)
        # num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        # num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        # objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # # note that we detach the deltas because Faster R-CNN do not backprop through
        # # the proposals
        # proposals = self.model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        # proposals = proposals.view(num_images, -1, 4)
        # boxes, scores = self.model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        
        # box_features = self.model.roi_heads.box_roi_pool(features, proposals, images.image_sizes) #[B* 1000, 256, 7, 7]
        # region_embedding = self.region(box_features) #[B*1000, 512]
        # region_embedding = region_embedding.reshape(-1, len(proposals[0]), 512) #reshape it to [B, 1000, 512]

        # original_proposals = self.process_proposal(images, proposals, original_image_sizes )

        # text_embedding = self.clip(categories, embedding="text")

        # #text_embedding = torch.cat([self.background, text_embedding], dim=0)
        # text_embedding = torch.tensor(text_embedding)
        # cls_logits = self.sim(region_embedding, text_embedding) #[B, Bbox, Categories]
        # # softmax = nn.Softmax(dim=-1)
        # # result = softmax(result)
        # self.postprocess(cls_logits, original_proposals)

        # return result

        detections, detector_losses = self.model.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.model.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

    def process_proposal(self, images, proposals, original_image_sizes):
        proposals = torch.stack(proposals, dim=0) # list[Tensor] -> tensor [B,1000, 4]
        image_sizes = images.image_sizes
        ratios = torch.tensor([
            (torch.tensor(s[0], dtype=torch.float32, device=proposals.device)
            / torch.tensor(s_org[0], dtype=torch.float32, device=proposals.device), 
            torch.tensor(s[1], dtype=torch.float32, device=proposals.device)
            / torch.tensor(s_org[1], dtype=torch.float32, device=proposals.device)) 
            for s, s_org in zip(image_sizes, original_image_sizes)
        ])
        ratios_height, ratios_width = ratios.permute(1, 0) # transpose matrix
        xmin, ymin, xmax, ymax = proposals.unbind(2)
        xmin = xmin * ratios_width
        ymin = ymin * ratios_height
        xmax = xmax * ratios_width
        ymax = ymax * ratios_height
        return torch.stack((xmin,ymin,xmax,ymax), dim=1)

    def sim(self, region_embedding, text_embedding):
        NumBbox= region_embedding.shape[1] # Number of Bbox per image
        NumCls = text_embedding.shape[0] # Number of Categories 
        ind = []
        for i in range(NumBbox):
            for j in range(NumCls):
                ind.append(i+j*NumBbox)
                # ind.append(i+NumBbox)
        region_embeddingR = region_embedding.repeat((1,NumCls,1))[:,ind,:]  
        text_embeddingR = text_embedding.repeat((NumBbox,1))
        sim = nn.CosineSimilarity(dim=-1)
        result = sim(region_embeddingR, text_embeddingR)
        return result.reshape(-1,NumBbox,NumCls)

    def postprocess(cls_logits, proposals):
        result = torch.nn.functional.softmax(cls_logits, -1)
        

        return None

from PIL import Image
from torchvision import transforms
image = Image.open("3ppl.jpg").convert("RGB")
transform1 = transforms.Compose([
    # transforms.Resize((,244)),
    transforms.ToTensor()
])
image = transform1(image)
model =vild()
model.eval()
r = model(image.unsqueeze(0), ["luggage", "black luggage", "black object"])
