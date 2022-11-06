import torch
from torch import nn
import torch.nn.functional as F
import json
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
# from rpn_modified import RegionProposalNetwork, RPNHead
# from torchvision.models.detection.anchor_utils import AnchorGenerator
from text_embedding import CLIP
# params = JSON
class ViLDModel(torch.Module):
    def __init__(self, params_path, training = True):
        with open(params_path, "r") as f:
            params =json.load(f)
        self.params = params
        self.model = fasterrcnn_resnet50_fpn_v2(
            # min_size, max_size
            num_classes = params["architecture"]["num_classes"],
            rpn_pre_nms_top_n_train = params["roi_proposal"]["rpn_pre_nms_top_k"],
            rpn_pre_nms_top_n_test = params["roi_proposal"]["test_rpn_pre_nms_top_k"],
            rpn_post_nms_top_n_train = params["roi_proposal"]["rpn_post_nms_top_k"],
            rpn_post_nms_top_n_test = params["roi_proposal"]["test_rpn_post_nms_top_k"],
            rpn_nms_thresh = params["roi_proposal"]["rpn_nms_threshold"],
            rpn_fg_iou_thresh = params["roi_sampling"]["fg_iou_thresh"],
            rpn_bg_iou_thresh = params["roi_sampling"]["bg_iou_thresh_hi"],
            rpn_batch_size_per_image = params["roi_sampling"]["num_samples_per_image"],
            rpn_positive_fraction = params["roi_sampling"]["fg_fraction"],
            box_score_thresh = params["postprocess"]["score_threshold"],
            box_nms_thresh = params["postprocess"]["nms_iou_threshold"]
        )
        # For Bbox delta back prob
        # num_classes = params["architecture"]["num_classes"],
        # rpn_pre_nms_top_n_train = params["roi_proposal"]["rpn_pre_nms_top_k"],
        # rpn_pre_nms_top_n_test = params["roi_proposal"]["test_rpn_pre_nms_top_k"],
        # rpn_post_nms_top_n_train = params["roi_proposal"]["rpn_post_nms_top_k"],
        # rpn_post_nms_top_n_test = params["roi_proposal"]["test_rpn_post_nms_top_k"],
        # rpn_nms_thresh = params["roi_proposal"]["rpn_nms_threshold"],
        # rpn_fg_iou_thresh = params["roi_sampling"]["fg_iou_thresh"],
        # rpn_bg_iou_thresh = params["roi_sampling"]["bg_iou_thresh_hi"],
        # rpn_batch_size_per_image = params["roi_sampling"]["num_samples_per_image"],
        # rpn_positive_fraction = params["roi_sampling"]["fg_fraction"],
        # box_score_thresh = params["postprocess"]["score_threshold"],
        # box_nms_thresh = params["postprocess"]["nms_iou_threshold"]
        # self.model = fasterrcnn_resnet50_fpn_v2(
        #     num_classes = num_classes,
        #     box_score_thresh = box_score_thresh,
        #     box_nms_thresh = box_nms_thresh
        # )

        # #================modified RPN========================
        # out_channels = self.model.backbone.out_channels

        # if rpn_anchor_generator is None:
        #     rpn_anchor_generator = _default_anchorgen()
        # if rpn_head is None:
        #     rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        # rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        # rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        # self.model.rpn = RegionProposalNetwork(
        #     rpn_anchor_generator,
        #     rpn_head,
        #     rpn_fg_iou_thresh,
        #     rpn_bg_iou_thresh,
        #     rpn_batch_size_per_image,
        #     rpn_positive_fraction,
        #     rpn_pre_nms_top_n,
        #     rpn_post_nms_top_n,
        #     rpn_nms_thresh,
        #     score_thresh=0.0
        # )
    
        self.training= training

        self.clip = CLIP()

        frcnn_params = params["fcrnn_head"]
        layer_list = []
        flatten_size = self.model.backbone.out_channels*self.model.roi_heads.box_roi_pool.output_size[0]**2
        # input size: [B,N_proposal,256,7,7]
        for i in range(frcnn_params["num_convs"]):
            layer_list.append(nn.Conv2d(256, frcnn_params["num_filters"], kernel_size=(3,3), stride=1, padding=(1,1)))
        layer_list.append(nn.Flatten(start_dim=-3))
        input_size = flatten_size
        for i in range(frcnn_params["num_fcs"]):
            layer_list.append(nn.Linear(input_size,frcnn_params["fc_dims"]))
            input_size = frcnn_params["fc_dims"]
        
        layer_list.append(nn.BatchNorm1d())

        self.region = nn.Sequential( *layer_list )

        self.vild_projection = nn.Sequential(
            nn.Linear(input_size, frcnn_params["clip_dim"])
        )

        self.background = nn.parameter(torch.rand(1,512))
    
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

        if isinstance(images, torch.Tensor): # convert tensors into list[Tensor]
            images = [i for i in images]

        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")
        
        original_image_sizes = [] # list[Tuple[int,int]]
        for image in images:
            shape = image.shape
            original_image_sizes.append(shape[-2], shape[-1])

        images, targets = self.model.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )
        
        features = self.model.backbone(images.tensors) # feature map of whole image
        proposals, proposal_losses = self.model.rpn(images, features, targets)  # proposal:list[Tensor[1000, 4]]
        # objectness, pred_bbox_deltas = self.model.rpn.head(features) #from rpn.py forward()

        box_features = self.model.roi_heads.box_roi_pool(features, proposals, images.image_sizes) #[B* 1000, 256, 7, 7]
        region_embedding = self.region(box_features) #[B*1000, 512]
        region_embedding = region_embedding.reshape(-1, len(proposals[0]), self.params["frcnn_params"]["fc_dims"]) #reshape it to [B, 1000, 512]

        original_proposals = self.process_proposal(images, proposals, original_image_sizes )

        text_embedding = self.clip(categories, embedding="text")
        text_embedding = torch.tensor(text_embedding)
        cls_logits = self.sim(region_embedding, text_embedding) #[B, 1000, Categories]
        result = self.postprocess(cls_logits, original_proposals, original_image_sizes)
        return result


        # detections, detector_losses = self.model.roi_heads(features, proposals, images.image_sizes, targets)
        # detections = self.model.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

    def process_proposal(self, images, proposals, original_image_sizes):
        proposals = torch.stack(proposals, dim=0) # list[Tensor] -> tensor [B,1000, 4]
        image_sizes = images.image_sizes
        ratios = torch.tensor([
            (torch.tensor(s_org[0], dtype=torch.float32, device=proposals.device)
            / torch.tensor(s[0], dtype=torch.float32, device=proposals.device), 
            torch.tensor(s_org[1], dtype=torch.float32, device=proposals.device)
            / torch.tensor(s[1], dtype=torch.float32, device=proposals.device)) 
            for s, s_org in zip(image_sizes, original_image_sizes)
        ])
        ratios_height, ratios_width = ratios.permute(1, 0) # transpose matrix
        xmin, ymin, xmax, ymax = proposals.unbind(2)
        xmin = xmin * ratios_width
        ymin = ymin * ratios_height
        xmax = xmax * ratios_width
        ymax = ymax * ratios_height
        return torch.stack((xmin,ymin,xmax,ymax), dim=2)

    def sim(self, region_embedding, text_embedding):
        NumBbox= region_embedding.shape[1] # Number of Bbox per image
        NumCls = text_embedding.shape[0] # Number of Categories 
        ind = []
        for i in range(NumBbox):
            for j in range(NumCls):
                ind.append(i+j*NumBbox)
        region_embeddingR = region_embedding.repeat((1,NumCls,1))[:,ind,:]  
        text_embeddingR = text_embedding.repeat((NumBbox,1))
        sim = nn.CosineSimilarity(dim=-1)
        result = sim(region_embeddingR, text_embeddingR)
        return result.reshape(-1,NumBbox,NumCls)
    
    def postprocess(self, cls_logits, proposals, original_image_sizes):
        """
        cls_logits: Tensor[B, 1000, Categories]
        proposals: [B, 1000, 4]
        original_image_sizes: [B, 4]
        """
        scores = torch.nn.functional.softmax(cls_logits, -1)
        device = cls_logits.device
        num_class = cls_logits.shape[-1]
        result=[]
        proposals_x = proposals[... , 0::2] #[B, 1000, 2]
        proposals_y = proposals[... , 1::2] #[B, 1000, 2]
        p_list=[]
        for p_x, p_y, img_s in zip(proposals_x, proposals_y, original_image_sizes):
            height, width =img_s
            p_x = p_x.clamp(min=0, max=width)  #[1000, 2]=[x1, x2]
            p_y = p_y.clamp(min=0, max=height) #[1000, 2]=[y1, y2]
            proposals_C = torch.stack([p_x, p_y], p_x.dim()) #[1000, 2, new_axis] -> p_C[0]=[[x1,y1],[x2,y2]]
            proposal_C = proposals_C.reshape(-1, 4)
            p_list.append(proposal_C)
            # expend_ind = torch.arange
        
        # duplicate the class-agnostic proposals for each catergories
        proposals = torch.stack(p_list, dim=0)
        ind = torch.arange(proposals.shape[1]).reshape(-1, 1).repeat(1, num_class-1).reshape(-1)
        proposals = proposals[:, ind, :] 
        all_boxes = []
        all_scores = []
        all_labels = []
        for proposals_per_image, scores_per_image in zip(proposals, scores):
            labels = torch.arange(num_class, device=device)
            labels = labels.view(1, -1).expand_as(scores_per_image)

            # proposals_per_image = proposals_per_image[:, 1:]
            scores_per_image = scores_per_image[:, 1:]
            labels = labels[:, 1:]
            # batch everything, making every class prediction a separate instance
            proposals_per_image = proposals_per_image.reshape(-1, 4)
            scores_per_image = scores_per_image.reshape(-1)
            labels = labels.reshape(-1)
            # Threshold
            inds = torch.where(scores_per_image > self.model.roi_heads.score_thresh)[0] 
            proposals_per_image, scores_per_image, labels = proposals_per_image[inds], scores_per_image[inds], labels[inds]

            # Empty box
            x_min, y_min, x_max, y_max = proposals_per_image[:, 0], proposals_per_image[:, 1], proposals_per_image[:, 2], proposals_per_image[:, 3]
            area = (x_max-x_min)*(y_max-y_min)
            inds2 = torch.where(area > 1e2)[0]
            proposals_per_image, scores_per_image, labels = proposals_per_image[inds2], scores_per_image[inds2], labels[inds2]

            # NMS by categories
            proposals_per_image, scores_per_image, labels = self.NMS(proposals_per_image, scores_per_image, labels, 0.5)
            
            all_boxes.append(proposals_per_image)
            all_scores.append(scores_per_image)
            all_labels.append(labels)
        
        for i in range(len(all_boxes)):
            result.append({
                "boxes": all_boxes[i],
                "labels": all_labels[i],
                "scores": all_scores[i]
            })
        return result
    
    def NMS(self, proposals, scores, labels, box_nms_thresh):
        labels_uniq = torch.unique(labels)
        keep_p = []
        keep_s = []
        keep_l = []
        for l in labels_uniq:
            temp=[]
            # select by class
            ind_label = torch.where(labels == l)[0]
            p_selected,s_selected=proposals[ind_label], scores[ind_label]
            Xmin, Ymin, Xmax, Ymax = p_selected.unbind(1)
            Area = (Xmax-Xmin)* (Ymax-Ymin)

            # sort by socre
            orders = s_selected.argsort(descending=True)
            # p_selected, r_selected, l_selected = p_selected[orders], r_selected[orders], l_selected[orders]

            while (len(orders) > 0):
                i = orders[0]
                temp.append(i)
                xmin = torch.max(Xmin[i], Xmin[orders[1:]])
                xmax = torch.min(Xmax[i], Xmax[orders[1:]])
                ymin = torch.max(Ymin[i], Ymin[orders[1:]])
                ymax = torch.min(Ymax[i], Ymax[orders[1:]])

                w = torch.max(torch.tensor(0), xmax - xmin)
                h = torch.max(torch.tensor(0), ymax - ymin)
                inter = w*h

                IoU = inter/(Area[i] + Area[orders[1:]] - inter)
                ind_iou = torch.where(IoU <= box_nms_thresh)[0]
                temp_iou = orders[ind_iou]
                orders = temp_iou
            keep_p.append(p_selected[temp])
            keep_s.append(s_selected[temp]) 
            keep_l.append(torch.tensor([l]*ind_label.shape[0]))
        
        keep_p = torch.cat(keep_p, dim=0).reshape(-1, 4)
        keep_s = torch.cat(keep_s, dim=0).reshape(-1)
        keep_l = torch.cat(keep_l, dim=0).reshape(-1)
        return keep_p, keep_s, keep_l

# def _default_anchorgen():
#     anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
#     aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
#     return AnchorGenerator(anchor_sizes, aspect_ratios)      
