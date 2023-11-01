'''
custom handler 만들기

Custom handlers

Customize the behavior of TorchServe by writing a Python script that you package with the model when you use the model archiver. TorchServe executes this code when it runs.

Provide a custom script to:

    Initialize the model instance
    Pre-process input data before it is sent to the model for inference or captum explanations
    Customize how the model is invoked for inference or explanations
    Post-process output from the model before sending the response to the user

Following is applicable to all types of custom handlers

    data - The input data from the incoming request
    context - Is the TorchServe context. You can use following information for customizaton model_name, model_dir, manifest, batch_size, gpu etc.
'''

import base64

import cv2
import numpy as np
import torch
from packaging import version
from torchvision import __version__ as torchvision_version
from ts.torch_handler.vision_handler import VisionHandler


class CustomHandler(VisionHandler):

    def __init__(self):

        #super(CustomHandler, self).__init__()
        super().__init__()
        self.originheight = None
        self.originwidth = None

        self.targetheight = 512
        self.targetwidth = 512

    def initialize(self, context):
        super().initialize(context)

        # Torchvision breaks with object detector models before 0.6.0
        if version.parse(torchvision_version) < version.parse("0.6.0"):
            self.initialized = False
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            self.initialized = True

    def preprocess(self, data):

        """
        Args:
            data (List): Input data from the request is in the form of a Tensor
        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")

            if isinstance(image, str):

                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = np.frombuffer(image, dtype=np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                self.originheight, self.originwidth, _ = image.shape

                image = cv2.resize(image, dsize=(self.targetwidth, self.targetheight), interpolation=cv2.INTER_AREA)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = torch.FloatTensor(image)
            else:
                raise NotImplementedError

            images.append(image)
        return torch.stack(images).to(self.device)

    def result_processing(self, ids, scores, bboxes):

        # 1. 내림차순 정렬
        indice = scores.argsort(dim=0, descending=True)[:, 0]  # 내림차순 정렬

        ids = ids[indice]
        scores = scores[indice]

        xmin = bboxes[:, 0:1][indice]
        ymin = bboxes[:, 1:2][indice]
        xmax = bboxes[:, 2:3][indice]
        ymax = bboxes[:, 3:4][indice]

        # 2. 배경 제외
        index = ids!=-1
        ids = ids[index]
        scores = scores[index]
        xmin = xmin[index]
        ymin = ymin[index]
        xmax = xmax[index]
        ymax = ymax[index]

        return ids, scores, xmin, ymin, xmax, ymax

    def postprocess(self, data):
        '''
        data : tuple
            topk class id shape : torch.Size([1, 100, 1])
            topk class scores shape : torch.Size([1, 100, 1])
            topk box predictions shape : torch.Size([1, 100, 4])
            topk landmark predictions shape : torch.Size([1, 100, 10])
        result : json
        [
          {
            "faces": [
              167.4222869873047,
              57.03825378417969,
              301.305419921875,
              436.68682861328125
            ],
            "score": 0.9995299577713013
          },
          {
            "faces": [
              89.61490631103516,
              64.8980484008789,
              191.40206909179688,
              446.6605224609375
            ],
            "score": 0.9995074272155762
          }
        ]
        '''
        ids, scores, bboxes, _ = data

        if self.mapping is not None and not isinstance(self.mapping, dict):
            raise Exception('Mapping must be a dict')

        results = []
        for ids_, scores_, bboxes_ in zip(ids, scores, bboxes):
            result = []
            ids_pc, scores_pc, xmin, ymin, xmax, ymax = self.result_processing(ids_, scores_, bboxes_)

            # from target image box size to origin image box size
            x_scale = self.originwidth / self.targetwidth
            y_scale = self.originheight / self.targetheight

            xmin = x_scale * xmin
            xmax = x_scale * xmax
            ymin = y_scale * ymin
            ymax = y_scale * ymax

            for id_, score_, xmi, ymi, xma, yma in zip(ids_pc.tolist(), scores_pc.tolist(), xmin.tolist(), ymin.tolist(), xmax.tolist(), ymax.tolist()):
                info = {}
                info[self.mapping[str(int(id_))]] = [xmi, ymi, xma, yma]
                info["score"] =  score_
                result.append(info)
            results.append(result)

        return results
