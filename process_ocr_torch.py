import torch
import os

from process_ocr_base import OCR_Processer

class OCR_torch_Processer(OCR_Processer):
    def __init__(self, model_size='xl'):
        super().__init__()
        from models.detector import TextDetectorModel, CenterNetDetector
        from models.transformer import ModelDimensions, Transformer, TransformerPredictor

        model = TextDetectorModel(model_size=model_size)
        if os.path.exists('model.pt'):
            data = torch.load('model.pt', map_location="cpu", weights_only=True)
            model.load_state_dict(data['model_state_dict'])

        detector = CenterNetDetector(model.detector)
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        device = torch.device(device)
        detector.to(device=device)
        detector.eval()
        self.detector = detector
        self.device = device

        if os.path.exists('model3.pt'):
            data = torch.load('model3.pt', map_location="cpu", weights_only=True)
            config = ModelDimensions(**data['config'])
            model = Transformer(**config.__dict__)
            model.load_state_dict(data['model_state_dict'])
        else:
            config = ModelDimensions()
            model = Transformer(**config.__dict__)
        model2 = TransformerPredictor(model.encoder, model.decoder)
        model2.to(device)
        model2.eval()
        self.transformer = model2

    def call_detector(self, image_input):
        images = torch.from_numpy(image_input / 255.).permute(0,3,1,2).to(device=self.device)
        with torch.no_grad():
            heatmap, features = self.detector(images)
            heatmap = heatmap.cpu().numpy()
            features = features.cpu().numpy()
        return heatmap, features

    def call_transformer(self, encoder_input):
        encoder_input = torch.tensor(encoder_input, device=self.device)
        pred = self.transformer(encoder_input).squeeze(0).cpu().numpy()
        return pred