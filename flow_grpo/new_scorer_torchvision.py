import os
import cv2
import numpy as np
import torchvision.transforms as T
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import base64
import requests
import json
from io import BytesIO
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.regression import MeanSquaredError

PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


class ImageMetrics:
    """A class to compute image quality metrics including SSIM, PSNR, and MSE."""
    def __init__(self, device="cuda"):
        self.device = device
        # background preservation
        self.psnr_metric_calculator = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
        self.mse_metric_calculator = MeanSquaredError().to(device)
        self.ssim_metric_calculator = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    @staticmethod
    def load_image(image_path, target_size=(224, 224)):
        """Load and resize an image."""
        image = Image.open(image_path)
        image = image.resize(target_size, Image.LANCZOS)
        image = np.array(image)
        return image
    
    @staticmethod
    def normalize_image(image):
        """Normalize image to [0, 1] range."""
        return np.array(image, dtype=np.float32) / 255.0
    
    @staticmethod
    def align_images(original, edited):
        """Align two images using SIFT features and affine transformation."""
        gray1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        k1, d1 = sift.detectAndCompute(gray1, None)
        k2, d2 = sift.detectAndCompute(gray2, None)
        if d1 is None or d2 is None:
            return None
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(d1, d2, k=2)
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]
        if len(good) < 4:
            return None
        src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.LMEDS)
        if matrix is None:
            return None
        h, w = original.shape[:2]
        return cv2.warpAffine(edited, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # 2. PSNR
    def calculate_psnr(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt

        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        score = self.psnr_metric_calculator(img_pred_tensor, img_gt_tensor)
        score = score.cpu().item()
        return score

    # 3. LPIPS
    def calculate_lpips(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt

        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        score = self.lpips_metric_calculator(img_pred_tensor * 2 - 1, img_gt_tensor * 2 - 1)
        score = score.cpu().item()
        return score

    # 4. MSE
    def calculate_mse(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt

        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).to(self.device)
        score = self.mse_metric_calculator(img_pred_tensor.contiguous(), img_gt_tensor.contiguous())
        score = score.cpu().item()
        return score

    # 5. SSIM
    def calculate_ssim(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt

        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        score = self.ssim_metric_calculator(img_pred_tensor, img_gt_tensor)
        score = score.cpu().item()
        return score


class VIEScore:
    """VIEScore evaluation class for AI image generation quality assessment using OpenAI API."""
    
    def __init__(self, api_key=None, model_name="gpt-4o"):
        """
        Initialize VIEScore evaluator.
        
        Args:
            api_key: OpenAI API key
            model_name: Model name for OpenAI API (default: gpt-4o)
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Provide via parameter or OPENAI_API_KEY environment variable.")
        
        self.url = "https://api.openai.com/v1/chat/completions"
        self.model_name = model_name
        
        # VIE prompts for TIE (Two Image Edit) evaluation
        self.context = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

You will have to give your output in this way (Keep your reasoning concise and short.):
{
"score" : [...],
"reasoning" : "..."
}"""
        
        self.tie_rule = """RULES:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
The objective is to evaluate how successfully the editing instruction has been executed in the second image.

Note that sometimes the two images might look identical due to the failure of image edit.

From scale 0 to 10: 
A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

Editing instruction: <instruction>"""
        
        self.pq_rule = """RULES:

The image is an AI-generated image.
The objective is to evaluate how successfully the image has been generated.

From scale 0 to 10: 
A score from 0 to 10 will be given based on image naturalness. 
(
    0 indicates that the scene in the image does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting. 
    10 indicates that the image looks natural.
)
A second score from 0 to 10 will rate the image artifacts. 
(
    0 indicates that the image contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized. 
    10 indicates the image has no artifacts.
)
Put the score in a list such that output score = [naturalness, artifacts]"""

    def encode_image(self, image_path):
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def prepare_prompt(self, image_paths, text_prompt):
        """Prepare prompt content with images and text."""
        content = [{"type": "text", "text": text_prompt}]
        
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
            
        for image_path in image_paths:
            if os.path.exists(image_path):
                base64_image = self.encode_image(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
        
        return content
    
    def get_parsed_output(self, prompt_content):
        """Send request to OpenAI API and get response."""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt_content}],
            "max_tokens": 1400
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.post(self.url, json=payload, headers=headers)
            response_data = response.json()
            
            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content']
            else:
                print(f"API Error: {response_data}")
                return None
                
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return None
    
    def parse_score(self, response_text):
        """Parse score from API response."""
        try:
            # Find JSON-like content in response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                return result.get('score', []), result.get('reasoning', '')
            
        except json.JSONDecodeError:
            pass
        
        return [], ""
    
    def evaluate_tie(self, original_path, edited_path, instruction):
        """
        Evaluate Two Image Edit (TIE) task.
        
        Args:
            original_path: Path to original image
            edited_path: Path to edited image  
            instruction: Editing instruction text
            
        Returns:
            Dictionary containing scores and reasoning
        """
        if not os.path.exists(original_path) or not os.path.exists(edited_path):
            return None
        
        # Prepare prompt
        prompt_text = "\n".join([self.context, self.tie_rule.replace("<instruction>", instruction)])
        prompt_content = self.prepare_prompt([original_path, edited_path], prompt_text)
        
        # Get API response
        response = self.get_parsed_output(prompt_content)
        if not response:
            return None
        
        # Parse scores
        scores, reasoning = self.parse_score(response)
        
        return {
            'tie_scores': scores,
            'tie_reasoning': reasoning,
            'editing_success': scores[0] if len(scores) > 0 else None,
            'overediting_score': scores[1] if len(scores) > 1 else None
        }
    
    def evaluate_pq(self, image_path):
        """
        Evaluate Perceptual Quality (PQ) of single image.
        
        Args:
            image_path: Path to image
            
        Returns:
            Dictionary containing naturalness and artifacts scores
        """
        if not os.path.exists(image_path):
            return None
        
        # Prepare prompt
        prompt_text = "\n".join([self.context, self.pq_rule])
        prompt_content = self.prepare_prompt([image_path], prompt_text)
        
        # Get API response
        response = self.get_parsed_output(prompt_content)
        if not response:
            return None
        
        # Parse scores
        scores, reasoning = self.parse_score(response)
        
        return {
            'pq_scores': scores,
            'pq_reasoning': reasoning,
            'naturalness': scores[0] if len(scores) > 0 else None,
            'artifacts': scores[1] if len(scores) > 1 else None
        }


def compute_image_metrics(original_path, edited_path, mask_path, vie_instruction=None, api_key=None):
    """
    Compute all image metrics (SSIM, PSNR, MSE, VIEScore) for a pair of images.
    
    Args:
        original_path: Path to the original image
        edited_path: Path to the edited image
        object_mask: Optional mask array to exclude certain areas
        vie_instruction: Optional editing instruction for VIEScore evaluation
        api_key: Optional OpenAI API key for VIEScore evaluation
        
    Returns:
        Dictionary containing all computed metrics
    """
    image_processor = VaeImageProcessor(vae_scale_factor=8)
    ImageMetrics_instance = ImageMetrics(device="cpu")

    original_image = image_processor_flux_context(image_processor, original_path, if_mask=False)
    edited_image = image_processor_flux_context(image_processor, edited_path, if_mask=False).resize((original_image.size[0], original_image.size[1]), resample=Image.BICUBIC)
    mask_image = image_processor_flux_context(image_processor, mask_path, if_mask=True)

    assert original_image.size == edited_image.size, "Image shapes should be the same."
    assert original_image.size == mask_image.size, "Image shapes should be the same."

    # process mask
    mask_image = np.asarray(mask_image, dtype=np.int64) / 255
    mask_image = 1 - mask_image
    mask_image = mask_image[:, :, np.newaxis].repeat([3], axis=2)
        
    metrics = {
        'ssim': ImageMetrics_instance.calculate_ssim(original_image, edited_image, mask_image),
        'psnr': ImageMetrics_instance.calculate_psnr(original_image, edited_image, mask_image), 
        'mse': ImageMetrics_instance.calculate_mse(original_image, edited_image, mask_image)
    }
    
    # Add VIEScore evaluation if instruction is provided
    if vie_instruction and (api_key or os.environ.get('OPENAI_API_KEY')):
        try:
            vie_scorer = VIEScore(api_key=api_key)
            vie_results = vie_scorer.evaluate_tie(original_path, edited_path, vie_instruction)
            if vie_results:
                metrics.update(vie_results)
        except Exception as e:
            print(f"VIEScore evaluation failed: {e}")
            metrics['vie_error'] = str(e)
    
    return metrics


def compute_single_image_quality(image_path, api_key=None):
    """
    Compute perceptual quality metrics for a single image using VIEScore.
    
    Args:
        image_path: Path to the image
        api_key: Optional OpenAI API key for VIEScore evaluation
        
    Returns:
        Dictionary containing perceptual quality scores
    """
    if not os.path.exists(image_path):
        return None
    
    try:
        vie_scorer = VIEScore(api_key=api_key)
        results = vie_scorer.evaluate_pq(image_path)
        return results
    except Exception as e:
        print(f"Single image quality evaluation failed: {e}")
        return {'vie_error': str(e)}

def image_processor_flux_context(image_processor, image_path, _auto_resize=True, if_mask=False):
    """
    Get the context of the image processor for Flux.
    """

    multiple_of = image_processor.vae_scale_factor * 2

    if image_path is not None:
        if if_mask:
            image = Image.open(image_path).convert("L")
        else:
            image = Image.open(image_path).convert("RGB")

    image_height, image_width = image_processor.get_default_height_width(image)
    aspect_ratio = image_width / image_height
    if _auto_resize:
        # Kontext is trained on specific resolutions, using one of them is recommended
        _, image_width, image_height = min(
            (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
        )
    image_width = image_width // multiple_of * multiple_of
    image_height = image_height // multiple_of * multiple_of

    if if_mask:
        image = image.resize((image_width, image_height), resample=Image.NEAREST)
    else:
        image = image.resize((image_width, image_height), resample=Image.BICUBIC)

    return image

if __name__ == "__main__":
    from diffusers.image_processor import VaeImageProcessor
    image_processor = VaeImageProcessor(vae_scale_factor=8)

    original_path = "/Users/ljq/Downloads/020.png"
    edited_path = "/Users/ljq/Downloads/image_ovis.webp"
    mask_path = "/Users/ljq/Downloads/020_mask.jpg"

    result = compute_image_metrics(original_path, edited_path, mask_path)

    print(result)

    ##check if the mask is correct, show masked image
    
    def create_mask_visualization(original_image, mask_image):
        """
        Create a visualization combining original image, mask, and masked result
        
        Args:
            original_image: PIL Image of original image
            mask_image: PIL Image of mask
        
        Returns:
            PIL Image with three images concatenated horizontally
        """
        from PIL import Image
        import numpy as np
        
        # Convert PIL Images to numpy
        original_np = np.array(original_image) / 255.0  # Normalize to [0,1]
        mask_np = np.array(mask_image) / 255.0
        
        # Ensure mask is grayscale (single channel)
        if len(mask_np.shape) == 3:
            mask_np = mask_np.mean(axis=2)  # Convert to grayscale if RGB
        
        # Create masked result
        masked_result = original_np * mask_np[..., np.newaxis]
        
        # Convert to PIL Images
        original_pil = Image.fromarray((np.clip(original_np, 0, 1) * 255).astype(np.uint8))
        
        # Handle mask - convert to 3-channel for consistency
        mask_3ch = np.stack([mask_np, mask_np, mask_np], axis=-1)
        mask_pil = Image.fromarray((mask_3ch * 255).astype(np.uint8))
        
        masked_pil = Image.fromarray((np.clip(masked_result, 0, 1) * 255).astype(np.uint8))
        
        # Get dimensions
        width, height = original_pil.size
        
        # Create concatenated image (horizontal)
        total_width = width * 3
        concatenated = Image.new('RGB', (total_width, height))
        
        # Paste images side by side
        concatenated.paste(original_pil, (0, 0))
        concatenated.paste(mask_pil, (width, 0))
        concatenated.paste(masked_pil, (width * 2, 0))
        
        return concatenated
    
    # Create and show the visualization
    viz_image = create_mask_visualization(original_image, mask_image)
    viz_image.show()
    
