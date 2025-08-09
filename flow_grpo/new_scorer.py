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


class ImageMetrics:
    """A class to compute image quality metrics including SSIM, PSNR, and MSE."""
    
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
    
    @staticmethod
    def calculate_ssim(image_path, edit_image_path, object_mask=None):
        """
        Calculate SSIM (Structural Similarity Index Measure) between two images.
        
        Args:
            image_path: Path to the original image
            edit_image_path: Path to the edited image
            object_mask: Optional mask to exclude certain areas from calculation
            
        Returns:
            SSIM score (float)
        """
        try:
            image = ImageMetrics.load_image(image_path)
            edit_image = ImageMetrics.load_image(edit_image_path)
            
            # Convert to grayscale
            image = T.Grayscale()(Image.fromarray(image))
            edit_image = T.Grayscale()(Image.fromarray(edit_image))
            
            # Normalize
            image = ImageMetrics.normalize_image(image)
            edit_image = ImageMetrics.normalize_image(edit_image)
            
            # Apply mask if provided
            if object_mask is not None:
                object_mask = np.array(object_mask)
                rest_mask = np.ones(object_mask.shape, dtype=object_mask.dtype) - object_mask
                image = image * rest_mask
                edit_image = edit_image * rest_mask
            
            # Compute SSIM
            sim, _ = ssim(image, edit_image, data_range=1.0, full=True)
            return float(sim)
            
        except Exception as e:
            print(f"Error calculating SSIM: {e}")
            return None
    
    @staticmethod
    def calculate_psnr(image_path, edit_image_path, object_mask=None):
        """
        Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.
        
        Args:
            image_path: Path to the original image
            edit_image_path: Path to the edited image  
            object_mask: Optional mask to exclude certain areas from calculation
            
        Returns:
            PSNR score (float)
        """
        try:
            original = cv2.imread(image_path)
            edited = cv2.imread(edit_image_path)
            
            if original is None or edited is None:
                return None
            
            # Align images
            aligned = ImageMetrics.align_images(original, edited)
            if aligned is None:
                aligned = edited
                
            # Resize if needed
            if aligned.shape != original.shape:
                aligned = cv2.resize(aligned, (original.shape[1], original.shape[0]))
            
            # Apply mask if provided
            if object_mask is not None:
                mask = np.array(object_mask, dtype=np.uint8)
                if len(mask.shape) > 2:
                    mask = mask[:, :, 0]
                if mask.shape != original.shape[:2]:
                    mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
                inv_mask = (mask < 128).astype(np.uint8)
                diff = (original.astype(np.float32) - aligned.astype(np.float32)) ** 2
                mse = np.sum(diff * inv_mask[..., None]) / (np.sum(inv_mask) * 3 + 1e-10)
            else:
                mse = np.mean((original.astype(np.float32) - aligned.astype(np.float32)) ** 2)
            
            if mse == 0:
                return float('inf')
            
            psnr = 10 * np.log10((255 ** 2) / mse)
            return float(psnr)
            
        except Exception as e:
            print(f"Error calculating PSNR: {e}")
            return None
    
    @staticmethod
    def calculate_mse(image_path, edit_image_path, object_mask=None):
        """
        Calculate MSE (Mean Squared Error) between two images.
        
        Args:
            image_path: Path to the original image
            edit_image_path: Path to the edited image
            object_mask: Optional mask to exclude certain areas from calculation
            
        Returns:
            MSE score (float)
        """
        try:
            original = cv2.imread(image_path)
            edited = cv2.imread(edit_image_path)
            
            if original is None or edited is None:
                return None
            
            # Align images
            aligned = ImageMetrics.align_images(original, edited)
            if aligned is None:
                aligned = edited
                
            # Resize if needed
            if aligned.shape != original.shape:
                aligned = cv2.resize(aligned, (original.shape[1], original.shape[0]))
            
            # Apply mask if provided
            if object_mask is not None:
                mask = np.array(object_mask, dtype=np.uint8)
                if len(mask.shape) > 2:
                    mask = mask[:, :, 0]
                if mask.shape != original.shape[:2]:
                    mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
                inv_mask = (mask < 128).astype(np.uint8)
                diff = (original.astype(np.float32) - aligned.astype(np.float32)) ** 2
                pixel_count = np.sum(inv_mask)
                if pixel_count == 0:
                    return None
                mse = np.sum(diff * inv_mask[..., None]) / (pixel_count * 3 + 1e-10)
            else:
                mse = np.mean((original.astype(np.float32) - aligned.astype(np.float32)) ** 2)
            
            return float(mse)
            
        except Exception as e:
            print(f"Error calculating MSE: {e}")
            return None


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


def compute_image_metrics(original_path, edited_path, object_mask=None, vie_instruction=None, api_key=None):
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
    if not os.path.exists(original_path) or not os.path.exists(edited_path):
        return None
        
    metrics = {
        'ssim': ImageMetrics.calculate_ssim(original_path, edited_path, object_mask),
        'psnr': ImageMetrics.calculate_psnr(original_path, edited_path, object_mask), 
        'mse': ImageMetrics.calculate_mse(original_path, edited_path, object_mask)
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