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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps

def retry_on_failure(max_retries=3, delay=1.0, backoff=2.0):
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplicative factor for delay
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        print(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {retry_delay:.1f}s...")
                        time.sleep(retry_delay)
                        retry_delay *= backoff
                    else:
                        print(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator


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
                # Resize mask to match image dimensions (224, 224)
                if len(object_mask.shape) == 3:
                    object_mask = object_mask.mean(axis=2)  # Convert to grayscale
                mask_image = Image.fromarray((object_mask * 255).astype(np.uint8))
                mask_resized = mask_image.resize((224, 224), Image.LANCZOS)
                object_mask = np.array(mask_resized) / 255.0
                object_mask = object_mask.reshape(image.shape)
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


@retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
def process_single_image_metrics(args):
    """
    Process metrics for a single image pair with retry mechanism.
    
    Args:
        args: Tuple containing (i, ref_img, edited_img, mask_img, prompt, api_key, image_processor)
    
    Returns:
        Dictionary containing computed metrics for the image pair
    """
    i, ref_img, edited_img, mask_img, prompt, api_key, image_processor = args
    
    import tempfile
    
    try:
        # Process images to kontext-compatible size using the unified function
        ref_resized = image_processor_flux_context(image_processor, ref_img, _auto_resize=True, if_mask=False)
        edited_resized = image_processor_flux_context(image_processor, edited_img, _auto_resize=True, if_mask=False)
        mask_resized = image_processor_flux_context(image_processor, mask_img, _auto_resize=True, if_mask=True)
        
        # Assert all images have the same size
        assert ref_resized.size == edited_resized.size == mask_resized.size, \
            f"Image sizes don't match: ref={ref_resized.size}, edited={edited_resized.size}, mask={mask_resized.size}"
        
        # Create temporary files for metric calculations
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as ref_tmp, \
             tempfile.NamedTemporaryFile(suffix='.png', delete=False) as edited_tmp:
            
            ref_resized.save(ref_tmp.name)
            edited_resized.save(edited_tmp.name)
            
            # Convert mask to numpy array for metric calculations
            mask_array = np.array(mask_resized)
            if len(mask_array.shape) == 3:
                mask_array = mask_array[:, :, 0]  # Convert to grayscale if needed
            
            # Calculate metrics
            metrics = {
                'image_index': i,
                'target_size': ref_resized.size,
                'mse': ImageMetrics.calculate_mse(ref_tmp.name, edited_tmp.name, mask_array),
                'psnr': ImageMetrics.calculate_psnr(ref_tmp.name, edited_tmp.name, mask_array),
                'ssim': ImageMetrics.calculate_ssim(ref_tmp.name, edited_tmp.name, mask_array),
            }
            
            # Add VIEScore evaluation
            if api_key and prompt:
                try:
                    vie_scorer = VIEScore(api_key=api_key)
                    vie_tie_results = vie_scorer.evaluate_tie(ref_tmp.name, edited_tmp.name, prompt)
                    vie_pq_results = vie_scorer.evaluate_pq(edited_tmp.name)
                    if vie_tie_results:
                        metrics.update(vie_tie_results)
                    if vie_pq_results:
                        metrics.update(vie_pq_results)
                except Exception as e:
                    print(f"VIEScore evaluation failed for image {i}: {e}")
                    metrics['vie_error'] = str(e)
            
            # Clean up temporary files
            os.unlink(ref_tmp.name)
            os.unlink(edited_tmp.name)
            
            return metrics
            
    except Exception as e:
        print(f"Error processing image {i}: {e}")
        return {
            'image_index': i,
            'error': str(e)
        }


def compute_image_metrics_pil_batch(ref_images, edited_images, mask_images, prompts, api_key, max_workers=4):
    """
    Compute all image metrics (SSIM, PSNR, MSE, VIEScore) for batches of PIL images using multithreading.
    
    Args:
        ref_images: List of PIL.Image objects (reference images)
        edited_images: List of PIL.Image objects (edited images)  
        mask_images: List of PIL.Image objects (mask images)
        prompts: List of str (editing instructions)
        api_key: str (OpenAI API key for VIEScore evaluation)
        max_workers: int (maximum number of worker threads, default=4)
        
    Returns:
        List of dictionaries containing computed metrics for each image pair
    """
    from diffusers.image_processor import VaeImageProcessor
    
    # Initialize image processor
    image_processor = VaeImageProcessor(vae_scale_factor=8)
    
    # Prepare arguments for each worker
    worker_args = [
        (i, ref_img, edited_img, mask_img, prompt, api_key, image_processor)
        for i, (ref_img, edited_img, mask_img, prompt) in enumerate(zip(ref_images, edited_images, mask_images, prompts))
    ]
    
    results = [None] * len(worker_args)  # Pre-allocate results list to maintain order
    
    # Process images in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_image_metrics, args): args[0] 
            for args in worker_args
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
                print(f"Completed processing image {index}")
            except Exception as e:
                print(f"Failed to process image {index}: {e}")
                results[index] = {
                    'image_index': index,
                    'error': str(e)
                }
    
    return results


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

def image_processor_flux_context(image_processor, image_input, _auto_resize=True, if_mask=False):
    """
    Get the context of the image processor for Flux.
    
    Args:
        image_processor: VaeImageProcessor instance
        image_input: str (image path) or PIL.Image object
        _auto_resize: bool, whether to auto resize to kontext resolutions
        if_mask: bool, whether this is a mask image
    
    Returns:
        PIL.Image: Processed image
    """

    multiple_of = image_processor.vae_scale_factor * 2

    # Handle both path string and PIL.Image input
    if image_input is not None:
        if isinstance(image_input, str):
            # Input is a file path
            if if_mask:
                image = Image.open(image_input).convert("L")
            else:
                image = Image.open(image_input).convert("RGB")
        else:
            # Input is already a PIL.Image
            if if_mask:
                image = image_input.convert("L")
            else:
                image = image_input.convert("RGB")
    else:
        raise ValueError("image_input cannot be None")

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
    
    # Initialize
    api_key = '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'
    
    # Test file paths
    original_path = "/Users/ljq/Downloads/new.jpg"
    edited_path = "/Users/ljq/Downloads/1755608073.png"
    edited_ovis_path = "/Users/ljq/Downloads/1755608073.png"
    mask_path = "/Users/ljq/Downloads/020_mask.jpg"
    
    print("=== Testing New Batch Processing Function ===")
    
    try:
        # Load PIL images for batch processing
        ref_images = [
            Image.open(original_path),
            Image.open(original_path)  # Using same image twice for demo
        ]
        
        edited_images = [
            Image.open(edited_path),
            Image.open(edited_ovis_path)
        ]
        
        mask_images = [
            Image.open(mask_path),
            Image.open(mask_path)  # Using same mask twice for demo
        ]
        
        prompts = [
            "Change the middle bird to a chicken",
            "Change the middle bird to a chicken"
        ]
        
        print(f"Processing {len(ref_images)} image pairs...")
        print(f"Original image sizes: {[img.size for img in ref_images]}")
        print(f"Edited image sizes: {[img.size for img in edited_images]}")
        print(f"Mask image sizes: {[img.size for img in mask_images]}")
        
        # Test batch processing with multithreading
        import time
        start_time = time.time()
        
        results = compute_image_metrics_pil_batch(
            ref_images=ref_images,
            edited_images=edited_images, 
            mask_images=mask_images,
            prompts=prompts,
            api_key=api_key,
            max_workers=2  # Use 2 workers for this small test
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        print(f"Average time per image: {processing_time/len(ref_images):.2f} seconds")
        
        # Display results
        for i, result in enumerate(results):
            print(f"\n--- Results for Image Pair {i} ---")
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Target size: {result['target_size']}")
                print(f"SSIM: {result.get('ssim', 'N/A'):.4f}" if result.get('ssim') is not None else "SSIM: Failed")
                print(f"PSNR: {result.get('psnr', 'N/A'):.4f}" if result.get('psnr') is not None else "PSNR: Failed") 
                print(f"MSE: {result.get('mse', 'N/A'):.4f}" if result.get('mse') is not None else "MSE: Failed")
                
                # VIEScore results
                if 'editing_success' in result:
                    print(f"VIE Editing Success: {result['editing_success']}")
                    print(f"VIE Overediting Score: {result['overediting_score']}")
                if 'naturalness' in result:
                    print(f"VIE Naturalness: {result['naturalness']}")
                    print(f"VIE Artifacts: {result['artifacts']}")
                if 'vie_error' in result:
                    print(f"VIE Error: {result['vie_error']}")
        
        print("\n=== Batch Processing Test Completed ===")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Testing Single Image Processing (Legacy) ===")
    
    # Legacy single image test for comparison
    try:
        image_processor = VaeImageProcessor(vae_scale_factor=8)
        
        # Test the updated image_processor_flux_context function with PIL images
        original_pil = Image.open(original_path)
        mask_pil = Image.open(mask_path)
        
        original_processed = image_processor_flux_context(image_processor, original_pil, if_mask=False)
        mask_processed = image_processor_flux_context(image_processor, mask_pil, if_mask=True)
        
        print(f"Original processed size: {original_processed.size}")
        print(f"Mask processed size: {mask_processed.size}")
        
        print("Single image processing test completed successfully")
        
    except Exception as e:
        print(f"Single image test failed: {e}")
        import traceback
        traceback.print_exc()

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
    
    # # Create and show the visualization
    # viz_image = create_mask_visualization(original_image, mask_image)
    # viz_image.show()
    
