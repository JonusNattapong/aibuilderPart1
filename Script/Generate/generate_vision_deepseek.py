import argparse
import os
import requests
import json
import time
import csv
import random
import base64
import torch
from PIL import Image
import io
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from config_generate_vision import (
    VISION_TASKS,
    VISION_TOPICS,
    OUTPUT_DIR,
    MAX_RETRIES,
    RETRY_DELAY,
    NUM_SAMPLES_PER_TASK,
    DEEPSEEK_API_URL,
    DEEPSEEK_MODEL,
    IMAGE_SIZE,
    IMAGE_CHANNELS,
    IMAGE_FORMAT,
    IMAGE_QUALITY,
    ANNOTATION_FORMATS
)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate vision datasets using DeepSeek API.')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Directory to save the generated datasets.')
    parser.add_argument('--max_retries', type=int, default=MAX_RETRIES, help='Maximum number of retries for API requests.')
    parser.add_argument('--retry_delay', type=int, default=RETRY_DELAY, help='Delay between retries in seconds.')
    parser.add_argument('--num_samples_per_task', type=int, default=NUM_SAMPLES_PER_TASK, help='Number of samples to generate per vision task.')
    parser.add_argument('--deepseek_api_url', type=str, default=DEEPSEEK_API_URL, help='DeepSeek API URL.')
    parser.add_argument('--deepseek_model', type=str, default=DEEPSEEK_MODEL, help='DeepSeek model to use.')
    parser.add_argument('--chroma_model', type=str, default='lodestones/Chroma', help='Chroma model to use for vision generation.')
    parser.add_argument('--use_chroma', action='store_true', help='Use Chroma model for generation.')
    return parser.parse_args()

def invoke_deepseek(system_prompt, user_prompt, deepseek_model, deepseek_api_url, max_retries, delay, temperature=0.7, max_tokens=512):
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY environment variable not set.")
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    payload = {
        "model": deepseek_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(deepseek_api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            if result and 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0].get('message', {}).get('content')
                if content:
                    return content.strip()
            print(f"Warning: Unexpected DeepSeek API response: {result}")
        except Exception as e:
            print(f"DeepSeek API request failed (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
    return None

def parse_json_output(llm_output):
    try:
        match = llm_output
        if "```json" in llm_output:
            match = llm_output.split("```json")[-1].split("```")[0]
        return json.loads(match)
    except Exception as e:
        print(f"Error parsing JSON: {e}\nOutput: {llm_output}")
        return None

def save_image(image_data, file_path):
    try:
        # Decode base64 image data
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Resize image if needed
        if image.size != IMAGE_SIZE:
            image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save image
        image.save(file_path, format=IMAGE_FORMAT, quality=IMAGE_QUALITY)
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

def generate_image_with_chroma(prompt, model, tokenizer, size=(512, 512)):
    """Generate image using Chroma model"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        image = model.generate_images(
            inputs.input_ids,
            num_inference_steps=50,
            guidance_scale=7.5,
            height=size[0],
            width=size[1]
        )
    return image[0]  # Returns PIL Image

def generate_vision_samples(task, topics, n, deepseek_model, deepseek_api_url, max_retries, retry_delay, use_chroma=False, chroma_model_name=None):
    # Initialize Chroma model and tokenizer at the function level
    chroma_tokenizer = None
    chroma_model = None
    
    if use_chroma and chroma_model_name:
        print(f"Loading Chroma model: {chroma_model_name}")
        try:
            chroma_tokenizer = AutoTokenizer.from_pretrained(chroma_model_name)
            chroma_model = AutoModelForCausalLM.from_pretrained(chroma_model_name)
            print("Successfully loaded Chroma model and tokenizer")
        except Exception as e:
            print(f"Error loading Chroma model: {e}")
            use_chroma = False
    
    task_prompts = {
        'image_classification': {
            'system': 'สร้างรูปภาพพร้อมระบุประเภทของรูปภาพ ให้มีความหลากหลายและสมจริง',
            'example': '{"image": "base64_encoded_image_data", "label": "cat", "confidence": 0.95}'
        },
        'object_detection': {
            'system': 'สร้างรูปภาพพร้อมระบุตำแหน่งและประเภทของวัตถุในรูปภาพ',
            'example': '{"image": "base64_encoded_image_data", "objects": [{"label": "person", "bbox": [0, 0, 100, 200], "confidence": 0.9}]}'
        },
        # ... [other task prompts remain unchanged]
    }

    system_prompt = (
        f"คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Computer Vision ประเภท {task} "
        f"หน้าที่ของคุณคือ{task_prompts[task]['system']} "
        "โดยใช้หัวข้อที่กำหนดให้ ให้เนื้อหามีความสมจริง หลากหลาย ไม่ซ้ำ และสอดคล้องกับบริบท "
        "ผลลัพธ์ต้องอยู่ในรูปแบบ JSON เท่านั้น เช่น "
        f"{task_prompts[task]['example']}"
    )

    user_prompt_template = (
        "หัวข้อ: {topic}\n"
        f"โปรดสร้างข้อมูลสำหรับงาน {task} โดยใช้หัวข้อนี้ "
        "เนื้อหาต้องสมจริง หลีกเลี่ยงการซ้ำหรือใช้โครงสร้างเดิมๆ"
    )

    samples = []
    for i in range(n):
        topic = random.choice(topics)
        user_prompt = user_prompt_template.format(topic=topic)
        
        output = None
        try:
            if use_chroma and chroma_model and chroma_tokenizer:
                # Generate image using Chroma
                image_prompt = f"Generate a {topic} image for {task}"
                generated_image = generate_image_with_chroma(image_prompt, chroma_model, chroma_tokenizer)
                
                # Save generated image
                image_dir = os.path.join(OUTPUT_DIR, f"{task}_images")
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f"sample_{i+1}.{IMAGE_FORMAT.lower()}")
                generated_image.save(image_path, format=IMAGE_FORMAT, quality=IMAGE_QUALITY)
                
                # Store generated image info
                samples.append([image_path])
                
                # Generate metadata using DeepSeek
                metadata_output = invoke_deepseek(system_prompt, user_prompt, deepseek_model, deepseek_api_url, max_retries, retry_delay)
                if metadata_output:
                    metadata = parse_json_output(metadata_output)
                    if metadata:
                        # Update the last sample with metadata
                        samples[-1].extend(list(metadata.values()))
                    else:
                        print("Failed to parse metadata JSON")
                else:
                    print("Failed to generate metadata")
            else:
                # Use DeepSeek for both image and metadata
                output = invoke_deepseek(system_prompt, user_prompt, deepseek_model, deepseek_api_url, max_retries, retry_delay)
                
            # Process DeepSeek output if available
            if output:
                parsed = parse_json_output(output)
                if parsed:
                    # Process DeepSeek output
                    image_dir = os.path.join(OUTPUT_DIR, f"{task}_images")
                    os.makedirs(image_dir, exist_ok=True)
                    
                    if task == 'image_classification':
                        image_path = os.path.join(image_dir, f"sample_{i+1}.{IMAGE_FORMAT.lower()}")
                        if save_image(parsed.get("image", ""), image_path):
                            samples.append([image_path, parsed.get("label", ""), parsed.get("confidence", 0.0)])
                    elif task == 'object_detection':
                        image_path = os.path.join(image_dir, f"sample_{i+1}.{IMAGE_FORMAT.lower()}")
                        if save_image(parsed.get("image", ""), image_path):
                            for obj in parsed.get("objects", []):
                                samples.append([image_path, obj.get("label", ""), *obj.get("bbox", [0,0,0,0]), obj.get("confidence", 0.0)])
                    elif not use_chroma:  # Only print this if not using Chroma
                        print(f"Failed to parse DeepSeek output: {output}")
                elif not use_chroma:  # Only print this if not using Chroma
                    print("Failed to parse DeepSeek output")
                    
        except Exception as e:
            print(f"Error during generation: {e}")
            continue
        
        if output:
            parsed = parse_json_output(output)
            if parsed:
                # Create subdirectory for images if needed
                image_dir = os.path.join(OUTPUT_DIR, f"{task}_images")
                os.makedirs(image_dir, exist_ok=True)

                if task == 'image_classification':
                    image_path = os.path.join(image_dir, f"sample_{i+1}.{IMAGE_FORMAT.lower()}")
                    if save_image(parsed.get("image", ""), image_path):
                        samples.append([image_path, parsed.get("label", ""), parsed.get("confidence", 0.0)])
                elif task == 'object_detection':
                    image_path = os.path.join(image_dir, f"sample_{i+1}.{IMAGE_FORMAT.lower()}")
                    if save_image(parsed.get("image", ""), image_path):
                        for obj in parsed.get("objects", []):
                            samples.append([image_path, obj.get("label", ""), *obj.get("bbox", [0,0,0,0]), obj.get("confidence", 0.0)])
                # ... [other task handling remains unchanged]
                
                print(f"Generated {task} sample {len(samples)+1}/{n}")
            else:
                print(f"Invalid output: {output}")
        else:
            print(f"Failed to generate {task} sample {len(samples)+1}/{n}")
    return samples

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    for task, topics in VISION_TASKS.items():
        samples = generate_vision_samples(
            task, VISION_TOPICS, args.num_samples_per_task, 
            args.deepseek_model, args.deepseek_api_url, args.max_retries, args.retry_delay,
            use_chroma=args.use_chroma, chroma_model_name=args.chroma_model
        )

        out_path = os.path.join(args.output_dir, f"{task}.csv")
        task_headers = ANNOTATION_FORMATS.get(task, ["id", "content"])  # Default headers if task not found
        
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(task_headers)
            writer.writerows([[str(i+1)] + list(row) for i, row in enumerate(samples)])
        print(f"Saved {len(samples)} {task} samples to {out_path}")

if __name__ == "__main__":
    main()