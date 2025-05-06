"""
Utility functions for reasoning dataset generation.
"""
import os
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from config import (
    TASK_CONFIG,
    MODEL_CONFIG,
    MAX_PROMPT_LENGTH,
    MAX_NEW_TOKENS
)

class ReasoningTaskManager:
    """Manages reasoning tasks and their processing."""
    
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.tokenizer = None

    def process_batch(self, model: torch.nn.Module, prompts: List[str], config: Dict[str, Any]) -> List[Dict]:
        """Process a batch of prompts for the specified task."""
        
        if self.task_name == "cot":
            return self._process_chain_of_thought(model, prompts, config)
        elif self.task_name == "react":
            return self._process_react(model, prompts, config)
        elif self.task_name == "tot":
            return self._process_tree_of_thought(model, prompts, config)
        elif self.task_name == "meta":
            return self._process_meta_reasoning(model, prompts, config)
        else:
            raise ValueError(f"Unsupported task: {self.task_name}")

    def _process_chain_of_thought(self, model, prompts, config):
        """Process Chain-of-Thought reasoning."""
        results = []
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer or model.config.tokenizer_class,
            device=MODEL_CONFIG["device"]
        )

        for prompt in prompts:
            cot_prompt = f"Let's solve this step by step:\nQuestion: {prompt}\nLet's approach this step by step:\n1."
            
            output = generator(
                cot_prompt,
                max_new_tokens=config.get("max_tokens", MAX_NEW_TOKENS),
                temperature=config.get("temperature", 0.7),
                num_return_sequences=1
            )[0]["generated_text"]

            # Extract reasoning steps and final answer
            steps = []
            answer = ""
            
            # Parse numbered steps
            lines = output.split("\n")
            current_step = []
            for line in lines:
                if line.strip().startswith(str(len(steps) + 1) + "."):
                    if current_step:
                        steps.append(" ".join(current_step))
                        current_step = []
                    current_step.append(line.strip()[2:].strip())
                elif line.strip().lower().startswith("therefore") or \
                     line.strip().lower().startswith("answer"):
                    if current_step:
                        steps.append(" ".join(current_step))
                    answer = line.strip()
                    break
                elif current_step:
                    current_step.append(line.strip())

            results.append({
                "reasoning_steps": steps,
                "answer": answer
            })

        return results

    def _process_react(self, model, prompts, config):
        """Process ReAct (Reasoning + Acting) approach."""
        results = []
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer or model.config.tokenizer_class,
            device=MODEL_CONFIG["device"]
        )

        for prompt in prompts:
            react_prompt = (
                f"Let's approach this by Reasoning and Acting:\n"
                f"Question: {prompt}\n"
                f"Thought 1: Let me break this down."
            )

            output = generator(
                react_prompt,
                max_new_tokens=config.get("max_tokens", MAX_NEW_TOKENS),
                temperature=config.get("temperature", 0.7),
                num_return_sequences=1
            )[0]["generated_text"]

            # Extract thoughts, actions, and observations
            steps = []
            final_answer = ""
            
            lines = output.split("\n")
            current_sequence = []
            
            for line in lines:
                if line.strip().startswith("Thought"):
                    if current_sequence:
                        steps.append({
                            "type": "sequence",
                            "content": current_sequence
                        })
                        current_sequence = []
                    current_sequence.append({"type": "thought", "content": line})
                elif line.strip().startswith("Action"):
                    current_sequence.append({"type": "action", "content": line})
                elif line.strip().startswith("Observation"):
                    current_sequence.append({"type": "observation", "content": line})
                elif line.strip().lower().startswith("therefore") or \
                     line.strip().lower().startswith("final answer"):
                    if current_sequence:
                        steps.append({
                            "type": "sequence",
                            "content": current_sequence
                        })
                    final_answer = line.strip()
                    break

            results.append({
                "reasoning_steps": steps,
                "answer": final_answer
            })

        return results

    def _process_tree_of_thought(self, model, prompts, config):
        """Process Tree-of-Thought reasoning."""
        results = []
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer or model.config.tokenizer_class,
            device=MODEL_CONFIG["device"]
        )

        for prompt in prompts:
            tot_prompt = (
                f"Let's solve this using Tree of Thought reasoning:\n"
                f"Problem: {prompt}\n"
                f"Let's explore different approaches:\n"
                f"Branch 1:"
            )

            output = generator(
                tot_prompt,
                max_new_tokens=config.get("max_tokens", MAX_NEW_TOKENS),
                temperature=config.get("temperature", 0.8),
                num_return_sequences=config.get("num_branches", 3)
            )

            # Process each branch
            branches = []
            final_answer = ""
            
            for branch_output in output:
                branch_text = branch_output["generated_text"]
                branch_lines = branch_text.split("\n")
                
                current_branch = {
                    "steps": [],
                    "evaluation": "",
                    "confidence": 0.0
                }
                
                for line in branch_lines:
                    if line.strip().startswith("Step"):
                        current_branch["steps"].append(line.strip())
                    elif line.strip().startswith("Evaluation"):
                        current_branch["evaluation"] = line.strip()
                    elif line.strip().startswith("Confidence"):
                        try:
                            current_branch["confidence"] = float(
                                line.strip().split(":")[-1].strip().rstrip("%")
                            ) / 100
                        except ValueError:
                            current_branch["confidence"] = 0.0
                
                branches.append(current_branch)
            
            # Select best branch based on confidence
            best_branch = max(branches, key=lambda x: x["confidence"])
            final_answer = best_branch["evaluation"]

            results.append({
                "branches": branches,
                "reasoning_steps": best_branch["steps"],
                "answer": final_answer,
                "confidence": best_branch["confidence"]
            })

        return results

    def _process_meta_reasoning(self, model, prompts, config):
        """Process meta-reasoning approach."""
        results = []
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer or model.config.tokenizer_class,
            device=MODEL_CONFIG["device"]
        )

        for prompt in prompts:
            meta_prompt = (
                f"Let's solve this using meta-reasoning:\n"
                f"Problem: {prompt}\n\n"
                f"1. Strategy Selection:"
            )

            output = generator(
                meta_prompt,
                max_new_tokens=config.get("max_tokens", MAX_NEW_TOKENS),
                temperature=config.get("temperature", 0.7),
                num_return_sequences=1
            )[0]["generated_text"]

            # Extract meta-reasoning components
            components = {
                "strategy": "",
                "reasoning_steps": [],
                "monitoring": [],
                "evaluation": "",
                "answer": ""
            }
            
            lines = output.split("\n")
            current_section = ""
            current_content = []
            
            for line in lines:
                if line.strip().startswith("1. Strategy"):
                    current_section = "strategy"
                elif line.strip().startswith("2. Reasoning"):
                    if current_content:
                        components[current_section] = " ".join(current_content)
                        current_content = []
                    current_section = "reasoning_steps"
                elif line.strip().startswith("3. Monitoring"):
                    if current_content:
                        components[current_section] = current_content
                        current_content = []
                    current_section = "monitoring"
                elif line.strip().startswith("4. Evaluation"):
                    if current_content:
                        components[current_section] = current_content
                        current_content = []
                    current_section = "evaluation"
                elif line.strip().startswith("Final Answer"):
                    if current_content:
                        components[current_section] = " ".join(current_content)
                        current_content = []
                    components["answer"] = line.strip().split(":", 1)[1].strip()
                elif current_section:
                    current_content.append(line.strip())

            results.append(components)

        return results

def get_supported_tasks() -> Dict[str, Dict]:
    """Get dictionary of supported reasoning tasks."""
    return TASK_CONFIG

def get_supported_models(task: str) -> List[str]:
    """Get list of supported models for a task."""
    return TASK_CONFIG[task].get("models", ["huggingface/default-model"])

def load_model(task: str, model_name: str) -> torch.nn.Module:
    """Load the specified model for a task."""
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(MODEL_CONFIG["device"])
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_name} for task {task}: {str(e)}")

def save_dataset(data: List[Dict],
                filename: str,
                format_type: str,
                output_dir: str) -> str:
    """Save dataset in specified format."""
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{base_filename}.{format_type.lower()}")
    
    try:
        if format_type.upper() == "CSV":
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
        elif format_type.upper() == "JSON":
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format_type.upper() == "JSONL":
            with open(output_path, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return output_path
    except Exception as e:
        raise RuntimeError(f"Error saving dataset: {str(e)}")

def setup_device() -> torch.device:
    """Set up compute device (CPU/GPU)."""
    if torch.cuda.is_available() and MODEL_CONFIG["device"] == "cuda":
        return torch.device('cuda')
    else:
        return torch.device('cpu')