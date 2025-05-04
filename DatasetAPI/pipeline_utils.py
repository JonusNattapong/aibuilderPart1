"""
Utility functions for pipeline progress tracking and callbacks.
"""
import logging
from typing import Optional, Callable

# Setup logger
logger = logging.getLogger(__name__)

class PipelineProgress:
    """Tracks progress of the pipeline and handles callbacks."""
    
    STAGE_NAMES = {
        'generation': 'Generating Examples',
        'format': 'Validating Format',
        'execution': 'Simulating API Execution',
        'semantic': 'Checking Semantic Consistency'
    }

    def __init__(self, total_samples: int,
                progress_callback: Optional[Callable] = None,
                stage_callback: Optional[Callable] = None):
        self.total_samples = total_samples
        self.current_sample = 0
        self.stage_progress = 0.0
        self.progress_callback = progress_callback
        self.stage_callback = stage_callback
        self.stages = {
            'generation': 0.4,  # 40% of progress bar
            'format': 0.2,     # 20% of progress bar
            'execution': 0.2,   # 20% of progress bar
            'semantic': 0.2     # 20% of progress bar
        }
        self.current_stage = 'generation'

    def update_stage(self, stage: str):
        """Updates the current stage being processed."""
        try:
            if stage in self.stages:
                self.current_stage = stage
                self._update_progress()
                if self.stage_callback:
                    stage_name = self.STAGE_NAMES.get(stage, stage)
                    try:
                        self.stage_callback(stage_name)
                    except Exception as e:
                        logger.error(f"Error in stage callback: {e}")
            else:
                logger.warning(f"Unknown stage: {stage}")
        except Exception as e:
            logger.error(f"Error updating stage: {e}")
            # Continue execution even if progress tracking fails

    def increment_sample(self):
        """Increments the current sample count."""
        self.current_sample += 1
        self._update_progress()
def _update_progress(self):
    """Calculates and reports current progress."""
    try:
        # Calculate base progress from completed samples
        base_progress = (self.current_sample / self.total_samples)
        
        # Add stage-specific progress
        stage_weight = 0.0
        for stage, weight in self.stages.items():
            if stage == self.current_stage:
                stage_weight += weight * self.stage_progress
                break
            stage_weight += weight

        # Combine sample and stage progress
        total_progress = base_progress + (stage_weight / self.total_samples)
        
        # Ensure progress stays within bounds
        total_progress = min(1.0, max(0.0, total_progress))
        
        # Report progress through callback if available
        if self.progress_callback:
            try:
                self.progress_callback(total_progress)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

    except Exception as e:
        logger.error(f"Error calculating progress: {e}")
        # Return early but don't raise to avoid breaking the pipeline
        return

    def update_stage_progress(self, progress: float):
        """Updates progress within the current stage (0.0 to 1.0)."""
        self.stage_progress = min(1.0, max(0.0, progress))
        self._update_progress()

    def reset(self):
        """Resets progress tracking."""
        self.current_sample = 0
        self.stage_progress = 0.0
        self.current_stage = 'generation'
        self._update_progress()