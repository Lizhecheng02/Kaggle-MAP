from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # Model configuration
    VER: int = 1
    MODEL_NAME: str = "microsoft/phi-4"

    DEBUG: bool = False
    RANDOM_SEED: int = 42
    
    # GPU settings
    CUDA_VISIBLE_DEVICES: Optional[str] = None
    
    # Data paths
    TRAIN_DATA_PATH: str = "../outputs/train_fold.parquet"
    TEST_DATA_PATH: str = "../../input/map-charting-student-math-misunderstandings/test.csv"
    INFERENCE_DATA_PATH: str = "../outputs/train_fold.parquet"
    FOLDS: int = 5    
    MAX_LEN: int = 512

    # Prompt
    PROMPT_VERSION: str = "create_prompt_v1"
    
    # LoRA configurations
    LORA_RANK: int = 64
    LORA_ALPHA: int = 128
    LORA_TARGET_MODULES: List[str] = field(default_factory=lambda: [
        "q_proj",
        "v_proj",
        "k_proj", 
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ])
    LORA_DROPOUT: float = 0.1
    LORA_BIAS: str = "none"  # "none", "all", "lora_only"
    USE_DORA: bool = False

    # Training parameters
    EPOCHS: int = 4
    TRAIN_BATCH_SIZE: int = 32
    EVAL_BATCH_SIZE: int = 32
    GRADIENT_ACCUMULATION_STEPS: int = 2
    LEARNING_RATE: float = 1e-4
    LOGGING_STEPS: int = 10
    SAVE_STEPS: int = 100
    EVAL_STEPS: int = 100
    LABEL_SMOOTHING_FACTOR: float = 0.05
    TRAIN_FULL_DATA: bool = False
    WARM_UP: float = 0.0
    
    # Early stopping settings
    USE_EARLY_STOPPING: bool = True
    EARLY_STOPPING_PATIENCE: int = 10
    EARLY_STOPPING_THRESHOLD: float = 0.001

    # Memory optimization settings
    USE_GRADIENT_CHECKPOINTING: bool = True
    USE_8BIT_ADAM: bool = False
    MAX_GRAD_NORM: float = 1.0
    
    # Submission settings
    SUBMISSION_OUTPUT_PATH: str = "submission.csv"
    
    # WandB settings
    USE_WANDB: bool = True
    WANDB_PROJECT: str = "map"
    WANDB_ENTITY: Optional[str] = None
    
    @property
    def OUTPUT_DIR(self) -> str:
        return f"map_{self.MODEL_NAME.replace('/', '_')}_ver_{self.VER}_seed_{self.RANDOM_SEED}"
    
    @property
    def BEST_MODEL_PATH(self) -> str:
        return f"{self.OUTPUT_DIR}/best"
    
    @property
    def LABEL_ENCODER_PATH(self) -> str:
        return f"{self.OUTPUT_DIR}/label_encoder.joblib"
    
    @property
    def WANDB_RUN_NAME(self) -> str:
        return self.OUTPUT_DIR