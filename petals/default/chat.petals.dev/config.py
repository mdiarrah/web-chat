import torch
from petals.constants import PUBLIC_INITIAL_PEERS

from data_structures import ModelBackendConfig, ModelChatConfig, ModelConfig, ModelFrontendConfig

default_chat_config = ModelChatConfig(
    max_session_length=8192,
    sep_token="###",
    stop_token="###",
    extra_stop_sequences=["</s>"],
    generation_params=dict(do_sample=1, temperature=0.6, top_p=0.9),
)

MODEL_FAMILIES = {
    "Llama": [
         ModelConfig(
            ModelBackendConfig(repository="huggyllama/llama-65b"),
            ModelFrontendConfig(
                name="Llama-65B",
                model_card="https://github.com/facebookresearch/llama/blob/llama_v1/MODEL_CARD.md",
                license="https://bit.ly/llama-license",
            ),
            default_chat_config,
        ),
        ModelConfig(
            ModelBackendConfig(repository="huggyllama/llama-65b", adapter="timdettmers/guanaco-65b"),
            ModelFrontendConfig(
                name="Guanaco-65B",
                model_card="https://huggingface.co/timdettmers/guanaco-65b",
                license="https://huggingface.co/timdettmers/guanaco-65b",
            ),
            default_chat_config,
        ),    
    ],
    "Llama 2": [
        ModelConfig(
            ModelBackendConfig(repository="petals-team/StableBeluga2", aliases=["stabilityai/StableBeluga2"]),
            ModelFrontendConfig(
                name="Stable Beluga 2 (70B)",
                model_card="https://huggingface.co/stabilityai/StableBeluga2",
                license="https://huggingface.co/stabilityai/StableBeluga2/blob/main/LICENSE.txt",
            ),
            default_chat_config,
        )
    ],
    


}

INITIAL_PEERS = ['/ip4/51.79.102.103/tcp/31337/p2p/QmT3TtHZyKGHuXzgWaC5AXscQsFRrH9jJGU8PC4YJUwD5g']

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from cpufeature import CPUFeature

    has_avx512 = CPUFeature["AVX512f"] and CPUFeature["OS_AVX512"]
except ImportError:
    has_avx512 = False

if DEVICE == "cuda":
    TORCH_DTYPE = "auto"
elif has_avx512:
    TORCH_DTYPE = torch.bfloat16
else:
    TORCH_DTYPE = torch.float32  # You can use bfloat16 in this case too, but it will be slow

STEP_TIMEOUT = 5 * 60
