# Map Charting Student Math Misunderstandings

## Think Models

#### Overview

In the Eedi - Mining Misconceptions in Mathematics competition, Raja Biswas [demonstrated](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution) a novel technique to create model to act as a "reasoner" that could generate Chain of Thought (CoT) output for other models. 

We hypothesize that including CoT (what we would find between `<think>` tags) will help LLMs to better understand the student evaluations and improve the classifier scores.

Also the CoT may be helpful for reranker models, as Raja demonstrated in his solution.

#### Dataset

To create the dataset we have already created three additional adjunct datasets:

1. `questions.py` contains the question ids, question text, answer choices, and which choice is the correct answer
2. `misconceptions.py` contains the misconceptions associated with each question (as taken from `train.csv`) and using OpenAI O3, created a long-format name and explanation for what these misconceptions meant
3. `teacher_explanations.py` contains brief explanations of how to solve the problems (generated using OpenAI O3).

We then create the CoT output in `create_think_content.py`. We use Gemini-2.5-flash since it is cheap, but also substantially better at reasoning (via GPQA benchmark) compared to LLMs such as Claude Sonnet 4.

Note that at inference time, we need to use the `create_question_think_user_prompt` function from `prompts.py` with data from `test.csv` to create the user prompts in order to generate the CoT output for downstream tasks.

The final output is placed in the `data` and will be used for training the think models.

#### Training

The first attempt used Phi4-reasoning-plus. There are several issues with this model:

1. The original version had the model generate `<problem_solution_evaluation`> and `<evaluation>` tags and not `<think>`
2. Inference using 8bit or 4bit has substantially worse output than 16bit output
3. The worse performance of 8bit and 4bit seems to be related to `<think>` content not being generated
4. If 16bit is used, Phi4-reasoning-plus was extremely verbose in its `<think>` output, often using all of the token budget (consistent with experiences by others such as [Simon Willison](https://simonwillison.net/2025/May/6/phi-4-reasoning/))

The second attempt is to use Phi4 instead.

#### Tips For Using Unsloth

Weights from a single model can be distributed over multiple GPUs.

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = load_in_4bit,
    load_in_8bit = load_in_8bit,
    full_finetuning = full_finetuning,
    device_map = "balanced",    # Use this flag
)
```

Unsloth recommends using their version of tokenizers to patch any issues.

```python
# Unsloth recommends using their template:
# https://docs.unsloth.ai/basics/datasets-guide
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "phi-4", # change this to the right chat_template name
)
```

You can have the model calculate the loss based only on the assistant portion of the response by patching the `trainer` object.

```python
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user<|im_sep|>",      # Use the appropriate format for your model
    response_part="<|im_start|>assistant<|im_sep|>",    # Use the appropriate format for your model
)
```

To avoid issues with gradient accumulation use the `unsloth_train` function to start training.

```python
# https://unsloth.ai/blog/gradient
trainer_stats = unsloth_train(trainer)
```