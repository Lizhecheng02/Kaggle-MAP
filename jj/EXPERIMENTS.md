# List of Experiments


## `synthetic_data`

#### Overview

Attempt to create synthetic data by using the persona of a digital twin to mimic the persona of the original student.
Requires that the output is semantically equivalent and also has similar style to the original student.

#### Limitations

It is difficult to measure semantic and style similarity, even when using LLM as a judge


## `think_models`

#### Overview

Train a LLM to reason and produce CoT output in a manner similar to that of a large LLM (like Gemini-2.5-Flash).
Using `unsloth` to distil the information.

#### Limitations

Currently using Gemini-2.5-Flash. However seems to be limited compared to higher class models like Claude Sonnet.


## `think_zeroshot`

#### Overview

Instead of training a thinking model, use zero-shot reasoning from a fairly large model like `Qwen2.5-32B-Instruct-AWQ`

#### Limitations

The reasoning output is not on par with that from a larger model.
Producing reasoning output will take a significant amount of time.


## `reproducible_data`

* Date: 2025-09-01

#### Overview

Create definitive dataset for training that splits the whole dataset into 5 folds using `StratifiedKFold` using `QuestionId` and `label`.


## `fold_train`

* Date: 2025-09-01

#### Overview

Perform train over the folds of the data.