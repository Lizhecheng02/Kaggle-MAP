# MAP-Charting-Student-Math-Misunderstandings Competition Ideas

## Task

* A diagnostic multiple choice math question is presented to the student
  * The question contains 4 choices, of which 3 are called "distractors"
* Given a question, and the choice that the student selected, and the explanation for the answer that the student gave choose the top 3 most likely labels.
* The labels are composed of whether the student got the question correct or not ("True" or "False"), how the rater graded the student's explanation ("Explanation is correct", "Explanation is incorrect and displays a misconception", "Explanation is neither correct nor displays a misconcpetion"), and *if* the rater believes a misconception is present then which question-specific misconception that is.
* We are supposed to give the top 3 labels
* The metric used is MAP@3

## Thoughts

* There are 65 labels with no new labels in the test set (found via leaderboard probing)
  * Therefore generalization to out-of-distribution samples is not as important here
  * Synthetic data may be limited because the data we would try to create are dependent on multiple factors that we do not have control over:
    * For training data: the student's understanding and ability to express themselves (despite their understanding)
    * For labels: the rater's interpretation of the student's explanation
  * This is in contrast to other cases where the 
* While we can treat this as a simple classification problem, we may have better results by treating this as an information retrieval problem.
  * The difference would be that classification is a "one pass" approach while information retrieval is a "multi-pass" approach
* Chain of Thought may be crucial to getting better results
  * However this will depend on the models not having "definitive" language that might sway models to be very confident
  * This will also require having clear definitions for the labels and misconceptions and linking them to the label classes

## Preparation

* We need to establish good CV splits that are reusable across experiments
* Need to create folds across the whole data (`StratifiedKFold`)

## Ideas 

#### Reranker

