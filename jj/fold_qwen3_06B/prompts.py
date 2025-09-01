import json

questions = {
    "31772": {
        "QuestionText": "What fraction of the shape is not shaded? Give your answer in its simplest form. [Image: A triangle split into 9 equal smaller triangles. 6 of them are shaded.]",
        "Choices": [
            {"Choice": "\\( \\frac{1}{3} \\)", "Correct": True},
            {"Choice": "\\( \\frac{3}{6} \\)", "Correct": False},
            {"Choice": "\\( \\frac{3}{8} \\)", "Correct": False},
            {"Choice": "\\( \\frac{3}{9} \\)", "Correct": False},
        ],
    },
    "31774": {
        "QuestionText": "Calculate \\( \\frac{1}{2} \\div 6 \\)",
        "Choices": [
            {"Choice": "\\( 3 \\)", "Correct": False},
            {"Choice": "\\( \\frac{1}{12} \\)", "Correct": True},
            {"Choice": "\\( \\frac{1}{3} \\)", "Correct": False},
            {"Choice": "\\( \\frac{6}{2} \\)", "Correct": False},
        ],
    },
    "31777": {
        "QuestionText": "A box contains \\( 120 \\) counters. The counters are red or blue. \\( \\frac{3}{5} \\) of the counters are red.\nHow many red counters are there?",
        "Choices": [
            {"Choice": "\\( 24 \\)", "Correct": False},
            {"Choice": "\\( 48 \\)", "Correct": False},
            {"Choice": "\\( 60 \\)", "Correct": False},
            {"Choice": "\\( 72 \\)", "Correct": True},
        ],
    },
    "31778": {
        "QuestionText": "\\( \\frac{A}{10}=\\frac{9}{15} \\) What is the value of \\( A \\) ?",
        "Choices": [
            {"Choice": "\\( 3 \\)", "Correct": False},
            {"Choice": "\\( 4 \\)", "Correct": False},
            {"Choice": "\\( 6 \\)", "Correct": True},
            {"Choice": "\\( 9 \\)", "Correct": False},
        ],
    },
    "32829": {
        "QuestionText": "\\( 2 y=24 \\) What is the value of \\( y \\) ?",
        "Choices": [
            {"Choice": "\\( 12 \\)", "Correct": True},
            {"Choice": "\\( 22 \\)", "Correct": False},
            {"Choice": "\\( 4 \\)", "Correct": False},
            {"Choice": "\\( 48 \\)", "Correct": False},
        ],
    },
    "32833": {
        "QuestionText": "Calculate \\( \\frac{2}{3} \\times 5 \\)",
        "Choices": [
            {"Choice": "\\( 3 \\frac{1}{3} \\)", "Correct": True},
            {"Choice": "\\( 5 \\frac{2}{3} \\)", "Correct": False},
            {"Choice": "\\( \\frac{10}{15} \\)", "Correct": False},
            {"Choice": "\\( \\frac{2}{15} \\)", "Correct": False},
        ],
    },
    "32835": {
        "QuestionText": "Which number is the greatest?",
        "Choices": [
            {"Choice": "\\( 6 \\)", "Correct": False},
            {"Choice": "\\( 6.0001 \\)", "Correct": False},
            {"Choice": "\\( 6.079 \\)", "Correct": False},
            {"Choice": "\\( 6.2 \\)", "Correct": True},
        ],
    },
    "33471": {
        "QuestionText": "A bag contains \\( 24 \\) yellow and green balls. \\( \\frac{3}{8} \\) of the balls are yellow. How many of the balls are green?",
        "Choices": [
            {"Choice": "\\( 15 \\)", "Correct": True},
            {"Choice": "\\( 3 \\)", "Correct": False},
            {"Choice": "\\( 8 \\)", "Correct": False},
            {"Choice": "\\( 9 \\)", "Correct": False},
        ],
    },
    "33472": {
        "QuestionText": "\\( \\frac{1}{3}+\\frac{2}{5}= \\)",
        "Choices": [
            {"Choice": "\\( \\frac{11}{15} \\)", "Correct": True},
            {"Choice": "\\( \\frac{11}{30} \\)", "Correct": False},
            {"Choice": "\\( \\frac{3}{15} \\)", "Correct": False},
            {"Choice": "\\( \\frac{3}{8} \\)", "Correct": False},
        ],
    },
    "33474": {
        "QuestionText": "Sally has \\( \\frac{2}{3} \\) of a whole cake in the fridge. Robert eats \\( \\frac{1}{3} \\) of this piece. What fraction of the whole cake has Robert eaten?\nChoose the number sentence that would solve the word problem.",
        "Choices": [
            {"Choice": "\\( \\frac{1}{3} \\times \\frac{2}{3} \\)", "Correct": True},
            {"Choice": "\\( \\frac{1}{3}+\\frac{2}{3} \\)", "Correct": False},
            {"Choice": "\\( \\frac{2}{3} \\div \\frac{1}{3} \\)", "Correct": False},
            {"Choice": "\\( \\frac{2}{3}-\\frac{1}{3} \\)", "Correct": False},
        ],
    },
    "76870": {
        "QuestionText": "This is part of a regular polygon. How many sides does it have? [Image: A diagram showing an obtuse angle labelled 144 degrees]",
        "Choices": [
            {"Choice": "Not enough information", "Correct": False},
            {"Choice": "\\( 10 \\)", "Correct": True},
            {"Choice": "\\( 5 \\)", "Correct": False},
            {"Choice": "\\( 6 \\)", "Correct": False},
        ],
    },
    "89443": {
        "QuestionText": "Question: What number belongs in the box?\n\\((-8)-(-5)=x\\)",
        "Choices": [
            {"Choice": "\\( -13 \\)", "Correct": False},
            {"Choice": "\\( -3 \\)", "Correct": True},
            {"Choice": "\\( 13 \\)", "Correct": False},
            {"Choice": "\\( 3 \\)", "Correct": False},
        ],
    },
    "91695": {
        "QuestionText": "Dots have been arranged in these patterns: [Image: Pattern 1 consists of 6 dots, Pattern 2 consists of 10 dots, Pattern 3 consists of 14 dots and Pattern 4 consists of 18 dots] How many dots would there be in Pattern \\( 6 \\) ?",
        "Choices": [
            {"Choice": "\\( 20 \\)", "Correct": False},
            {"Choice": "\\( 22 \\)", "Correct": False},
            {"Choice": "\\( 26 \\)", "Correct": True},
            {"Choice": "\\( 36 \\)", "Correct": False},
        ],
    },
    "104665": {
        "QuestionText": "It takes \\( 3 \\) people a total of \\( 192 \\) hours to build a wall.\n\nHow long would it take if \\( 12 \\) people built the same wall?",
        "Choices": [
            {"Choice": "\\( 192 \\) hours", "Correct": False},
            {"Choice": "\\( 48 \\) hours", "Correct": True},
            {"Choice": "\\( 64 \\) hours", "Correct": False},
            {"Choice": "\\( 768 \\) hours", "Correct": False},
        ],
    },
    "109465": {
        "QuestionText": "The probability of an event occurring is \\( 0.9 \\).\n\nWhich of the following most accurately describes the likelihood of the event occurring?",
        "Choices": [
            {"Choice": "Certain", "Correct": False},
            {"Choice": "Impossible", "Correct": False},
            {"Choice": "Likely", "Correct": True},
            {"Choice": "Unlikely", "Correct": False},
        ],
    },
}

teacher_explanations = {
    "31772": "The shape has 9 equal parts in total, and 3 of them are not shaded. So the unshaded fraction is $\\frac{3}{9}$, which simplifies to $\\frac{1}{3}$.",
    "31774": "To divide a fraction by a whole number, you multiply the fraction by the reciprocal of the whole number. So, $\\frac{1}{2} \\div 6 = \\frac{1}{2} \\times \\frac{1}{6} = \\frac{1}{12}$.",
    "31777": "To find $\\frac{3}{5}$ of 120, you divide 120 by 5 (which is 24) and then multiply that result by 3 (which is 72).",
    "31778": "We can simplify the fraction $\\frac{9}{15}$ to $\\frac{3}{5}$ by dividing both the numerator and the denominator by 3. Then, to make the denominator 10, we multiply both the numerator and denominator of $\\frac{3}{5}$ by 2, which gives us $\\frac{6}{10}$. So, A is 6.",
    "32829": "To solve for y, you need to get y by itself. Since y is being multiplied by 2, you divide both sides of the equation by 2, so $y = 24 \\div 2 = 12$.",
    "32833": "To multiply a fraction by a whole number, you multiply the numerator by the whole number: $2 \\times 5 = 10$. This gives $\\frac{10}{3}$. As a mixed number, $\\frac{10}{3}$ is $3 \\frac{1}{3}$.",
    "32835": "To compare decimals, look at the digits from left to right. All choices start with 6. Comparing the tenths place, 6.2 has a 2, which is larger than the 0s in the other numbers, making 6.2 the greatest.",
    "33471": "If $\\frac{3}{8}$ of the balls are yellow, then $1 - \\frac{3}{8} = \\frac{5}{8}$ of the balls are green. To find how many green balls there are, calculate $\\frac{5}{8}$ of 24, which is $(24 \\div 8) \\times 5 = 3 \\times 5 = 15$.",
    "33472": "To add fractions, you need a common denominator. The least common multiple of 3 and 5 is 15. So, $\\frac{1}{3} = \\frac{5}{15}$ and $\\frac{2}{5} = \\frac{6}{15}$. Adding them gives $\\frac{5}{15} + \\frac{6}{15} = \\frac{11}{15}$.",
    "33474": "When you want to find a fraction 'of' another fraction, it means you need to multiply them. In this case, Robert eats $\\frac{1}{3}$ of Sally's $\\frac{2}{3}$ piece, so you multiply $\\frac{1}{3} \\times \\frac{2}{3}$.",
    "76870": "The exterior angle of a regular polygon is $180 - 144 = 36$ degrees. The number of sides is found by dividing 360 degrees by the exterior angle: $360 \\div 36 = 10$ sides.",
    "89443": "Subtracting a negative number is the same as adding the positive version of that number. So, $(-8) - (-5)$ becomes $(-8) + 5$, which equals $-3$.",
    "91695": "The number of dots increases by 4 each time (6, 10, 14, 18). So, for Pattern 5, there would be $18 + 4 = 22$ dots, and for Pattern 6, there would be $22 + 4 = 26$ dots.",
    "104665": "This is an inverse proportion problem: more people mean less time. Calculate the total work in hours ($3 \\text{ people} \\times 192 \\text{ hours} = 576 \\text{ hours}$). Then divide this by the new number of people ($576 \\text{ hours} \\div 12 \\text{ people} = 48 \\text{ hours}$).",
    "109465": "Probabilities range from 0 (impossible) to 1 (certain). A probability of 0.9 is very close to 1, meaning the event is highly likely to occur.",
}

misconception_explanations = {
    "31772": [
        {
            "Label": "Incomplete",
            "LabelLong": "Incomplete",
            "Explanation": "the student correctly identifies that the unshaded fraction is 3 out of 9, but they do not simplify the fraction to its simplest form as the question asks.",
        },
        {
            "Label": "WNB",
            "LabelLong": "Wrong Numerator and/or Base",
            "Explanation": "the student may have counted the shaded parts (6) instead of the total parts (9) for the denominator, leading to an incorrect fraction like 3/6.",
        },
    ],
    "31774": [
        {
            "Label": "SwapDividend",
            "LabelLong": "Swap Dividend and Divisor",
            "Explanation": "the student may have swapped the numbers in the division, calculating 6 divided by 2, which incorrectly gives 3.",
        },
        {
            "Label": "Mult",
            "LabelLong": "Multiplication",
            "Explanation": "the student, instead of dividing, might have multiplied the fraction's numerator by the whole number, calculating 1/2 * 6 = 3.",
        },
        {
            "Label": "FlipChange",
            "LabelLong": "Incorrect 'Flip and Change'",
            "Explanation": "the student remembers to change the operation to multiplication but forgets to 'flip' the whole number 6 into its reciprocal, 1/6, leading to an incorrect calculation.",
        },
    ],
    "31777": [
        {
            "Label": "Incomplete",
            "LabelLong": "Incomplete",
            "Explanation": "the student correctly performs the first step of dividing 120 by 5 to get 24, but then forgets to complete the calculation by multiplying by 3.",
        },
        {
            "Label": "Irrelevant",
            "LabelLong": "Irrelevant Information/Calculation",
            "Explanation": "the student might get confused and use the numbers in the wrong way, for example, by dividing 120 by the difference of the denominator and numerator (5-3=2) to get 60.",
        },
        {
            "Label": "Wrong_Fraction",
            "LabelLong": "Calculated Wrong Fraction",
            "Explanation": "the student correctly calculates a fraction of the total but finds the number of blue counters (2/5 of 120 = 48) instead of the number of red counters.",
        },
    ],
    "31778": [
        {
            "Label": "Additive",
            "LabelLong": "Assumes Additive Relationship",
            "Explanation": "the student sees that 15 - 5 = 10 and incorrectly applies the same subtraction to the top number, calculating 9 - 5 = 4.",
        },
        {
            "Label": "Irrelevant",
            "LabelLong": "Irrelevant Information/Calculation",
            "Explanation": "the student does not know how to find an equivalent fraction, and does something like guess that A is 9 because the other numerator is 9.",
        },
        {
            "Label": "WNB",
            "LabelLong": "Wrong Numerator and/or Base",
            "Explanation": "the student may have made a mistake when simplifying the fraction 9/15 or when trying to find the factor needed to change the denominator to 10.",
        },
        {
            "Label": "Incomplete",
            "LabelLong": "Incomplete",
            "Explanation": "the student correctly simplifies 9/15 to 3/5 but does not know how to continue and incorrectly chooses 3 as the answer.",
        },
    ],
    "32829": [
        {
            "Label": "Not_variable",
            "LabelLong": "Does Not Understand Variable Notation",
            "Explanation": "the student may not understand that 2y means 2 multiplied by y, and instead do something like 2 plus y.",
        },
        {
            "Label": "Adding_terms",
            "LabelLong": "Adding Terms",
            "Explanation": "the student incorrectly reads 2y as '2 + y' and solves the equation by doing something like subtracting 2 from 24 to get 22.",
        },
        {
            "Label": "Inverse_operation",
            "LabelLong": "Incorrect Inverse Operation",
            "Explanation": "the student knows a calculation with 2 is needed but uses the wrong operation, such as multiplying 24 by 2 to get 48.",
        },
    ],
    "32833": [
        {
            "Label": "Inversion",
            "LabelLong": "Inversion",
            "Explanation": "the student confuses multiplication with division and does something like calculates 2/3 divided by 5, which equals 2/15.",
        },
        {
            "Label": "Duplication",
            "LabelLong": "Duplication",
            "Explanation": "the student incorrectly multiplies both the numerator and the denominator by 5, resulting in the fraction 10/15.",
        },
        {
            "Label": "Wrong_Operation",
            "LabelLong": "Wrong Operation",
            "Explanation": "the student mistakes the multiplication sign for an addition sign and does something like incorrectly adds 5 and 2/3 together.",
        },
    ],
    "32835": [
        {
            "Label": "Whole_numbers_larger",
            "LabelLong": "Whole Numbers are Larger",
            "Explanation": "the student might ignore the decimal part of the numbers, thinking that the plain whole number 6 is somehow larger or more significant than the others.",
        },
        {
            "Label": "Longer_is_bigger",
            "LabelLong": "Longer Decimal is Bigger",
            "Explanation": "the student incorrectly believes that the number with the most digits after the decimal point is the largest, causing them to choose 6.0001.",
        },
        {
            "Label": "Ignores_zeroes",
            "LabelLong": "Ignores Zeroes/Place Value",
            "Explanation": "the student might compare 2 with 79 by ignoring the zeroes, thinking that 6.079 is bigger than 6.2 because 79 is bigger than 2.",
        },
        {
            "Label": "Shorter_is_bigger",
            "LabelLong": "Shorter Decimal is Bigger",
            "Explanation": "the student confuses decimals with fractions, and thinks that the number with the fewest decimal places is the largest, leading them to choose 6.",
        },
    ],
    "33471": [
        {
            "Label": "Wrong_fraction",
            "LabelLong": "Calculated Wrong Fraction",
            "Explanation": "the student correctly calculates a fractional part of the total but finds the number of yellow balls (9) instead of the number of green balls.",
        },
        {
            "Label": "Incomplete",
            "LabelLong": "Incomplete",
            "Explanation": "the student only completes the first step of the problem, finding what one-eighth of the total is (24 divided by 8 equals 3), but doesn't multiply to find the final answer.",
        },
    ],
    "33472": [
        {
            "Label": "Adding_across",
            "LabelLong": "Adding Across",
            "Explanation": "the student incorrectly adds the numerators (1+2=3) and the denominators (3+5=8) separately to get 3/8.",
        },
        {
            "Label": "Denominator-only_change",
            "LabelLong": "Denominator-Only Change",
            "Explanation": "the student finds a common denominator (15) but forgets to change the numerators, simply adding the original numerators (1+2=3) to get 3/15.",
        },
        {
            "Label": "Incorrect_equivalent_fraction_addition",
            "LabelLong": "Incorrect Equivalent Fraction Addition",
            "Explanation": "the student correctly calculates the numerator (11) but then makes a mistake with the denominator, possibly by adding the two common denominators (15+15) to get 11/30.",
        },
    ],
    "33474": [
        {
            "Label": "Division",
            "LabelLong": "Division",
            "Explanation": "the student might misunderstand the phrase 'fraction of this piece' and think it means to divide the amounts, leading them to choose the division option.",
        },
        {
            "Label": "Subtraction",
            "LabelLong": "Subtraction",
            "Explanation": "the student focuses on the word 'eats' and incorrectly assumes this means something is taken away, leading them to choose the subtraction option.",
        },
    ],
    "76870": [
        {
            "Label": "Unknowable",
            "LabelLong": "Unknowable/Not Enough Information",
            "Explanation": "the student may not know the formula connecting a regular polygon's angles to its sides and therefore incorrectly concludes that the problem cannot be solved.",
        },
        {
            "Label": "Definition",
            "LabelLong": "Definitional Error",
            "Explanation": "the student may not understand what a regular polygon is or may not know the relationship between its interior and exterior angles.",
        },
        {
            "Label": "Interior",
            "LabelLong": "Using Interior Angle Incorrectly",
            "Explanation": "the student incorrectly uses the interior angle (144°) in the formula to find the number of sides, instead of first calculating the exterior angle.",
        },
    ],
    "89443": [
        {
            "Label": "Positive",
            "LabelLong": "Incorrect Sign/Positive",
            "Explanation": "the student correctly finds that the difference between 8 and 5 is 3 but gets the sign wrong, possibly by ignoring the negative signs and just calculating 8 - 5 = 3.",
        },
        {
            "Label": "Tacking",
            "LabelLong": "Tacking/Incorrectly Combining",
            "Explanation": "the student ignores that subtracting a negative is the same as adding, and instead combines the numbers as if it were -8 - 5 to get -13.",
        },
    ],
    "91695": [
        {
            "Label": "Wrong_term",
            "LabelLong": "Finds Wrong Term",
            "Explanation": "the student correctly finds the rule for the pattern (add 4) but only calculates the next term (Pattern 5) instead of Pattern 6.",
        },
        {
            "Label": "Firstterm",
            "LabelLong": "Misusing the First Term",
            "Explanation": "the student might try to create a rule by incorrectly using the first term of the pattern, for example by multiplying the pattern number (6) by the first term (6) to get 36.",
        },
    ],
    "104665": [
        {
            "Label": "Base_rate",
            "LabelLong": "Misinterpreting the Base Rate",
            "Explanation": "the student might mistakenly believe the time is constant regardless of how many people work, or they might divide 192 by 3 to get 64 and think that is the answer.",
        },
        {
            "Label": "Multiplying_by_4",
            "LabelLong": "Multiplying by 4/Assuming Direct Proportion",
            "Explanation": "the student sees that the number of people quadrupled and incorrectly assumes the time will also quadruple, multiplying 192 by 4 instead of dividing.",
        },
    ],
    "109465": [
        {
            "Label": "Certainty",
            "LabelLong": "Confuses Likely with Certain",
            "Explanation": "the student understands that 0.9 is a high probability, but they incorrectly think this means the event is guaranteed to happen, choosing 'Certain' instead of 'Likely'.",
        },
        {
            "Label": "Scale",
            "LabelLong": "Misunderstands Probability Scale",
            "Explanation": "the student does not have a clear understanding of the probability scale from 0 (impossible) to 1 (certain) and cannot correctly place where 0.9 falls.",
        },
    ],
}


def create_messages_v1(row, think: str = "\n\n"):
    if row["is_correct"]:
        status = "Yes"
    else:
        status = "No"

    # Create messages in the standard format
    messages = [
        {
            "role": "system",
            "content": (
                "You are a math teacher grading students that took a diagnostic multiple choice math question. "
                "You must classify the explanation given by the student as to why they chose their answer."
            )
        },
        {
            "role": "user", 
            "content": (
                f"Question: {row['QuestionText']}\n"
                f"Answer: {row['MC_Answer']}\n"
                f"Correct?: {status}\n"
                f"Explanation: {row['StudentExplanation']}"
            )
        },
        {
            "role": "assistant",
            "content": f"<think>Let me analyze this mathematical misconception.\n{think}</think>\n\n"
        }
    ]

    return messages
