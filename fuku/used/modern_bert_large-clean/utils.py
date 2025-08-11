"""
共通ユーティリティ関数
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset
import torch


def prepare_correct_answers(train_data):
    """正解答案データを準備"""
    idx = train_data.apply(lambda row: row.Category.split('_')[0] == 'True', axis=1)
    correct = train_data.loc[idx].copy()
    correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
    correct = correct.sort_values('c', ascending=False)
    correct = correct.drop_duplicates(['QuestionId'])[['QuestionId','MC_Answer']]
    correct['is_correct'] = 1
    return correct


def format_input(row):
    """入力データをモデル用プロンプトにフォーマット"""
    x = "Yes"
    if not row['is_correct']:
        x = "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct? {x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )


def tokenize_dataset(dataset, tokenizer, max_len):
    """データセットをトークナイズ"""
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=max_len)

    dataset = dataset.map(tokenize, batched=True)
    columns = ['input_ids', 'attention_mask', 'label'] if 'label' in dataset.column_names else ['input_ids', 'attention_mask']
    dataset.set_format(type='torch', columns=columns)
    return dataset


def compute_map3(eval_pred):
    """Top-3 予測に基づくMAP@3を計算"""
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    top3 = np.argsort(-probs, axis=1)[:, :3]
    score = 0.0
    for i, label in enumerate(labels):
        ranks = top3[i]
        if ranks[0] == label:
            score += 1.0
        elif ranks[1] == label:
            score += 1.0 / 2
        elif ranks[2] == label:
            score += 1.0 / 3
    return {"map@3": score / len(labels)}


def create_submission(predictions, test_data, label_encoder):
    """予測結果から提出用ファイルを作成"""
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
    top3 = np.argsort(-probs, axis=1)[:, :3]
    flat = top3.flatten()
    decoded = label_encoder.inverse_transform(flat)
    top3_labels = decoded.reshape(top3.shape)
    pred_strings = [" ".join(r) for r in top3_labels]

    submission = pd.DataFrame({
        'row_id': test_data.row_id.values,
        'Category:Misconception': pred_strings
    })
    return submission

import re
import pandas as pd

# Misspelled words
misspelled = {
    "0ne": "one",
    "tree": "three",
    "nineth": "ninth",
    "simplee": "simple",
    "coloured": "colored",
    "threee": "three",
    "equivilent": "equivalent",
    "simplafide": "simplified",
    "shared": "shaded",
    "simplafied": "simplified",
    "shsded": "shaded",
    "fractionn": "fraction",
    "simplifued": "simplified",
    "thee": "the",
    "thr": "the",
    "amd": "and",
    "nott": "not",
    "simpflyed": "simplified",
    "answerr": "answer",
    "duvided": "divided",
    "simplist": "simplest",
    "simplified": "simplified",
    "youu": "you",
    "dimplify": "simplify",
    "isnt": "isn't",
    "simplestt": "simplest",
    "simplistt": "simplest",
    "triangles": "triangles",
    "togeather": "together",
    "denominater": "denominator",
    "numerater": "numerator",
    "havent": "haven't",
    "traiagle": "triangle",
    "coulered": "colored",
    "pirimed": "pyramid",
    "thatt": "that",
    "erhhhhhh": "",
    "theres": "there's",
    "arent": "aren't",
    "dide": "did",
    "numorator": "numerator",
    "simlifyed": "simplified",
    "srent": "aren't",
    "trhird": "third",
    "3rds": "thirds",
    "themm": "them",
    "rightt": "right",
    "becuse": "because",
    "dived": "divide",
    "thimk": "think",
    "shadedd": "shaded",
    "aut": "out",
    "Equivilent": "Equivalent",
    "versionn": "version",
    "shadead": "shaded",
    "inportant": "important",
    "gett": "get",
    "thirdd": "third",
    "simplefy": "simplify",
    "cancell": "cancel",
    "simplifird": "simplified",
    "theire": "there",
    "mot": "not",
    "simpilfy": "simplify",
    "ths": "the",
    "equall": "equal",
    "equaled": "equaled",
    "devided": "divided",
    "allthough": "although",
    "mothod": "method",
    "eitherr": "either",
    "quivalent": "equivalent",
    "dimplest": "simplest",
    "triangless": "triangles",
    "thyy": "they",
    "fractions": "fraction",
    "hafd": "had",
    "simpky": "simplify",
    "eucall": "equal",
    "arr": "are",
    "yiu": "you",
    "niths": "ninths",
    "thenn": "then",
    "becuase": "because",
    "becaus": "because",
    "theree": "there",
    "thaht": "that",
    "iud": "I'd",
    "whivh": "which",
    "fromm": "from",
    "denomiator": "denominator",
    "easil": "easily",
    "coluered": "colored",
    "simpler": "simpler",
    "alway": "always",
    "trinagle": "triangle",
    "triangales": "triangles",
    "simflied": "simplified",
    "arrrr": "are",
    "anwser": "answer",
    "thst": "that",
    "wholes": "whole",
    "colouered": "colored",
    "whiles": "while",
    "simpilfyed": "simplified",
    "coulers": "colors",
    "nime": "nine",
    "dovided": "divided",
    "tha": "the",
    "shadded": "shaded",
    "tringles": "triangles",
    "similer": "similar",
    "simplay": "simply",
    "becausr": "because",
    "coulerd": "colored",
    "simpulfy": "simplify",
    "beacuase": "because",
    "equalls": "equals",
    "andd": "and",
    "alot": "a lot",
    "thsy": "they",
    "equiled": "equaled",
    "colured": "colored",
    "thatts": "that's",
    "becuas": "because",
    "simlply": "simply",
    "whitch": "which",
    "coulred": "colored",
    "simlified": "simplified",
    "triange": "triangle",
    "triangels": "triangles",
    "alrady": "already",
    "colerd": "colored",
    "dificult": "difficult",
    "equivelant": "equivalent",
    "easiy": "easy",
    "simperfly": "simplify",
    "rite": "write",
    "alwasy": "always",
    "becuae": "because",
    "numeriiator": "numerator",
    "id": "I'd",
    "theyy": "they",
    "thiis": "this",
    "Becuse": "because",
    "9ths": "ninths",
    "3rd": "third",
    "whivh": "which",
    "colered": "colored",
    "simlify": "simplify",
    "colered": "colored",
    "alread": "already",
    "ths": "the",
    "colouered": "colored",
    "eqavilent": "equivalent",
    "equevilent": "equivalent",
    "eqeual": "equal",
    "thiss": "this",
    "thta": "that",
    "simmplifying": "simplifying",
    "aunt": "aren't",
    "wuld": "would",
    "itwas": "it",
    "invers": "inverse",
    "thatat": "that",
    "siplyfied": "simplified",
    "simpifying": "simplifying",
    "coluring": "coloring",
    "trangles": "triangles",
    "thaht": "that",
    "didviding": "dividing",
    "thaat": "that",
    "areen't": "aren't",
    "colure": "color",
    "aere": "are",
    "triangles.1": "triangles",
    "inver": "in",
    "themselfs": "themselves",
    "thats": "that's",
    "becaue": "because",
    "alltogeather": "altogether",
    "equilvent": "equivalent",
    "inverce": "inverse",
    "equil": "equal",
    "thattt": "that",
    "deomoninator": "denominator",
    "1twelth": "one",
    "thi": "the",
    "equell": "equal",
    "waws": "was",
    "alothger": "altogether",
    "thier": "there",
    "equvalint": "equivalent",
    "triangells": "triangles",
    "timsed": "times",
    "simlifyed": "simplified",
    "mkae": "make",
    "thaa": "tha",
    "themselve": "themselves",
    "triagls": "triangles",
    "thate": "that",
    "didived": "divided",
    "coulered": "colored",
    "thtt": "that",
    "didveded": "divided",
    "thrrd": "third",
    "thrd": "third",
    "simpilied": "simplified",
    "alother": "another",
    "alltogether": "altogether",
    "simpilfed": "simplified",
    "thaht": "that",
    "becuaae": "because",
    "triangale": "triangle",
    "triangals": "triangles",
    "deed": "did",
    "simpiified": "simplified",
    "tru": "true",
    "numirater": "numerator",
    "alother": "another",
    "3ds": "thirds",
    "didived": "divided",
    "numotater": "numerator",
    "anothet": "another",
    "eqeual": "equal",
    "simpilest": "simplest",
    "thare": "there",
    "dividded": "divided",
    "simpel": "simple",
    "denomanater": "denominator",
    "nitt": "not",
    "thart": "that",
    "alaways": "always",
    "duvide": "divide",
    "thea": "the",
    "thayt": "that",
    "divded": "divided",
    "triagnles": "triangles",
    "equevalent": "equivalent",
    "thena": "than",
    "simiplified": "simplified",
    "devided": "divided",
    "divede": "divided",
    "thaa": "the",
    "theere": "there",
    "doveded": "divided",
    "truiangle": "triangle",
    "anwsers": "answers",
    "shaeed": "shaded",
    "thrr": "three",
    "therr": "there",
    "simpfiy": "simplify",
    "equavilant": "equivalent",
    "denomenatar": "denominator",
    "trianglrs": "triangles",
    "simplifyd": "simplified",
    "triagnal": "triangle",
    "devidedd": "divided",
    "equevalant": "equivalent",
    "duidid": "divided",
    "equll": "equal",
    "dievided": "divided",
    "fomr": "from",
    "trnagled": "triangle",
    "thiing": "thing",
    "thaaa": "that",
    "togeher": "together",
    "shasded": "shaded",
    "nintth": "ninth",
    "devider": "divider",
    "thei": "the",
    "couloured": "colored",
    "piramyd": "pyramid",
    "becauss": "because",
    "deveid": "divide",
    "equeel": "equal",
    "thast": "that's",
    "denomonata": "denominator",
    "therees": "there's",
    "dedvided": "divided",
    "equale": "equal",
    "simpilfiy": "simplify",
    "denomaniter": "denominator",
    "whicc": "which",
    "devid": "divide",
    "denomaatar": "denominator",
    "formm": "form",
    "andorator": "numerator",
    "numertao": "numerator",
    "fromo": "from",
    "simiplfy": "simplify",
    "simplayfied": "simplified",
    "anwsere": "answer",
    "numeator": "numerator",
    "therer": "there",
    "thatre": "there",
    "becuasee": "because",
    "devideded": "divided",
    "tthat": "that",
    "thtee": "the",
    "whiter": "white",
    "devidrd": "divided",
    "whe": "the",
    "simlifyy": "simplify",
    "numenator": "numerator",
    "coulorred": "colored",
    "divieded": "divided",
    "numbertor": "numerator",
    "simpfiyed": "simplified",
    "becauee": "because",
    "euaql": "equal",
    "thaat": "that",
    "denomenator": "denominator",
    "numoretor": "numerator",
    "ther": "there",
    "siplyed": "simplified",
    "denomerator": "denominator",
    "becuasr": "because",
    "diveded": "divided",
    "trianglle": "triangle",
    "simiplyfed": "simplified",
    "simpplify": "simplify",
    "simplifies": "simplifies",
    "didvided": "divided",
    "eqivalent": "equivalent",
    "therefour": "therefore",
    "itca": "it",
    "trianges": "triangles",
    "thatty": "that",
    "equeal": "equal",
    "whith": "with",
    "thatr": "that",
    "becusee": "because",
    "triangles": "triangles",
    "truangle": "triangle",
    "equavelent": "equivalent",
    "numberater": "numerator",
    "thatt": "that",
    "alther": "altogether",
    "that's": "that's",
    "thrrrd": "three",
    "thatt": "that",
    "thaa": "tha",
    "becuasee": "because",
    "couler": "color",
    "theam": "them",
    "wll": "wall",
    "brlieve": "believe",
    "nugative": "negative",
    "negetive": "negative",
    "mske": "make",
    "pluss": "plus",
    "othe": "other",
    "addd": "add",
    "anegativge": "a negative",
    "takeaway": "take away",
    "rosy": "likely",
    "thsn": "than",
    "yiu": "you",
    "smybols": "symbols",
    "snswer": "answer",
    "sdd": "add",
    "thay": "they",
    "minuses": "minuses",
    "numbrrs": "numbers",
    "equivulant": "equivalent",
    "aawnser": "answer",
    "mimus": "minus",
    "iut": "out",
    "kmow": "know",
    "bidmas": "BODMAS",
    "bracets": "brackets",
    "esposise": "opposite",
    "thinkk": "think",
    "opositivity": "positivity",
    "pov": "possible",
    "plusminus": "plus minus",
    "addminus": "add minus",
    "guves": "gives",
    "poditive": "positive",
    "a plusnumber": "a plus number",
    "numberss": "numbers",
    "posotive": "positive",
    "postitive": "positive",
    "matgches": "matches",
    "minsus": "minus",
    "possitive": "positive",
    "mispelled": "misspelled",
    "bracts": "brackets",
    "minusis": "minuses",
    "equalelles": "equals",
    "minuss": "minus",
    "resive": "receive",
    "revrrse": "reverse",
    "us": "is",
    "subtractingg": "subtracting",
    "subtraction": "subtraction",
    "thy": "the",
    "lets": "let's",
    "gooooo": "go",
    "equall": "equal",
    "opistie": "opposite",
    "opus": "opposite",
    "equelles": "equals",
    "equasion": "equation",
    "eqyals": "equals",
    "posative": "positive",
    "postive": "positive",
    "togther": "together",
    "number’s": "numbers",
    "negativee": "negative",
    "nagiteve": "negative",
    "navitateve": "negative",
    "takr": "take",
    "twoo": "two",
    "breackets": "brackets",
    "squeezing": "squeezing",
    "minise": "minus",
    "numberr": "number",
    "yinn": "yin",
    "twenties": "twenties",
    "Fir": "for",
    "otherr": "other",
    "aree": "are",
    "number’s": "number's",
    "similarly": "similarly",
    "unto": "into",
    "thim": "think",
    "othes": "others",
    "othr": "other",
    "negarive": "negative",
    "becausee": "because",
    "becaude": "because",
    "prussian": "the sum",
    "toook": "took",
    "numb": "number",
    "mutiple": "multiple",
    "num": "number",
    "er": "",
    "brr": "",
    "mrr": "",
    "ssme": "same",
    "twos": "two's",
    "tghther": "together",
    "negitive": "negative",
    "cuz": "because",
    "u": "you",
    "evry": "every",
    "evrytime": "every time",
    "didnt": "didn't",
    "patten": "pattern",
    "therum": "therum",
    "conegts": "connects",
    "numbrs": "numbers",
    "eqils": "equals",
    "eachtime": "each time",
    "goung": "going",
    "kid": "kind",
    "witch": "which",
    "0ne": "one",
    "plud": "plus",
    "drcimal": "decimal",
    "love": "leave",
    "pice": "piece",
    "outt": "out",
    "biger": "bigger",
    "kinda": "kind",
    "you':": "you're",
    "kidd": "kid",
    "ps": "PS",
    "im": "I'm",
    "badd": "bad",
    "hitt": "hit",
    "numeroter": "numerator",
    "donomonator": "denominator",
    "desimall": "decimal",
    "reakly": "really",
    "interger": "integer",
    "doess": "does",
    "everythig": "everything",
    "snswer": "answer",
    "greatrst": "greatest",
    "itf": "it",
    "fith": "fifth",
    "themselvea": "themselves",
    "krep": "keep",
    "numorator": "numerator",
    "someting": "something",
    "denomenator": "denominator",
    "tomes": "times",
    "iut": "out",
    "15ths": "fifteenths",
    "multiplie": "multiply",
    "denomenator": "denominator",
    "multiplie": "multiply",
    "numeroter": "numerator",
    "desimal": "decimal",
    "thenth": "tenth",
    "domonator": "denominator",
    "nefore": "before",
    "commom": "common",
    "dinominater": "denominator",
    "ad": "and",
    "mutiplay": "multiply",
    "nuber": "number",
    "cangd": "changed",
    "bottam": "bottom",
    "andadd": "and add",
    "demonator": "denominator",
    "demonatoru": "denominator you",
    "dinominator": "denominator",
    "demonatorwe": "denominator we",
    "ever": "every",
    "bottamso": "bottoms so",
    "anwer": "answer",
    "dinominators": "denominators",
    "dnomnatiors": "denominators",
    "dominator": "denominator",
    "determiners": "denominators",
    "lcm": "LCM",
    "demoniator": "denominator",
    "commen": "common",
    "demoninator": "denominator",
    "coman": "common",
    "ind": "find",
    "covert": "convert",
    "numirators": "numerators",
    "bedt": "best",
    "donominator": "denominator",
    "andwer": "answer",
    "toget": "to get",
    "numaratores": "numerators",
    "u": "you",
    "convert": "converted",
    "numeratores": "numerators",
    "numerater": "numerator",
    "multipal": "multiple",
    "denuminators": "denominators",
    "denoter": "denominator",
    "tom": "top",
    "woukd": "would",
    "enumerable": "numerators",
    "comon": "common",
    "dud": "did",
    "dominate": "denominator",
    "thrn": "then",
    "is equall": "is equal",
    "3+5'is": "3+5 is",
    "fuve": "five",
    "denomter": "denominator",
    "numberators": "numerators",
    "3and": "3 and",
    "numbersthat": "number that",
    "denimator": "denominator",
    "whixh": "which",
    "weree": "were",
    "miltiple": "multiple",
    "dominate": "denominator",
    "dominotars": "denominators",
    "demorator": "denominator",
    "denomimator": "denominator",
    "tah": "that",
    "donominate": "denominator",
    "denometor": "denominator",
    "timess ": "times",
    "th": "the",
    "domminater": "denominator",
    "numberatour": "numerator",
    "des": "does",
    "cimmon": "common",
    "multiples": "multiple",
    "deominators": "denominators",
    "denomiter": "denominator",
    "fithteen": "fifteen",
    "equel": "equal",
    "denotinar": "denominator",
    "enumenators": "numerators",
    "denomter": "denominator",
    "denomater": "denominator",
    "newmaratores": "numerators",
    "don't": "do not",
    "dint": "different",
    "nominators": "numerators",
    "cimmom": "common",
    "drnominator": "denominator",
    "woule": "would",
    "numberator": "numerator",
    "vlosest": "closest",
    "3++5": "3 + 5",
    "numberating": "numerator",
    "thst": "that",
    "3or": "3 or",
    "1;3": "1/3",
    "3;15": "3/15",
    "plud": "plus",
    "numeratour": "numerator",
    "domminator": "denominator",
    "thi": "the",
    "throm": "from",
    "tooooo": "too",
    "denomenater": "denominator",
    "sooooo": "so",
    "wll": "wall",
    "nugative": "negative",
    "pluss": "plus",
    "them": "them",
    "smybols": "symbols",
    "aawnser": "answer",
    "bracets": "brackets",
    "opositivity": "positivity",
    "negetive": "negative",
    "a plusnumber": "a plus number",
    "postitive": "positive",
    "matgches": "matches",
    "mines": "minus",
    "minsus": "minuses",
    "resive": "receive",
    "numberating": "numerator",
    "numoretor": "numerator",
    "simplier": "simpler",
    "thee": "the",
}


def replace_misspelled_words(documents, mispelled):
    # Create a single regex pattern for all misspelled words
    sorted_misspellings = sorted(mispelled.keys(), key=len, reverse=True)
    pattern = r'\b(?:' + '|'.join(re.escape(word)
                                  for word in sorted_misspellings) + r')\b'

    # Compile the regex pattern once for efficiency
    regex = re.compile(pattern)

    # Create replacement function that looks up the correct spelling
    def replacer(match):
        misspelled_word = match.group(0)
        # Look up correct spelling in dictionary
        return mispelled[misspelled_word]

    # Process each document
    corrected_documents = []
    for document in documents:
        corrected_doc = regex.sub(replacer, document)
        corrected_documents.append(corrected_doc)

    return corrected_documents


def fix_question_text_89443(df: pd.DataFrame) -> pd.DataFrame:

    new_question_text = """
Question: What number belongs in the box?
\((-8)-(-5)=x\)
"""

    df.loc[df['QuestionId'] == 89443, 'QuestionText'] = new_question_text

    df['QuestionText'] = df['QuestionText'].apply(lambda x: x.strip())

    return df


def fix_question_category_false_neither_31778(df: pd.DataFrame) -> pd.DataFrame:

    df.loc[(df['QuestionId'] == 31778) & (df['MC_Answer'] == "\( 6 \)") & (
        df['Category'].str.contains("False")), 'Category'] = 'True_Neither'

    return df


def fix_question_category_wrong_31778(df: pd.DataFrame) -> pd.DataFrame:

    # Row: 14280
    # Category: True_Neither
    # Explanation: Because 10 is 2 / 3 of 15, and 2 is 6.
    # Possibly Misconception Incomplete?
    df.loc[(df['row_id'] == 14280), 'MC_Answer'] = '\( 6 \)'
    df.loc[(df['row_id'] == 14280), 'Category'] = 'True_Misconception'
    df.loc[(df['row_id'] == 14280), 'Misconception'] = 'Irrelevant'

    # Row: 14305
    # Category: True_Correct
    # Explanation: I divided 9/15 by 3, then got 3/5 and timsed it by 2 and got 6/10.
    df.loc[(df['row_id'] == 14305), 'MC_Answer'] = '\( 6 \)'

    # Row: 14321
    # Category: True_Correct
    # Explanation: I think it's C because 6/10 is the same as 9/15.
    df.loc[(df['row_id'] == 14321), 'MC_Answer'] = '\( 6 \)'

    # Row: 14335
    # Category: True_Correct
    # Explanation: Il believe that is the ansewer because I calculatted iti.
    df.loc[(df['row_id'] == 14335), 'Category'] = 'False_Neither'

    # Row: 14338
    # Category: True_Correct
    # Explanation: It is six because they are both equal to 3over5.
    df.loc[(df['row_id'] == 14338), 'MC_Answer'] = '\( 6 \)'

    # Row: 14352
    # Category: True_Correct
    # Explanation: To get a denominator of 10, we need to divide by 3 and multiply by 2. Then, 9/15=3/5=6/10, so A = 6.
    df.loc[(df['row_id'] == 14352), 'MC_Answer'] = '\( 6 \)'

    # Row: 14355
    # Category: True_Neither
    # Explanation: You have to change the denominator to 150 then you will get the answer.
    # Possibly Misconception Incomplete?
    df.loc[(df['row_id'] == 14355), 'Category'] = 'False_Neither'

    # Row: 14403
    # Category: True_Correct
    # Explanation: i think this is because 9/15=18/30 and 6/10 =18-30.
    df.loc[(df['row_id'] == 14403), 'MC_Answer'] = '\( 6 \)'
    df.loc[(df['row_id'] == 14403), 'Category'] = 'True_Misconception'
    df.loc[(df['row_id'] == 14403), 'Misconception'] = 'WNB'

    # Row: 14407
    # Category: False_Neither
    # Explanation: if you simplify it to 3/5 then you get 9/15.
    df.loc[(df['row_id'] == 14407), 'Category'] = 'False_Neither'

    # Row: 14412
    # Category: True_Neither
    # Explanation: since 9 - 3 = 6h are so i't must be these ohne!
    df.loc[(df['row_id'] == 14412), 'Category'] = 'False_Misconception'
    df.loc[(df['row_id'] == 14412), 'Misconception'] = 'WNB'

    # Row: 14413
    # Category: True_Correct
    # Explanation: so the common denominator is 30 and the product of 15x2=30 so 9x2, which is a multiple of 9, is 18 and 10x3 =30, so ax3, which we know is 18, is 18. therefore, a=6.
    df.loc[(df['row_id'] == 14413), 'MC_Answer'] = '\( 6 \)'

    # Row: 14418
    # Category: True_Misconception
    # Explanation: this is because the top numbers go up in threes, and the bottom number go down in fives.
    df.loc[(df['row_id'] == 14418), 'MC_Answer'] = '\( 6 \)'

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    _df = df.copy()

    # Create 'NA' Misconception
    _df = _df.fillna("NA")

    # Clean Data
    _df['StudentExplanation'] = _df['StudentExplanation'].apply(
        lambda x: x.strip().strip("."))
    _df['FixedStudentExplanation'] = replace_misspelled_words(
        df['StudentExplanation'].tolist(), misspelled)
    _df = fix_question_text_89443(_df)
    _df = fix_question_category_false_neither_31778(_df)
    _df = fix_question_category_wrong_31778(_df)

    _df['Correct'] = _df.Category.apply(
        lambda x: True if 'True' in x else False)

    # Remove unnecessary duplicates
    _df = _df.drop_duplicates(subset=['QuestionId', 'MC_Answer', 'FixedStudentExplanation',
                              'Category', 'Misconception', 'Correct']).reset_index(drop=True)

    return _df
