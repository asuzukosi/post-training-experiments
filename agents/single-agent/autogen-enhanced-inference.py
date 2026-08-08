"""
Autogen framework provides a drop in replaceemnt
for the inference sdk used in openai

The cost of using foundation models for text generation is typically measured in terms of 
the number of tokens in the input and output combined. 
"""
from typing import List, Dict


def eval_math_responses(responses: List[str], solution: str, **args) -> Dict:
    answer = voted_answer(responses)
    return {"success": is_equivalent(answer, solution)}