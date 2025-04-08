import os
import httpx
from openai import OpenAI
from util import *

# 定义超参数
N = 20 # optimization num
T = 8 # test_question num
M = 5 # 并行测试lesson plan分数的次数
K = 3 # lesson plan num per optimization
P = 3 # max lesson plan num in optimization prompt

# 文件路径
ROOT_DIR = os.getcwd()
DATASET_PATH = os.path.join(ROOT_DIR, "datasets/algebra222.csv")
ABILITY_TREE_PATH = os.path.join(ROOT_DIR, "persona/ability_tree.json")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["TRANSFORMERS_CACHE"] = os.path.join(ROOT_DIR, "models")

API_KEY = "your api key"
BASE_URL = "your base url"

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL

# client = OpenAI(
# http_client=httpx.Client(
#     base_url=BASE_URL,
#     follow_redirects=True,
#     ),
# )

client = OpenAI(
  base_url = BASE_URL,
  api_key = API_KEY
)

initial_lesson_plan = """Teaching Plan —— algebraic equation
# Part 1: Explanation of knowledge points
An algebraic equation is a mathematical equation containing unknown numbers, and its general form is ax+b=c. Among them, a, b, c are known numbers, and x is an unknown number. The process of solving an algebraic equation is to find the value of x that makes the equation true.

The basic methods for solving algebraic equations are as follows:
- Moving term method: According to the properties of the equation, move the unknown number to one side and the known number to the other side until the unknown number is independent.
- Equivalent deformation method: Through equivalent deformation of the equation, the equation is reduced to a simpler form, and finally the value of the unknown is obtained.
- Substitution method: Substituting known numbers into the equation to find the value of the unknown.
- Elimination method: For a system of equations containing two unknowns, one of the unknowns is eliminated through appropriate deformation, thereby transforming it into an equation containing only one unknown.
- Factoring: Factor the equation to find the value of the unknown.

In the process of solving algebraic equations, you need to pay attention to maintaining a balance on both sides of the equation and ensuring that the transformations at each step are equivalent to avoid introducing new errors.

# Part 2: Explanation of exercise
Question 1:
    A brownie recipe is asking for 350 grams of sugar, and a pound cake recipe requires 270 more grams of sugar than a brownie recipe. How many grams of sugar are needed for the pound cake? 
Solution: 
    Step 1: Identify the amount of sugar needed for the brownie recipe, which is 350 grams. 
    Step 2: Understand that the pound cake recipe requires 270 more grams of sugar than the brownie recipe. 
    Step 3: Add the additional 270 grams of sugar to the 350 grams required for the brownie recipe. 
    Step 4: The total amount of sugar needed for the pound cake recipe is 350 grams + 270 grams = 620 grams.
"""

initial_lesson_plan_score = 20

students = [
    [4, 4, 3, 4, 3]
]

persona = get_students_ability(ABILITY_TREE_PATH, students)

eval_task = f"""# Task
Given the student's ability level, explanation of knowledge points and the exercise explanation the student has received, what's the probability that the student can solve the problem correctly? Explain your reasoning and give a single number between 0 and 100 in square brackets, and the suggestion to optimize the explanation of knowledge points and the exercise explanation to improve the student's evaluation score.
Format：

<|reason_start|>
your reason to explain the evaluation score.
<|reason_end|>

<|score_start|>
[evaluation score]
<|score_end|>

<|suggest_start|>
your suggestion to optimize explanation of knowledge points and the exercise explanation to improve the student's evaluation score.
<|suggest_end|>\n\n"""


nested_string = """            {persona}
            Here's the instruction that the student receives. The student is asked to study a problem and its solution.
            {instruction}
            Now the student is asked to work on the following problem on a test:
            {problem}
            {eval_task}
        """

utility_string = f"""def utility(lesson_plan: str):
    '''
    Evaluates the lesson plan in terms of test performance. Returns final test score of the student.
    '''
    algebra = pd.read_csv(Dataset_Dir)
    questions = algebra["question"]

    selected_questions = questions.sample(n=T)

    messages = ""
    for problem in tqdm(selected_questions, total=T):
        # Here you can use the selected questions for your processing
        evaluation_prompt = f\"\"\"
""" + nested_string + """
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": evaluation_prompt},
                ],
            temperature=0.0,
            max_tokens=1024,
        )
        print(response.choices[0].message.content)
        messages += response.choices[0].message.content

    sum_score = 0

    scores = re.findall(r"\[.*?\]", messages)
    print(scores)
    for score in scores:
        sum_score += float(score[1:-1])
    avg_score = int(sum_score / len(scores))

    return avg_score
    \"\"\"
"""



opti_task = f"""Generate a new lesson plan to further increase the test score of the student. The lesson plan should follow the following rules:
- Teaching topics cannot be changed.
- Keep the lesson plan to have only two parts: knowledge explanation and exercise explanation.
- **Insert questions with new difficulty gradients and explain them.**

You will be evaluated based on this score function:
 '''python
 {utility_string}
 '''
The new lesson plan should begin with <LESSON_PLAN> and end with </LESSON_PLAN>.
"""

common_mistakes_db = f"""1、Transposition Error
2、Calculation Error
3、Algebraic Simplification Error
4、Ignoring Problem Conditions
5、Misinterpretation of the Problem
6、Misapplication of Formulas or Theorems
"""

ana_task = """You need to calculate the three mistakes that students will make in the above example based on their knowledge background and learning ability, and insert them at the end of the example in order of probability from largest to smallest.
- Combined with the question given above
- Incorporate students' background knowledge but don't reveal it in your reponse
- Do not output irrelevant content, such as: note and Here are...
- Responses include only Common Mistakes

Output Example:
Teaching Plan —— algebraic equation
# Part 1: Explanation of knowledge points
xxx

# Part 2: Explanation of exercise
Question 1: xxx

Solution: xxx

Common Mistakes 1:
    1.Transposition Error(50%): xxxx
    2.xxxxx
    ...

Question 2: xxx

Solution: xxx

Common Mistakes 2:
    1.Transposition Error(50%): xxxx
    2.xxxxx
    ...
"""

selected_questions = get_selected_questions(DATASET_PATH, T)
