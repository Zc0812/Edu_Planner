# Using CIDPP to evaluate generated lesson plan

from pre_prompt import client
def CIDPP_eval(lesson_plan: str)->str:
    CIDPP_prompt = f"""# Role: You are an impartial evaluator, experienced in educational content analysis and instructional design evaluation.

## Attention: You are responsible for assessing the quality of a given instructional design based on five specific evaluation criteria. Your evaluation should be objective and based solely on the Evaluation Standard provided below. 

## Lesson Plan:
 {lesson_plan} 
 
## Evaluation Standard: 
- [A] Clarity: The lesson plan's directness and simplicity, ensuring it avoids unnecessary complexity and redundancy. Integrity: Whether the lesson plan is complete and systematic, covering both knowledge point explanations and exercise explanations in a complementary manner. 
- [B] Depth: The ability of the lesson plan to inspire deep thinking and facilitate understanding of the underlying connections between knowledge points. 
- [C] Practicality: The practical application value of the examples in the lesson plan, ensuringstudents can use the knowledge to solve real-life problems.
- [D] Pertinence: The adaptability of the lesson plan to different students’ knowledge levels and learning needs to achieve optimal learning outcomes. 
- [E] Constraints: Avoid any bias in evaluation based on the content’s length or appearance. Be as objective as possible in assessing each aspect individually without favoring any specific structure or terminology. \

## Work flow: 
Output your final verdict in the following format:"[A]:[0-100 points]; [short analyzes]", "[B]: [0-100 points]; [short analyzes]", "[C]: [0-100 points]; [short analyzes]", "[D]: [0-100 points]; [short analyzes]", "[E]: [0-100 points]; [short analyzes]". 
Take a deep breath and think step by step!
"""
    response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": CIDPP_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                )
    return response.choices[0].message.content