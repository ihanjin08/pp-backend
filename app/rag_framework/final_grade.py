from openai import OpenAI

def final_grade(feedback, client, criterion, subject):
    # Prompt for GPT-4 with strict JSON output
    prompt = f"""
    You are part of an important IB MYP grading committee responsible for grading students' work based on specific evidence.
    Here is the given feedback per strand:
    {feedback}

    Given this feedback determine what grade the student should get for Criterion {criterion} in {subject}, knowing that each of the working levels is described as such:

    IB Level 1 - 2: Limited quality, lacks understanding of most concepts, rarely/infrequently demonstrates critical thinking, rarely/infrequently demonstrates creative thinking, inflexible, rarely/infrequently applies knowledge and skills.
    IB Level 3 - 4: Acceptable to good quality, basic understanding of concepts, few misunderstandings, some critical or creative thinking, some flexibility, requires some support.
    IB Level 5 - 6: Generally high-quality work, some innovation, confident, good to excellent understanding of concepts, shows critical thinking, creative, uses knowledge and skills frequently, independently.
    IB Level 7 - 8: High-quality work, innovative, extensive understanding of concepts, consistently demonstrates sophisticated critical thinking, creative, independent, transfers knowledge and skills.

    Output the result as a SINGLE INTEGER NUMBER FROM 1-8 (SHOULD CONTAIN NO WORDS OR BOLDED NUMBERS SIMPLY A SINGLE PLAIN NUMBER). """
    
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o-mini",
    )

    response_content = chat_completion.choices[0].message.content
    return response_content