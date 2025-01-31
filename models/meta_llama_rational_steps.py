from groq import Groq
from dotenv import dotenv_values


class MetaLlamaRationalSteps:

    def __init__(self, max_tokens=1200):
        CONFIG = dotenv_values("config/.env")

        self.client = Groq(api_key=CONFIG["GROQ_API_KEY"])
        self.model_name = CONFIG["MODEL_NAME"]
        self.max_tokens = max_tokens
    
    def question_steps_answer(self, question, system_prompt="Answer the following question"):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"{system_prompt}"
                },
                {
                    "role": "user",
                    "content": f"{question}"
                }
            ],
            model=self.model_name,
            max_tokens=self.max_tokens
        )
        answer = response.choices[0].message.content

        try:
            rational_step = answer.split("</think>")[0].replace("<think>", "")
            final_answer = answer.split("</think>")[1].replace("<answer>", "").replace("</answer>", "").replace("\n", "")
        except:
            rational_step = "format error"
            final_answer = "format error"

        return rational_step, final_answer