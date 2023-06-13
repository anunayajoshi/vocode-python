from vocode.streaming.agent.base_agent import RespondAgent
from typing import Optional, AsyncGenerator, Tuple
import logging
from vocode.streaming.models.agent import AgentConfig
from langchain import PromptTemplate
from langchain.chains import ConversationChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
import os
import yaml

OPEN_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = "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"

class ChainAgent(RespondAgent[AgentConfig]):
    def __init__(self, agent_config: AgentConfig):
        super().__init__(agent_config)
        yaml_file = open("PROMPTS.yaml", 'r')
        self.yaml_content = yaml.load(yaml_file,Loader=yaml.Loader)
        self.smallTalkRun_promptTemplate = PromptTemplate(
            input_variables=["name","location"],
            template=self.yaml_content['PROMPT']['smallTalkTemplate']
        )
        self.smallTalk_iteration = self.yaml_content['PROMPT']['smallTalk_iteration']
        self.questionExtender_iteration = self.yaml_content['PROMPT']['questionExtender_iteration']
        self.count_iterations = 0
        

        self.smallTalkAgent, self.smallTalkAgentMemory= self.smallTalk(
            prompt_template=PromptTemplate(
            input_variables=["name","location"],
            template=self.yaml_content['PROMPT']['smallTalkTemplate']
        ),
        variable_values={"name": self.yaml_content['PROMPT']['candidateName'], "location": self.yaml_content['PROMPT']['candidateLocation']},
        )



    def smallTalk(self, prompt_template, variable_values, initial_message="Let's think step by step"):

        agentMemory = ConversationBufferMemory()
        agentMemory.chat_memory.add_ai_message(initial_message)

        llm = ChatOpenAI(temperature=0,request_timeout=120)
        smallTalk_agent = ConversationChain(llm=llm, verbose=False, memory=agentMemory)

        smallTalk_agent.prompt.template = (
            prompt_template.format(**variable_values)
            + "\n\nCurrent conversation:\n{history}\nCandidate: {input}\nBot: "
        )
        return smallTalk_agent, agentMemory

    def smallTalkRun(self, message,max_interactions=3):
        response = self.smallTalkAgent.predict(input=message)
        if self.count_iterations == max_interactions - 1:
            self.smallTalkAgentMemory.chat_memory.add_ai_message(
                    """After this, move the conversation and say these words to the candidate - You will be asked questions related to your role. Please answer them carefully, as they will be evaluated.
                    (Note: Do not combine this with any other questions.)"""
                )
        self.count_iterations += 1
        return response


    def questionExtender(self, Q, A,template):
        davinci = OpenAI(model_name="text-davinci-003", openai_api_key=OPEN_API_KEY)
        long_prompt = PromptTemplate(
            template=template, input_variables=["question", "answer"]
        )
        llm_chain = LLMChain(prompt=long_prompt, llm=davinci)
        return llm_chain.run({"question": Q, "answer": A})


    def askQuestions(self,
            name, 
            bio,
            workEx,
            questionsList,
            askQuestionsTemplate,
            questionExtenderTemplate,
            QuestionExtenderIteration=3,
            ):
        
        QuestionExtenderIteration = QuestionExtenderIteration + 1
        max_interactions = len(questionsList)

        promptTemplate = PromptTemplate(
            input_variables=["name", "bio", "workEx", "question"],
            template=askQuestionsTemplate,
        )

        llm = ChatOpenAI(temperature=0, request_timeout=120)
        chain = LLMChain(llm=llm, prompt=promptTemplate)

        for question_index, question in enumerate(questionsList):
            # if question_index != 0:
            ext_iter = 1
            print(name, bio, workEx)
            res_ques = chain.run(
                {"name": name, "bio": bio, "workEx": workEx, "question": question}
            )
            # print(ext_iter,question_index,question)
            if QuestionExtenderIteration != 0:
                while ext_iter <= QuestionExtenderIteration:
                    if ext_iter > QuestionExtenderIteration or ext_iter == 1:
                        print("\n\nKwalBot ", res_ques)
                    else:
                        print(
                            "\nKwalAI Extended Question: ", res_ext.strip()
        # smallTalk_iteration = 2
                        )  # Add this to see the output of the extended question.

                    input_text = input("User==>")
                    res_ext = self.questionExtender(Q=res_ques, A=input_text,template=questionExtenderTemplate)
                    ext_iter += 1
            else:
                print("\n\nKwalBot ", res_ques)
    
    async def respond(self, message, conversation_id: str, is_interrupt: bool = False):
        if self.smallTalk_iteration != 0:
            return self.smallTalkRun(
                message,
                max_interactions=self.smallTalk_iteration,
            ), False
        
    #######################################################
    # print(yaml_content['PROMPT']['askQuestionsTemplate'])
        # QUESTIONS  = self.yaml_content['PROMPT']['DEFAULT_QUESTION']

        self.askQuestions(name=self.yaml_content['PROMPT']['candidateName'],
                    bio=self.yaml_content['PROMPT']['candidateBio'],
                    workEx=self.yaml_content['PROMPT']['candidateWorkEx'],
                    askQuestionsTemplate=self.yaml_content['PROMPT']['askQuestionsTemplate'],
                    questionsList=self.yaml_content['PROMPT']['DEFAULT_QUESTION'],
                    questionExtenderTemplate=self.yaml_content['PROMPT']['questionExtenderTemplate'],
                    QuestionExtenderIteration=self.questionExtender_iteration)
    
    async def generate_response(self, human_input, conversation_id: str, is_interrupt: bool = False) -> AsyncGenerator[str, None]:
        yield human_input



