import os
from operator import itemgetter
from typing import List, Tuple

from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import format_document
from langchain_community.vectorstores import FAISS, Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone as PineconeClient

load_dotenv()

# Keys
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_ENVIRONMENT = os.environ['PINECONE_ENVIRONMENT']
PINECONE_INDEX_NAME = os.environ['PINECONE_INDEX_NAME']
PINECONE_HOST = os.environ['PINECONE_HOST']

# Vectorstore & Retriever
pinecone = PineconeClient(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(name=PINECONE_INDEX_NAME, host=PINECONE_HOST)
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone(index, embeddings, 'text')
retriever = vectorstore.as_retriever()

# Availability Embed
availabilityVectorstore = FAISS.from_texts(
	['Available April 2024'], embedding=OpenAIEmbeddings()
)
availabilityRetriever = availabilityVectorstore.as_retriever()

# Languages Embed
languagesVectorstore = FAISS.from_texts(
	[
		'Figma, HTML, CSS, React, XState, Redux, Tailwind CSS, Shadcn/UI, Headless UI, CSS-in-JS, TypeScript, JavaScript, Storybook, Next.js, Remix, React Router, React Testing Library, Zod, Jest, Cypress, Playwright, Node JS, Express, NPM, GraphQL, REST API, SOAP, Prisma, Drizzle, Postgres, MySQL, MongoDB, AWS, AWS Amplify, AWS Serverless, Lambda, Cognito, Auth.JS, Vercel, Fly.io, Docker, CI/CD, GIT, VSCode, Slack, LangChain, AI SDK, Open AI, Hugging Face, SendBird, XMPP, Open Fire, JWT, React Native, iOS, Android, Expo, Fastlane, Python, .Net'
	],
	embedding=OpenAIEmbeddings(),
)
languagesRetriever = languagesVectorstore.as_retriever()

_TEMPLATE = """Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)


# RAG prompt
template = """You are an AI designed to emulate the thoughts and views of Dave Hudson. Your responses should be in the first person, as if Dave himself is speaking. Use phrases like "In my view..." or "I believe...".
	Your responses should be based solely on the context provided, which includes Dave's blog posts and his thoughts on various topics.
	If the question asks about a programming languages that is not in {languages} then your response should always containt the text "I am not famililar with that language".
    If a question is asked that cannot be answered based on the context, respond with "I'm sorry, I don't have any views on that topic yet. Please feel free to email me at dave@applification.net for further discussion."
	If a question is asked about a full time job, respond with "I am an I.T contractor operating outside of IR35, full-time employment is not of interest to me at this time."
	If a question is asked about day rate, respond with "My day rate depends on the specific requirements of the contract."
	Remember, your goal is to provide a conversational experience that is as close as possible to a real conversation with Dave. Do not invent or assume any views that are not explicitly stated in the context.
	Context: {context}
	question: {question}
	If the question asks about availability then your response should include {currentAvailability} 
	answer:
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

# DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template='{page_content}')


# def _combine_documents(
# 	docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator='\n\n'
# ):
# 	"""Combine documents into a single string."""
# 	doc_strings = [format_document(doc, document_prompt) for doc in docs]
# 	return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
	"""Format chat history into a string."""
	buffer = ''
	for dialogue_turn in chat_history:
		human = 'Human: ' + dialogue_turn[0]
		ai = 'Assistant: ' + dialogue_turn[1]
		buffer += '\n' + '\n'.join([human, ai])
	return buffer


# LLM
model = ChatOpenAI(temperature=0, model='gpt-4-1106-preview')

_inputs = RunnableMap(
	standalone_question=RunnablePassthrough.assign(
		chat_history=lambda x: _format_chat_history(x['chat_history']),
	)
	| CONDENSE_QUESTION_PROMPT
	| ChatOpenAI(temperature=0)
	| StrOutputParser(),
)
_context = {
	# 'context': itemgetter('standalone_question') | retriever | _combine_documents,
	'context': itemgetter('standalone_question') | retriever,
	'currentAvailability': itemgetter('standalone_question') | availabilityRetriever,
	'languages': itemgetter('standalone_question') | languagesRetriever,
	'question': lambda x: x['standalone_question'],
}


# LCEL Chain
class ChatHistory(BaseModel):
	"""Chat history with the bot."""

	chat_history: List[Tuple[str, str]] = Field(
		...,
		extra={'widget': {'type': 'chat', 'input': 'question'}},
	)
	question: str


conversational_qa_chain = (
	_inputs | _context | ANSWER_PROMPT | ChatOpenAI() | StrOutputParser()
)
chain = conversational_qa_chain.with_types(input_type=ChatHistory)
