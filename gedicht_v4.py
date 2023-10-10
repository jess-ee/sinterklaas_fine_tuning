#Importing dependencies
import os 
import langchain
import streamlit as st 
import time
import re
import requests
import pandas as pd
from io import BytesIO
import traceback

from langchain.callbacks import LangChainTracer
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessage
)

from langchain.chains import LLMChain
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.schema.runnable import RunnableConfig


from langsmith import Client
from streamlit_feedback import streamlit_feedback


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "gedicht-finetuning"


apikey = os.getenv('OPENAI_API_KEY')

client = Client(api_url=os.environ["LANGCHAIN_ENDPOINT"], api_key=os.environ["LANGCHAIN_API_KEY"])

run_collector = RunCollectorCallbackHandler()
runnable_config = RunnableConfig(
    callbacks=[run_collector],
    tags=["Streamlit Chat"],
)

product_df = pd.read_csv('product_information_small.csv')


with open('Goodtraits.csv', 'r') as f:
    goodtraits_options = f.read().splitlines()
with open('badtraits.csv', 'r') as f:
    badtraits_options = f.read().splitlines()

#App framework
st.title('Coolblue Sinterklaas gedichten ✍️')

st.markdown("""
    Welkom bij de Coolblue Sinterklaas gedichten generator!

    """)

name = st.text_input('Dit cadeau is voor?')
goodtrait = st.multiselect('Is goed in... (selecteer er 2)', goodtraits_options, max_selections=2)
badtrait= st.multiselect('Is slecht in...? (selecteer er 2)',badtraits_options,max_selections=2)
product_id = st.text_input('Vul het Product ID in')

#CSV retrieval 

# Initialize variables
product_info_row = pd.DataFrame()
product_type_name = ""
product_info_text = ""

if product_id:
    try:
        # Convert the Product ID to an integer
        product_id_int = int(product_id)
        
        # Lookup product information
        product_info_row = product_df[product_df['product_id'] == product_id_int]

    except ValueError:
        st.error("Please enter a valid integer for Product ID.")

# If a matching product is found, extract and use its information
if not product_info_row.empty:
    product_type_name = product_info_row['producttype_name_singular'].values[0]
    product_info_text = ', '.join(product_info_row[['product_pro', 'product_pro_2', 'product_pro_3']].values[0])


#Chatmodel 

chat_model= ChatOpenAI(temperature=0.6, model="gpt-4")

#Prompt template

system_message_prompt = SystemMessagePromptTemplate.from_template("""Je bent een vindingrijke en geestige dichter die is ingehuurd door Coolblue om klanten te helpen de traditie van Sinterklaas te vieren met gepersonaliseerde gedichten.

#instructie
- Baseer de gedichten op de verstrekte informatie over de klant en het product dat ze hebben gekocht.
- Zorg ervoor dat het gedicht grappig, positief, en blij is. Geef geen details over het product weg, maar hint op een speelse manier naar de eigenschappen ervan.
- Laten we stap voor stap nadenken om er voor te zorgen dat het gedicht rijmt

#format
Begin elk gedicht altijd met "Beste [naam]," en zorg dat het gedicht uit 8 zinnen bestaat.

#veiligheid
Als iemand een beledigende naam invoert, antwoord dan met "Sodemijter! Dit gedicht kunnen we niet genereren. Even je mond spoelen en dan nog een keer proberen."


""")
human_message_prompt = HumanMessagePromptTemplate.from_template("""Informatie over de klant:
- Naam: {name}
- Is goed in: {goodtrait}
- is slecht in: {badtrait}

Informatie over het product:
- {product_type_name}
- {product_info_text}
""")
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

#LLM Chain

gedicht_chain = LLMChain(llm=chat_model, prompt=chat_prompt, verbose = True)

# show stuff
if st.session_state.get('previous_output') and st.session_state.get('run_id'):
    run_id = st.session_state['run_id']  # Use the stored run_id
if st.button('Vraag G-Piet-R om een gedicht!'):
        
    response = gedicht_chain.invoke(input={
        "name": name,
        "goodtrait": ','.join(goodtrait),
        "badtrait": ','.join(badtrait), 
        "product_type_name": product_type_name,
        "product_info_text": product_info_text,
    },
    config=runnable_config
    )
    
    run=run_collector.traced_runs[0]
    st.session_state['response'] = response
    st.session_state['run_id'] = run_collector.traced_runs[0].id
    run_collector.traced_runs = []
if st.session_state.get('response'):
    poem_text = st.session_state['response'].get('text', 'Poem not generated.')
    st.text(poem_text)
if st.session_state.get('run_id'):
    with st.form(key='feedback_form'):
        # Modify the options for the rating radio button
        rating = st.radio("Reeting:", [(0, "niet correct"), (1, "correct")])
        
        # When the user presses the Submit button, record the feedback
        if st.form_submit_button('Submit'):
            feedback_record = client.create_feedback(
                st.session_state['run_id'],
                f"score_{rating[0]}",  # Use the first element of the tuple for the score
                score=rating[0],  # Use the first element of the tuple for the score
                # comment=feedback.get("text"),
            )
            st.write('feedback recorded')
