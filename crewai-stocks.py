#Import das libs
import json
import os

from langchain.tools import Tool
import yfinance as yf
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
import streamlit as st

def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
    return stock

yahoo_finance_tools = Tool(
    name = "Yahoo Finance Tool",
    description = "Fetches stocks prices for {ticket} from the last year about a specific company from Yahoo Finance API",
    func= lambda ticket: fetch_stock_price(ticket)
)

os.environ['OPENAI_API_KEY'] = "Adicione sua chave da OpenAI"
llm = ChatOpenAI(model="gpt-3.5-turbo")

stockPriceAnalyst = Agent(
    role="Senior stock price Analyst",
    goal="Find the {ticket} stock price and analyses trends",
    backstory="""You're a highly experienced in analyzing the price of na specific stock
    and make predictions about its future price.""",
    verbose=True,
    llm= llm,
    max_iter= 5,
    allow_delegation= False,
    memory= True,
    tools=[yahoo_finance_tools]
)

getStockPrice = Task(
    description= "Analyze the stock {ticket} price history and create a trend analyses of up, down or sideways",
    expected_output = """ Speficy the current trend stock price - up, down, or sideways.
    eg. stock='AAPL, price UP'
    """,
    agent= stockPriceAnalyst
)

search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)

newsAnalyst = Agent(
    role="Stock News Analyst",
    goal="""Create a short summary of the market news related to the stock {ticket} company. Speficy the current trend - up, down or sideways with
    the news context. For each request stock asset, speficy a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.""",
    backstory="""You're a highly experienced in analyzing the market trends and news and have tracked assets for more than 10 years.

    You're also master level analyts in the tradicional market and have deep understanding of human psychology.

    You understand news, theirs titles and information, but you look at those with health dose of skepticism.
    You consider also the source of the news articles.
    """,
    verbose=True,
    llm= llm,
    max_iter= 10,
    allow_delegation= False,
    memory= True,
    tools=[search_tool]
)

get_news = Task(
    description= """Take the stock and always include BTC to it (if not request)
    use the search tool to search each one individually.

    Compose the results into a helpfull report.""",
    expected_output = """ A summary of the overall market and one sentence summary for each request asset.
    Include a fear/greed score for each asset based on the news. Use the format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORED>
    """,
    agent= newsAnalyst 
)

stockAnalystWriter = Agent(
    role="Senior Stock Analyst Writer",
    goal="""Analyze the trends price and news and write on insighfull compelling and informative 3 paragraph long newsletter based on the stock report and price trend. """,
    backstory="""You're widely accepted as the best stock analyst in the market. You understand complex concepts and create complelling stories
    and narratives that resonate with wider audiences.

    You understand macro factors and combine multiple theories - eg. cycle theory and fundamental analyses.
    You're able to hold multiple opinions when analyzing anything.
    """,
    verbose=True,
    llm= llm,
    max_iter= 5,
    allow_delegation= True,
    memory= True,
    tools=[search_tool]
)

writeAnalyses = Task(
    description= """Use the stock price trend and the stock news report to create an analyses and write the newsletter about the {ticket} company
    that is brief and highlights the most important points.
    Focus on the stock price trend, news and fear/greed score. What are the near future considerations?
    Include the previous analyses of stock trend and news summary.
""",
    expected_output= """An eloquent 3 paragraphs newsletter formated as markdown in an easy readable manner, It should contain:

    -3 bullets executive summary
    - Introduction - set the overall picture and spike up the interest
    - main part provides the meat of the analysis including the news summary and fear/greed scores
    - summary - key facts and concrete future trend prediction - up, down, or sideways.
    """,
    agent= stockAnalystWriter,
    context= [getStockPrice, get_news]
)

crew = Crew(
    agents= [stockPriceAnalyst, newsAnalyst, stockAnalystWriter],
    tasks= [getStockPrice, get_news, writeAnalyses],
    verbose= 2,
    process= Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15
)

with st.sidebar:
    st.header('Enter the Stock to Research')

    with st.form(key='research_form'):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label = "Run Research")

if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        results= crew.kickoff(inputs={'ticket': topic})

        st.subheader("Result of your research:")
        st.write(results['final_output'] )